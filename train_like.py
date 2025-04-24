import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import os
import time
import argparse
import json
import torch
import math
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dataset import Dataset, AudioDataset
# from my_gtcrn_v1_6 import GTCRN as student
from discriminator import batch_pesq, MetricDiscriminator
from loss import HybridLoss
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
# from pesq import pesq
import numpy as np
from joblib import Parallel, delayed

def snr_cost(s_estimate, s_true):
    snr = torch.mean(s_true**2, dim=-1, keepdim=True) / \
          (torch.mean((s_true - s_estimate)**2, dim=-1, keepdim=True) + 1e-7)
    num = torch.log(snr)
    denom = torch.log(torch.tensor(10.0, dtype=num.dtype, device=s_estimate.device))
    loss = -10 * (num / denom)
    return loss.mean() 

def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        pesq_score = -1
    return pesq_score

def val_pesq(denoised, clean):
    denoised_pesq_score = Parallel(n_jobs=16)(delayed(cal_pesq)(c, n) for c, n in zip(clean, denoised))
    denoised_pesq_score = np.array(denoised_pesq_score)
    return (denoised_pesq_score).mean()

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)

    device = torch.device('cuda:{:d}'.format(rank))
    last_epoch = -1
    steps = 0
    state_dict_do = None

    student_model = student().to(device)
    state_dict = load_checkpoint("/home/adminus/aosofono/se/gtcrn/my_gtcrn_v1_6/g_best", device)
    student_model.load_state_dict(state_dict['student_model'])
    
    discriminator = MetricDiscriminator().to(device)

    if h.num_gpus > 1:
        student_model = DistributedDataParallel(student_model, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(student_model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    

    trainset = Dataset(a.input_clean_wavs_dir, a.input_noise_wavs_dir, h.segment_size * h.sampling_rate, h.sampling_rate)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    
    path_to_val_mix = '/home/adminus/aosofono/se/training_set/noisy'
    path_to_val_speech = '/home/adminus/aosofono/se/training_set/clean'
    val_noisy_files = [os.path.join(path_to_val_mix, f) for f in os.listdir(path_to_val_mix)]
    val_clean_files = [os.path.join(path_to_val_speech, f) for f in os.listdir(path_to_val_speech)]
    val_dataset = AudioDataset(val_noisy_files, val_clean_files)

    validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    student_model.train()
    discriminator.train()
    val_noisy_pesq = 1.1010355619341134
    best_pesq = 0.0 #1.154 2.3178422994166614-1.1010355619341134 = 1.216806737482548
    loss_func = HybridLoss(h.n_fft, h.hop_size, h.win_size)
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(train_loader)):

            clean_audio, noisy_audio = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))

            noisy_spec = torch.stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, window=torch.hann_window(h.win_size).pow(0.5).cuda(), return_complex=False)
            clean_spec = torch.stft(clean_audio, h.n_fft, h.hop_size, h.win_size, window=torch.hann_window(h.win_size).pow(0.5).cuda(), return_complex=False)
                
            est_spec = student_model(noisy_spec)
            audio_g = torch.istft(est_spec, h.n_fft, h.hop_size, h.win_size, window=torch.hann_window(h.win_size).pow(0.5).cuda())

            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            est_mag = (torch.abs(est_spec) + 1e-10) ** (0.3)
            clean_mag = (torch.abs(torch.view_as_complex(clean_spec)) + 1e-10) ** (0.3)

            # Discriminator
            optim_d.zero_grad()
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, est_mag.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                # print('pesq is None!')
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L2 Complex Loss
            loss_com = loss_func(torch.view_as_real(est_spec), clean_spec) 

            # Time Loss
            loss_time = torch.nn.L1Loss()(audio_g, clean_audio)

            metric_g = discriminator(clean_mag, est_mag)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = loss_com + loss_time * 0.5 + loss_metric * 0.5
            loss_gen_all.backward()
            # utils.clip_grad_norm_(student_model.parameters(), 5)
            # gru_parameters = []
            # # Iterate over all modules in the model
            # for module in student_model.modules():
            #     # Check if the module is a GRU layer
            #     if isinstance(module, torch.nn.GRU):
            #         # Extend the parameter list with parameters from this GRU
            #         gru_parameters.extend(module.parameters())
            # utils.clip_grad_norm_(gru_parameters, max_norm=5)
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        # mag_error = F.mse_loss(clean_mag, est_mag).item()
                        com_error = F.mse_loss(torch.view_as_real(est_spec), clean_spec).item()
                        time_error = snr_cost(audio_g, clean_audio).item()


                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    # sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)


                
                steps += 1
 
        checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
        save_checkpoint(checkpoint_path,
                        {'student_model': (student_model.module if h.num_gpus > 1 else student_model).state_dict()})
        checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
        save_checkpoint(checkpoint_path, 
                        {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                            'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                            'epoch': epoch})
        # Validation
        student_model.eval()
        torch.cuda.empty_cache()
        val_pesq_score = 0
        with torch.no_grad():
            for j, batch in enumerate(tqdm(validation_loader)):
                clean_audio, noisy_audio = batch
                noisy_audio = noisy_audio.cuda()
                # norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0)).cuda()
                # noisy_audio = (noisy_audio * norm_factor)
                noisy_spec = torch.stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, window=torch.hann_window(h.win_size).pow(0.5).cuda(),
                                return_complex=False)
                est_spec = student_model(noisy_spec)
                denoised = torch.istft(est_spec, h.n_fft, h.hop_size, h.win_size, window=torch.hann_window(h.win_size).pow(0.5).cuda())
                clean = clean_audio.numpy()
                # denoised = (denoised/norm_factor)
                denoised = denoised.cpu().numpy()
                pesq_score = val_pesq(denoised,clean)
                val_pesq_score +=pesq_score
            val_pesq_score = val_pesq_score/(j+1)
            val_pesq_score -= val_noisy_pesq
            print('Steps : {:d}, PESQ Score: {:4.3f}'.format(steps, val_pesq_score))
            sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)      

        if val_pesq_score > best_pesq:
            best_pesq = val_pesq_score
            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
            save_checkpoint(best_checkpoint_path,
                        {'student_model': (student_model.module if h.num_gpus > 1 else student_model).state_dict()})

        student_model.train()
        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='/home/adminus/aosofono/my_clean/gt10sec/')
    parser.add_argument('--input_noise_wavs_dir', default='/home/adminus/aosofono/datasets/datasets/')
    parser.add_argument('--checkpoint_path', default='my_gtcrn_v1_6_2')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=500, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=500, type=int)
    parser.add_argument('--validation_interval', default=5, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
