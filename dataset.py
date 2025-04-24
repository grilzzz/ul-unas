import os
import glob
import random
import torch
import torch.utils.data
import librosa
import numpy as np

EPS = np.finfo(float).eps

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio


def dns_snr_mixer(clean, noise, snr):
    target_level=-25
    clipping_threshold=0.99
    print(noise.shape, clean.shape)
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean)-len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise)-len(clean)))
    
    clean = clean/(max(abs(clean))+EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean()**0.5

    noise = noise/(max(abs(noise))+EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean()**0.5

    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    noisyspeech = clean + noisenewlevel
    
    noisy_rms_level = np.random.randint(-35, -15)
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
    return clean, noisyspeech 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_wavs_dir, noise_wavs_dir, segment_size=None, sampling_rate=16000):
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.noise_wavs = glob.glob(os.path.join(noise_wavs_dir, '**', "*.wav"), recursive=True)
        self.clean_wavs = glob.glob(os.path.join(clean_wavs_dir, '**', "*.wav"), recursive=True)
        
    def __getitem__(self, index):
        clean_path = random.choice(self.clean_wavs)
        clean_audio, _ = librosa.load(clean_path, sr=self.sampling_rate)
        noise_path = random.choice(self.noise_wavs)

        noise_audio, _ = librosa.load(noise_path, sr=self.sampling_rate)
        snr = random.randint(-20, 15)
        if self.segment_size:
            max_audio_start = len(clean_audio) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            clean_audio = clean_audio[audio_start: audio_start+self.segment_size]
            noise_audio = noise_audio[audio_start: audio_start+self.segment_size]

        clean_audio, noisy_audio = dns_snr_mixer(clean_audio, noise_audio, snr)
 
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        # norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        # clean_audio = (clean_audio * norm_factor)
        # noisy_audio = (noisy_audio * norm_factor)

        return (clean_audio, noisy_audio, snr)

    def __len__(self):
        return 2
        # return len(self.noise_wavs)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_files, clean_files):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
    
    def __len__(self):
        # return 1602
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean = self.clean_files[idx]
        noisy = self.noisy_files[idx]
        noisy, _ = librosa.load(noisy, sr=16000)
        clean, _ = librosa.load(clean, sr=16000)
        # noisy = noisy[:16000*9-100]
        # clean = clean[:16000*9-100]
        clean, noisy = torch.FloatTensor(clean), torch.FloatTensor(noisy)
        return (clean.squeeze(0), noisy.squeeze(0))
