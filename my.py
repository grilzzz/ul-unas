from dataset import Dataset
from utils import stft, istft
from model import UL_UNAS
import torch

model = UL_UNAS()
dataset = Dataset(r'D:\audio\clean', r'D:\audio\noise2', 16000*2)


clean, noise, snr = next(iter(dataset))

spec = stft(clean)

magnitude = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
phase = torch.atan2(spec[..., 1], spec[..., 0])

enhanced_magnitude = model(magnitude.unsqueeze(0).unsqueeze(0)).squeeze(0, 1)



# Воссоздаем комплекcное представление в виде двухканального тензора:
real_part = enhanced_magnitude * torch.cos(phase)
imag_part = enhanced_magnitude * torch.sin(phase)

# Собираем тензор с размерностью [..., 2]
spec = torch.stack([real_part, imag_part], dim=-1)

out = istft(spec)

print(clean.shape)
print(out.shape)

print(clean[100:110])
print(out[100:110])