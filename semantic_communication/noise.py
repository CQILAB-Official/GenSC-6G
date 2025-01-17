import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def add_awgn_noise(signal, snr):
    signal_power = torch.mean(torch.square(signal))
    noise_power = signal_power / snr

    noise = torch.randn(signal.size(), device=signal.device) * torch.sqrt(noise_power)
    noisy_signal = signal + noise

    return noisy_signal

def add_rayleigh_noise(signal, snr):
    # Compute the signal power
     # Compute the signal power
    signal_power = torch.mean(torch.square(torch.abs(signal)))
    
    # Compute the noise power from the SNR
    noise_power = signal_power / snr
    std = torch.sqrt(noise_power / 2)

    # Create a tensor of same shape as signal with all elements equal to std
    std_tensor = torch.full(signal.shape, std.item(), device=signal.device)

    # Generate real and imaginary noise
    noise_real = torch.normal(mean=0.0, std=std_tensor)
    noise_imag = torch.normal(mean=0.0, std=std_tensor)

    # Generate Rayleigh fading variable
    h = torch.sqrt(torch.normal(mean=0.0, std=1, size=signal.shape) ** 2
                   + torch.normal(mean=0.0, std=1, size=signal.shape) ** 2) / np.sqrt(2)
    
    noise = noise_real + 1j * noise_imag

    noise = noise.to(signal.device)
    h = h.to(signal.device)
    
    return signal * h + torch.abs(noise)

