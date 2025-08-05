# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:25:09 2025

@author: yl5922
"""
import torch
import torch.nn as nn

def generate_rician_fading(x, K=10.0):
    """
    Generate Rician fading channel coefficient h for real or complex input x.

    Args:
        x (torch.Tensor): Input tensor (real or complex), shape (batch_size, features, ...).
        K (float): Rician K-factor in dB.

    Returns:
        torch.Tensor: Fading coefficient h with same dtype and shape as x.
    """
    K_linear = 10 ** (K / 10.0)
    dtype = x.real.dtype if torch.is_complex(x) else x.dtype

    s = torch.sqrt(torch.tensor(K_linear / (K_linear + 1), device=x.device, dtype=dtype))

    if torch.is_complex(x):
        sigma = torch.sqrt(torch.tensor(1 / (2 * (K_linear + 1)), device=x.device, dtype=dtype))
        s = s + 0j  # Ensure s is complex
        noise = torch.randn_like(x) + 1j * torch.randn_like(x)
    else:
        sigma = torch.sqrt(torch.tensor(1 / (K_linear + 1), device=x.device, dtype=dtype))
        noise = torch.randn_like(x)

    h = s + sigma * noise
    return h[:,0].unsqueeze(-1)


def generate_gaussian_noise(x):
    """
    Generate unscaled Gaussian noise with same shape and dtype as x.

    Args:
        x (torch.Tensor): Input signal (real or complex).

    Returns:
        torch.Tensor: Gaussian noise tensor with same shape and dtype.
    """
    if torch.is_complex(x):
        return torch.randn_like(x) + 1j * torch.randn_like(x)
    else:
        return torch.randn_like(x)

def apply_rician_channel(x, h, snr_db, noise):
    """
    Apply Rician channel with known h and AWGN.

    Args:
        x (torch.Tensor): Input signal (real or complex).
        h (torch.Tensor): Channel coefficient (same shape as x).
        snr_db (float): SNR in dB.
        noise (torch.Tensor): Pre-generated noise (same shape as x).

    Returns:
        torch.Tensor: Output y = hx + n
    """
    if snr_db is None:
        return h * x

    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = (x.abs()**2 if torch.is_complex(x) else x**2).mean().detach()
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    if torch.is_complex(x):
        noise_std = noise_std / torch.sqrt(torch.tensor(2.0, device=x.device))
        return h * x + noise * noise_std
    else:
        return h * x + noise * noise_std

def rician_equalize(y, h):
    """
    Equalize received signal y by dividing with h.

    Args:
        y (torch.Tensor): Received signal (real or complex).
        h (torch.Tensor): Channel coefficient.

    Returns:
        torch.Tensor: Equalized signal.
    """
    return y / h

def apply_gaussian_channel(x, snr_db, noise):
    """
    Applies AWGN channel to real or complex input x using pre-generated noise.

    Args:
        x (torch.Tensor): Input signal, real or complex.
        snr_db (float): SNR in dB.
        noise (torch.Tensor): Unscaled Gaussian noise.

    Returns:
        torch.Tensor: Noisy signal (same dtype and shape as x).
    """
    if snr_db is None:
        return x

    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = (x.real.pow(2) + x.imag.pow(2)).mean().detach() if torch.is_complex(x) else x.pow(2).mean().detach()
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    if torch.is_complex(x):
        noise_std = noise_std / torch.sqrt(torch.tensor(2.0, device=x.device))
        return x + noise * noise_std
    else:
        return x + noise * noise_std




def add_source_noise(images: torch.Tensor, snr_db: float, dist: str = 'gaussian', all_range = False) -> torch.Tensor:
    """
    Add Gaussian noise to achieve a target SNR (in dB).
    """
    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10.0)
        
        if all_range:
            snr_range = torch.rand([images.size(0),1,1,1], device = images.device)*20-10
            snr_linear = 10 ** (snr_range / 10.0)
            
        signal_power = images.pow(2).mean()
        noise_power = signal_power / snr_linear
    
        if dist.lower() == "gaussian":
            # variance = noise_power -> std = sqrt(noise_power)
            std = torch.sqrt(noise_power)
            noise = std * torch.randn_like(images)
    
        elif dist.lower() == "laplacian":
            # Laplacian(0, b), var = 2*b^2 => b^2 = noise_power/2 => b = sqrt(noise_power/2)
            b = torch.sqrt(noise_power / 2)
            dist_lap = torch.distributions.Laplace(loc=0.0, scale=b)
            noise = dist_lap.sample(images.shape).to(images.device)
    
        elif dist.lower() == "uniform":
            # Uniform(-r, +r), var = r^2 / 3 => r^2/3 = noise_power => r = sqrt(3*noise_power)
            r = torch.sqrt(3.0 * noise_power)
            # Sample in [-r, +r]
            noise = 2.0 * r * torch.rand_like(images) - r
    
        else:
            raise ValueError(f"Unsupported distribution for SNR noise: {dist}")
    
        noisy_images = images + noise    
    else:
        noisy_images = images
        
    return torch.clamp(noisy_images, 0.0, 1.0)

def add_source_noise_DVSG(images: torch.Tensor, snr_db: float, dist: str = 'gaussian', all_range = False) -> torch.Tensor:
    """
    Add Gaussian noise to achieve a target SNR (in dB).
    For DVSG, since it is a spike dataset, its avergae power is the average firing probability of input neurons.
    """
    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10.0)
        
        if all_range:
            snr_range = torch.rand([images.size(0),1,1,1], device = images.device)*20-10
            snr_linear = 10 ** (snr_range / 10.0)
            
        signal_power = images.mean()
        noise_power = signal_power / snr_linear
    
        if dist.lower() == "gaussian":
            # variance = noise_power -> std = sqrt(noise_power)
            std = torch.sqrt(noise_power)
            noise = std * torch.randn_like(images)
    
        elif dist.lower() == "laplacian":
            # Laplacian(0, b), var = 2*b^2 => b^2 = noise_power/2 => b = sqrt(noise_power/2)
            b = torch.sqrt(noise_power / 2)
            dist_lap = torch.distributions.Laplace(loc=0.0, scale=b)
            noise = dist_lap.sample(images.shape).to(images.device)
    
        elif dist.lower() == "uniform":
            # Uniform(-r, +r), var = r^2 / 3 => r^2/3 = noise_power => r = sqrt(3*noise_power)
            r = torch.sqrt(3.0 * noise_power)
            # Sample in [-r, +r]
            noise = 2.0 * r * torch.rand_like(images) - r
    
        else:
            raise ValueError(f"Unsupported distribution for SNR noise: {dist}")
    
        noisy_images = images + noise    
    else:
        noisy_images = images
        
    return noisy_images
