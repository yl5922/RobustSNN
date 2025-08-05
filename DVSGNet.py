# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:51:56 2025

@author: yl5922
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from channels import *
from utils import real_to_complex, reset_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import surrogate, neuron
import math
from spiking_neuron import LIFNeuron


def normalize_to_avg_power(x: torch.Tensor, target_power=0.5):
    """
    Normalize each sample in the batch (N, L) so that average power = target_power.
    Used for ANN to avoid explosion
    
    Args:
        x: Tensor of shape (N, L)
        target_power: Desired average power per sample (default = 0.5)

    Returns:
        Normalized tensor with same shape
    """
    # Compute average power per sample (along dimension L)
    avg_power = x.pow(2).mean(dim=1, keepdim=True) + 1e-10 # Shape: (N, 1)

    # Scale each row to have the desired average power
    scaling_factor = torch.sqrt(target_power / avg_power)  # Shape: (N, 1)
    
    scaling_factor = torch.where(avg_power > target_power, scaling_factor, torch.ones_like(scaling_factor))

    
    return x * scaling_factor
    
class ANNGesture(nn.Module):
    def __init__(self, T: int = 4, channel_snr=None, channel_type='gaussian',  rician_factor = 1.0, use_complex = False):
        super(ANNGesture, self).__init__()
        conv_channels = 64
        block_length = 256
        self.T = T
        self.channel_snr = channel_snr
        self.channel_type = channel_type.lower()
        self.rician_factor = rician_factor
        self.use_complex = use_complex
        
        self.conv1 = nn.Conv2d(2, conv_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.lif1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(conv_channels)
        self.lif2 =  nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(conv_channels)
        self.lif3 = nn.ReLU()      
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(conv_channels)
        self.lif4 = nn.ReLU()   
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(conv_channels)
        self.lif5 = nn.ReLU()   
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        W = int(128/2/2/2/2/2)
        self.fc1 = nn.Linear(conv_channels*W*W, block_length*2, bias = True)
        self.lif6 = nn.ReLU() 
        
        # decoder
        self.fc2 = nn.Linear(block_length*2, 512, bias = True)
        self.lif7 = nn.ReLU()
        
        self.fc3 = nn.Linear(512, 11, bias = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(
        self,
        x: torch.Tensor,
        noisy_thres: float = 0.0
    ) -> (torch.Tensor, torch.Tensor):
        output = 0.0
        penalty_acc = 0.0
        
        l1_rate = torch.tensor(0.0, device=x.device)
        l2_rate = torch.tensor(0.0, device=x.device)
        l3_rate = torch.tensor(0.0, device=x.device)
        l4_rate = torch.tensor(0.0, device=x.device)
        l5_rate = torch.tensor(0.0, device=x.device)
        l6_rate = torch.tensor(0.0, device=x.device)
        l7_rate = torch.tensor(0.0, device=x.device)
        
        n = None
        h = None
        for t in range(self.T):
            # conv1 + lif1 + pool1            
            conv_out = self.conv1(x[:,t])
            conv_out = self.lif1(conv_out)
            conv_out = self.pool1(conv_out)

            # conv2 + lif2 + pool2
            conv_out = self.conv2(conv_out)
            conv_out = self.lif2(conv_out)
            conv_out = self.pool2(conv_out)
            
            # conv3 + lif3 + pool3
            conv_out = self.conv3(conv_out)
            conv_out = self.bn3(conv_out)
            conv_out = self.lif3(conv_out)
            conv_out = self.pool3(conv_out)

            # conv4 + lif4 + pool4
            conv_out = self.conv4(conv_out)
            conv_out = self.bn4(conv_out)
            conv_out = self.lif4(conv_out)
            conv_out = self.pool4(conv_out)
            
            # conv5 + lif5 + pool5
            conv_out = self.conv5(conv_out)
            conv_out = self.bn5(conv_out)
            conv_out = self.lif5(conv_out)
            conv_out = self.pool5(conv_out)
            
            # flatten
            conv_out = torch.flatten(conv_out, 1)

            # fc1 + lif3
            conv_out = self.fc1(conv_out)
            conv_out = self.lif6(conv_out)
            
            conv_out = normalize_to_avg_power(conv_out)
            
            # 4) pass the channel
            conv_out = self.apply_channel(conv_out)
            
            conv_out = self.fc2(conv_out)
            
            conv_out = self.lif7(conv_out)
            conv_out = self.fc3(conv_out)
            output += conv_out

        output /= float(self.T)
        penalty_acc /= float(self.T)
        
        l1_rate /= self.T
        l2_rate /= self.T
        l3_rate /= self.T
        l4_rate /= self.T
        l5_rate /= self.T
        l6_rate /= self.T
        l7_rate /= self.T
        
        # Return the dictionary of spiking rates
        spiking_rates = {
            "layer1": l1_rate,
            "layer2": l2_rate,
            "layer3": l3_rate,
            "layer4": l4_rate,
            "layer5": l5_rate,
            "layer6": l6_rate,
            "layer7": l7_rate
            }

        return output, penalty_acc, spiking_rates  
    
    def apply_channel(self, x):
        """
        Apply wireless channel (Gaussian or Rician) to real or complex input x.
    
        Returns:
            x after passing through the channel (real-valued tensor).
        """
        if self.use_complex:
            x = real_to_complex(x)
    
            if self.channel_snr is not None:
                if self.channel_type == 'gaussian':
                    n = generate_gaussian_noise(x)
                    x = apply_gaussian_channel(x, self.channel_snr, n)
                elif self.channel_type == 'rician':
                    h = generate_rician_fading(x, K=self.rician_K)
                    n = generate_gaussian_noise(x)
                    y = apply_rician_channel(x, h, self.channel_snr, n)
                    x = rician_equalize(y, h)
    
            # back to real (concat real and imag parts)
            x = torch.cat([x.real, x.imag], dim=1)
    
        else:
            if self.channel_snr is not None:
                if self.channel_type == 'gaussian':
                    n = generate_gaussian_noise(x)
                    x = apply_gaussian_channel(x, self.channel_snr, n)
                elif self.channel_type == 'rician':
                    h = generate_rician_fading(x, K=self.rician_K)
                    n = generate_gaussian_noise(x)
                    y = apply_rician_channel(x, h, self.channel_snr, n)
                    x = rician_equalize(y, h)
    
        return x
    
class SNNGesture(nn.Module):
    def __init__(self, T: int = 4, activation_buffer_limit: int = 10, channel_snr=None, channel_type='gaussian', rician_factor = 1.0, use_complex = False):
        super(SNNGesture, self).__init__()
        conv_channels = 64
        block_length = 256
        self.T = T
        self.activation_buffer_limit = activation_buffer_limit
        self.channel_snr = channel_snr
        self.channel_type = channel_type.lower()
        self.rician_factor = rician_factor
        self.use_complex = use_complex
        
        self.conv1 = nn.Conv2d(2, conv_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.lif1 = LIFNeuron()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(conv_channels)
        self.lif2 = LIFNeuron()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(conv_channels)
        self.lif3 = LIFNeuron()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(conv_channels)
        self.lif4 = LIFNeuron()   
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(conv_channels, conv_channels, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(conv_channels)
        self.lif5 = LIFNeuron()  
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        W = int(128/2/2/2/2/2)
        self.fc1 = nn.Linear(conv_channels*W*W, block_length*2, bias = False)
        self.lif6 = LIFNeuron() 
        
        # decoder
        self.fc2 = nn.Linear(block_length*2, 512, bias = False)
        self.lif7 = LIFNeuron() 
        
        self.fc3 = nn.Linear(512, 11, bias = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(
        self,
        x: torch.Tensor,
        noisy_thres: float = 0.0
    ) -> (torch.Tensor, torch.Tensor):
        output = 0.0
        
        l1_rate = torch.tensor(0.0, device=x.device)
        l2_rate = torch.tensor(0.0, device=x.device)
        l3_rate = torch.tensor(0.0, device=x.device)
        l4_rate = torch.tensor(0.0, device=x.device)
        l5_rate = torch.tensor(0.0, device=x.device)
        l6_rate = torch.tensor(0.0, device=x.device)
        l7_rate = torch.tensor(0.0, device=x.device)
        
        penalty_weights = {
            "penalty1": 1e0,
            "penalty2": 1e0,
            "penalty3": 0e-2,
            "penalty4": 0e-2,
            "penalty5": 0e-2,
            "penalty6": 0e-1,
            "penalty7": 1e-1,
        }
        
        penalty1 = torch.tensor(0.0, device=x.device)
        penalty2 = torch.tensor(0.0, device=x.device)
        penalty3 = torch.tensor(0.0, device=x.device)
        penalty4 = torch.tensor(0.0, device=x.device)
        penalty5 = torch.tensor(0.0, device=x.device)
        penalty6 = torch.tensor(0.0, device=x.device)
        penalty7 = torch.tensor(0.0, device=x.device)
        
        n = None
        h = None
        for t in range(self.T):

            for lif in [self.lif1, self.lif2, self.lif3, self.lif4, self.lif5, self.lif6]:
                lif.noise_range = noisy_thres
                
            # conv1 + lif1 + pool1            
            conv_out = self.conv1(x[:,t])
            penalty1 += self.exp_penalty(self.lif1.get_v(conv_out)) 
            conv_out = self.lif1(conv_out)
            l1_rate += conv_out.mean()
            conv_out = self.pool1(conv_out)

            # conv2 + lif2 + pool2
            conv_out = self.conv2(conv_out)
            penalty2 += self.exp_penalty(self.lif2.get_v(conv_out))
            conv_out = self.lif2(conv_out)
            l2_rate += conv_out.mean()
            conv_out = self.pool2(conv_out)
            
            # conv3 + lif3 + pool3
            conv_out = self.conv3(conv_out)
            conv_out = self.bn3(conv_out)
            penalty3 += self.exp_penalty(self.lif3.get_v(conv_out))
            conv_out = self.lif3(conv_out)
            l3_rate += conv_out.mean()
            conv_out = self.pool3(conv_out)

            # conv4 + lif4 + pool4
            conv_out = self.conv4(conv_out)
            conv_out = self.bn4(conv_out)
            penalty4 += self.exp_penalty(self.lif4.get_v(conv_out))
            conv_out = self.lif4(conv_out)
            l4_rate += conv_out.mean()
            conv_out = self.pool4(conv_out)
            
            # conv5 + lif5 + pool5
            conv_out = self.conv5(conv_out)
            conv_out = self.bn5(conv_out)
            penalty5 += self.exp_penalty(self.lif5.get_v(conv_out))
            conv_out = self.lif5(conv_out)
            l5_rate += conv_out.mean()
            conv_out = self.pool5(conv_out)
            
            # flatten
            conv_out = torch.flatten(conv_out, 1)

            # fc1 + lif3
            conv_out = self.fc1(conv_out)
            penalty6 += self.exp_penalty(self.lif6.get_v(conv_out))
            conv_out = self.lif6(conv_out)
            l6_rate += conv_out.mean()

            # 4) pass the channel
            conv_out = self.apply_channel(conv_out)
            
            conv_out = self.fc2(conv_out)
            
            penalty7 += self.exp_penalty(self.lif7.get_v(conv_out))

            conv_out = self.lif7(conv_out)
            l7_rate += conv_out.mean()
            conv_out = self.fc3(conv_out)
            output += conv_out

        output /= float(self.T)
        
        l1_rate /= self.T
        l2_rate /= self.T
        l3_rate /= self.T
        l4_rate /= self.T
        l5_rate /= self.T
        l6_rate /= self.T
        l7_rate /= self.T
        
        # Return the dictionary of spiking rates
        spiking_rates = {
            "layer1": l1_rate,
            "layer2": l2_rate,
            "layer3": l3_rate,
            "layer4": l4_rate,
            "layer5": l5_rate,
            "layer6": l6_rate,
            "layer7": l7_rate
            }
        
        # Average across time
        penalty1 /= self.T
        penalty2 /= self.T
        penalty3 /= self.T
        penalty4 /= self.T
        penalty5 /= self.T
        penalty6 /= self.T
        penalty7 /= self.T
        
        # Weighted sum of penalties
        penalty_acc = (
            penalty_weights["penalty1"] * penalty1 +
            penalty_weights["penalty2"] * penalty2 +
            penalty_weights["penalty3"] * penalty3 +
            penalty_weights["penalty4"] * penalty4 +
            penalty_weights["penalty5"] * penalty5 +
            penalty_weights["penalty6"] * penalty6 +
            penalty_weights["penalty7"] * penalty7
        )

        return output, penalty_acc, spiking_rates
                
    def exp_penalty(self, activations: torch.Tensor) -> torch.Tensor:
        """
        For each activation x, penalty = exp(-(x - 1)^2).
        We take the mean across all elements to get a single scalar.
        """
        flat = activations.view(-1)
        penalty_vals = torch.exp(-((flat - 1.0) ** 2)/0.1)
        return torch.mean(penalty_vals)
    
    def apply_channel(self, x):
        """
        Apply wireless channel (Gaussian or Rician) to real or complex input x.
    
        Returns:
            x after passing through the channel (real-valued tensor).
        """
        if self.use_complex:
            x = real_to_complex(x)
    
            if self.channel_snr is not None:
                if self.channel_type == 'gaussian':
                    n = generate_gaussian_noise(x)
                    x = apply_gaussian_channel(x, self.channel_snr, n)
                elif self.channel_type == 'rician':
                    h = generate_rician_fading(x, K=self.rician_K)
                    n = generate_gaussian_noise(x)
                    y = apply_rician_channel(x, h, self.channel_snr, n)
                    x = rician_equalize(y, h)
    
            # back to real (concat real and imag parts)
            x = torch.cat([x.real, x.imag], dim=1)
    
        else:
            if self.channel_snr is not None:
                if self.channel_type == 'gaussian':
                    n = generate_gaussian_noise(x)
                    x = apply_gaussian_channel(x, self.channel_snr, n)
                elif self.channel_type == 'rician':
                    h = generate_rician_fading(x, K=self.rician_K)
                    n = generate_gaussian_noise(x)
                    y = apply_rician_channel(x, h, self.channel_snr, n)
                    x = rician_equalize(y, h)
    
        return x