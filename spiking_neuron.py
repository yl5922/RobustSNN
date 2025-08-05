# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 21:23:43 2025

@author: yl5922
"""

import torch
import torch.nn as nn

class ArctanSpike(torch.autograd.Function):
    """
    Surrogate spike function with an arctan-based gradient.
    Forward  : Hard threshold (Heaviside step).
    Backward : Approximate derivative using an arctan curve around the threshold.
    """
    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        """
        Forward pass: output = 1.0 if V >= threshold, else 0.0
        """
        # Save tensors for backward
        ctx.save_for_backward(membrane_potential, threshold)
        # Hard spike function
        out = (membrane_potential >= threshold).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use the derivative of arctan (centered at threshold) as a surrogate.
        d/dV [H(V - Vth)] ~ α * 1 / (pi * (1 + (α*(V - Vth))^2))
        """
        membrane_potential, threshold = ctx.saved_tensors
        
        # Surrogate scale factor α to control the "sharpness"
        alpha = 1.0  
        
        # Shift the membrane potential around threshold and scale
        x = alpha * (membrane_potential - threshold)
        
        # Derivative of arctan(x) = 1 / (1 + x^2)
        # We include 1/pi so that the integral is ~1 across the main lobe
        grad_spike = 1.0 / (1.0 + x.pow(2)*torch.pi**2)
        
        # Return chain rule: grad_output * dSpike/dV
        # - We pass `None` for threshold because we don't need gradient wrt threshold
        return grad_output * grad_spike, None


arctan_spike_function = ArctanSpike.apply

class IFNeuron(nn.Module):
    def __init__(self, noise_range = 0.0, threshold=1.0, v_reset=0.0):
        """
        threshold: membrane potential threshold for spiking
        v_reset  : membrane potential after a spike
        """
        super(IFNeuron, self).__init__()
        self.base_threshold = threshold
        self.noise_range = noise_range
        self.v_reset = v_reset
        self.v = 0.0
        
    def forward(self, input_current):
        """
        input_current: tensor of shape [batch, ...] with the net current for this timestep.
        returns      : spikes (0 or 1) with the same shape
        """
        # If the stored membrane_potential has a different shape than input_current, re-init
        if isinstance(self.v ,float):
            self.v = torch.zeros_like(input_current)
        
        threshold = self.base_threshold*torch.ones_like(input_current) + (torch.rand_like(input_current)*2-1)*self.noise_range
        threshold = threshold.detach()
        # Integrate
        self.v = self.v + input_current
        
        # Generate spike using the hard threshold + arctan surrogate in backprop
        spikes = arctan_spike_function(self.v, threshold)
        
        # Reset the membrane potential where a spike was emitted
        self.v = torch.where(spikes > 0, torch.tensor(self.v_reset, device=input_current.device),self.v)
        return spikes
    
    def reset(self):
        """
        Resets the internal state (membrane potential and spike counter).
        Should be called before processing a new batch.
        """
        self.v = 0.0

class LIFNeuron(nn.Module):
    def __init__(self, tau = 2, noise_range = 0.0, threshold=1.0, v_reset=0.0):
        """
        threshold: membrane potential threshold for spiking
        v_reset  : membrane potential after a spike
        """
        super(LIFNeuron, self).__init__()
        self.base_threshold = threshold
        self.noise_range = noise_range
        self.v_reset = v_reset
        self.v = 0.0
        self.tau = tau
        
    def forward(self, input_current):
        """
        input_current: tensor of shape [batch, ...] with the net current for this timestep.
        returns      : spikes (0 or 1) with the same shape
        """
        # If the stored membrane_potential has a different shape than input_current, re-init
        if isinstance(self.v ,float):
            self.v = torch.zeros_like(input_current)
        
        threshold = self.base_threshold*torch.ones_like(input_current) + (torch.rand_like(input_current)*2-1)*self.noise_range
        threshold = threshold.detach()
        # Integrate
        #self.v = self.tau*self.v + input_current
        self.v = self.v + (input_current - self.v) / self.tau

        # Generate spike using the hard threshold + arctan surrogate in backprop
        spikes = arctan_spike_function(self.v, threshold)
        
        # Reset the membrane potential where a spike was emitted
        self.v = torch.where(spikes > 0, torch.tensor(self.v_reset, device=input_current.device),self.v)
        return spikes
    
    def get_v(self, input_current):
        return self.v + (input_current - self.v) / self.tau

    def reset(self):
        """
        Resets the internal state (membrane potential and spike counter).
        Should be called before processing a new batch.
        """
        self.v = 0.0