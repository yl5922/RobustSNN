# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:40:45 2025

@author: yl5922
"""
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import torchvision

def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()
            
def set_seed(seed_value: int):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    # Ensure deterministic behavior in cuDNN (potential performance trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def real_to_complex(x):
    """
    Convert real-valued input into a complex modulated signal.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features).

    Returns:
        torch.Tensor: Complex-valued tensor of shape (batch_size, features/2).
    """
    batch_size, features = x.shape
    if features % 2 != 0:
        raise ValueError("Feature size must be even for conversion")

    # Split input into real (I) and imaginary (Q) components
    I, Q = x[:, :features//2], x[:, features//2:]

    # Combine as complex numbers
    complex_output = torch.complex(I, Q)  # Shape: (batch_size, features/2)
    
    return complex_output

def l1_proj(delta, epsilon):
    """
    Projects delta onto the L1 ball of radius epsilon.
    Based on Duchi et al. (2008).
    """
    flat = delta.view(delta.size(0), -1)
    abs_flat = flat.abs()

    # Sort descending
    sorted_abs, _ = torch.sort(abs_flat, dim=1, descending=True)
    cssv = torch.cumsum(sorted_abs, dim=1) - epsilon

    rho = torch.arange(1, flat.size(1)+1, device=delta.device).float().view(1, -1)
    condition = sorted_abs > cssv / rho
    k = condition.sum(dim=1) - 1

    # Gather thresholds
    threshold = (cssv.gather(1, k.view(-1, 1)) / (k+1).view(-1, 1)).squeeze(1)
    threshold = threshold.view(-1, 1)

    # Shrink
    proj_flat = torch.sign(flat) * torch.clamp(abs_flat - threshold, min=0.0)
    return proj_flat.view_as(delta)

def spike_attack(model, inputs, labels, epsilon=3, max_step_per_pixel = 50, num_iter=10, targeted=False):
    """
    Perform L1-norm bounded PGD attack on a batch of inputs.
    """
    batch_size = inputs.size(0)
    adv_inputs = inputs.clone().detach()
    adv_inputs.requires_grad = True

    label_onehot = F.one_hot(labels, 11).float()
    loss_fn = F.mse_loss
    alpha = 0.2*epsilon

    for _ in range(num_iter):
        reset_net(model)
        outputs, _, _ = model(adv_inputs)
        loss = loss_fn(outputs, label_onehot)

        if targeted:
            loss = -loss

        loss.backward()

        with torch.no_grad():
            grad = adv_inputs.grad
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=1, dim=1).view(-1, 1, 1, 1, 1)
            normalized_grad = grad / (grad_norm + 1e-16)

            adv_inputs = adv_inputs + alpha * normalized_grad

            # Project delta to L1 ball
            delta = adv_inputs - inputs
            delta = l1_proj(delta, epsilon)
            adv_inputs = inputs + delta
            # Optional clamp: adv_inputs = torch.clamp(inputs + delta, -1, 2)

        adv_inputs.requires_grad = True
        
    with torch.no_grad():
        direction = adv_inputs - inputs  # use true accumulated direction
    
        for i in range(batch_size):
            d = direction[i].flatten()
            signs = d.sign()
            scores = d.abs()
            total_score = scores.sum()
    
            if total_score == 0:
                continue
    
            # Step 1: Proportional float allocation
            float_alloc = scores / total_score * epsilon
            base_alloc = float_alloc.floor()
            remainder = float_alloc - base_alloc
    
            # Step 2: Convert to signed integer delta
            delta = base_alloc * signs
            used = base_alloc.sum()
    
            # Step 3: Distribute leftover steps
            leftover = int(epsilon - used.item())
            if leftover > 0:
                _, extra_idx = torch.topk(remainder, leftover)
                delta[extra_idx] += signs[extra_idx]
    
            delta = delta.view_as(inputs[i])
    
            # Step 4: Apply constraints
            updated = inputs[i].float() + delta
            updated = torch.clamp(updated, min=0)
            delta = updated - inputs[i].float()
            delta = delta.clamp(min=-max_step_per_pixel, max=max_step_per_pixel)
    
            adv_inputs[i] = torch.round(inputs[i].float() + delta)
    
    return adv_inputs

def evaluate_under_spike_attack(model, dataloader, device, epsilon=3.0, max_step_per_pixel = 50, targeted=False):
    """
    Evaluate model accuracy under L2-norm PGD attack over the entire dataloader.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        adv_inputs = spike_attack(model, inputs, labels, epsilon=epsilon, max_step_per_pixel = max_step_per_pixel, targeted=targeted)

        with torch.no_grad():
            reset_net(model)
            outputs,_,_ = model(adv_inputs)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        # print(f"Spikes under epsilon = {epsilon} attack is {(inputs-adv_inputs).abs().view(inputs.size(0),-1).sum(-1)}")

    accuracy = total_correct / total_samples
    print(f"Accuracy under L2 PGD attack: {accuracy * 100:.2f}%")

    return accuracy

def evaluate_spike_attack_multiple_epsilons(model, dataloader, device, epsilons, max_step_per_pixel = 50, targeted=False):
    """
    Evaluate model accuracy under different L2 PGD attack strengths (different epsilon values),
    using the existing evaluate_under_l2_attack() function.
    
    Args:
        model: the neural network.
        dataloader: evaluation dataloader.
        device: 'cuda' or 'cpu'.
        epsilons: list of epsilon values.
        alpha_ratio: step size relative to epsilon (e.g., 0.1 * epsilon).
        num_iter: number of PGD steps.
        targeted: perform targeted attack if True.
    
    Returns:
        A list of attack accuracies for each epsilon.
    """
    results = []

    for eps in epsilons:
        print(f"Evaluating with epsilon = {eps:.4f}")

        acc = evaluate_under_spike_attack(
            model=model,
            dataloader=dataloader,
            device=device,
            epsilon=eps,
            max_step_per_pixel=max_step_per_pixel,
            targeted=targeted
        )

        results.append(acc)

    return results

def parseval_projection(module, beta=0.0001):
    """Perform a single Parseval projection on the given layer's weights to enforce orthogonality."""
    with torch.no_grad():  
        # For fully connected (linear) layers
        if isinstance(module, nn.Linear):
            W = module.weight            # W shape: (out_features, in_features)
            WWT = W @ W.T               # Compute W * W^T (shape: out_features x out_features)
            # Apply Parseval update: W <- (1 + beta) * W - beta * (W W^T) * W
            module.weight.copy_((1 + beta) * W - beta * WWT @ W)
        
        # For convolutional layers
        elif isinstance(module, nn.Conv2d):
            W = module.weight           # W shape: (out_channels, in_channels, kH, kW)
            out_channels = W.shape[0]
            W_mat = W.view(out_channels, -1)    # Flatten to 2D: (out_channels, in_channels * kH * kW)
            WWT = W_mat @ W_mat.T              # Compute W_mat * W_mat^T (shape: out_channels x out_channels)
            # Apply Parseval update and reshape back to original convolutional weight shape
            W_mat_new = (1 + beta) * W_mat - beta * WWT @ W_mat
            module.weight.copy_(W_mat_new.view_as(W))