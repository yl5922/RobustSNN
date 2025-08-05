# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:56:43 2025

@author: yl5922
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from DVSGNet import SNNGesture, ANNGesture
from utils import *
from channels import add_source_noise, add_source_noise_DVSG
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import random

def layerwise_param_groups(model, decay_dict):
    """
    Generates parameter groups for an entire model, 
    assigning weight_decay based on partial matches in param names.

    Args:
        model (nn.Module): PyTorch model.
        decay_dict (dict): mapping from name-substring -> weight_decay value.
                           e.g. {"layer1": 1e-4, "layer2": 1e-5, "bias": 0.0}

    Returns:
        list of dicts: parameter groups for optimizer
    """
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen params

        # default decay
        chosen_decay = 0.0
        for key_substring, decay_val in decay_dict.items():
            if key_substring in name:
                chosen_decay = decay_val
                break

        param_groups.append({'params': param, 'weight_decay': chosen_decay})

    return param_groups

def compute_average_spikes_from_loader(dataloader):
    """
    Computes the average number of spikes per sample from a DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding spike frames (as tensors).
                                 Expected shape per sample: [C, T, H, W] or [C, H, W]

    Returns:
        float: Average number of spikes per sample
    """


    total_spikes = 0.0
    total_samples = 0

    for batch in dataloader:
        # If dataset returns (data, label), extract data
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        data = data.float()  # convert to float for summation

        # Sum all spikes per sample and add to total
        batch_spike_counts = data.view(data.size(0), -1).sum(dim=1)  # shape: [B]
        total_spikes += batch_spike_counts.sum().item()
        total_samples += data.size(0)

    return total_spikes / total_samples if total_samples > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=8, help="Timesteps for SNN")
    parser.add_argument("--penalty_separation_weight", type=float, default=0.1)#1
    parser.add_argument("--l2_reg", type=float, default=5e-2, 
                        help="Regularization on the weights. Although the paper uses the $\ell_q$ norm, we found that the $\ell_2$ norm achieves good performance. For consistency, we adopt the $\ell_2$ norm throughout.")

    parser.add_argument("--noisy_thres", default=0.00, 
                        help="add noise to firing threshold")
    parser.add_argument("--adversarial_training", action = 'store_true', default=False)
    parser.add_argument("--parseval_normalization", action = 'store_true', default=False)
    parser.add_argument("--evaluate_interval", type = int, default = 10, help = 'evaluate the performance every X epochs')
    parser.add_argument("--epsilon",  type = int, default=5)
    parser.add_argument("--attack_step",  type = int, default=10)
    
    parser.add_argument("--model_type", type=str, choices=["ANN", "SNN"], default="SNN")
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--epochs_before_penalty", type=int, default=1,
                        help="Start applying penalty after these many epochs for SNN")

    parser.add_argument("--train_under_channel_noise", type=float, default=None, 
                        help="If not None, train images with channel SNR in dB. Otherwise, train under lossless channel.")
    parser.add_argument("--train_channel_dist", type=str, default="gaussian", choices=['gaussian', 'rayleigh'], 
                        help="Noise distribution type for traing channel noise")
    parser.add_argument("--test_channel_dist", type=str, default="gaussian", choices=['gaussian', 'rayleigh'], 
                        help="Noise distribution type for testing channel noise")
    parser.add_argument("--rician factor", type=float, default=1, 
                        help="rayleigh scale of the rayleigh channel")


    set_seed(2025)
    args = parser.parse_args()
    print(args)
    
    train_augment = transforms.Compose([
    transforms.RandomAffine(
        degrees=0,               # rotate ±10 degrees
        translate=(0.0, 0.0),     # translate up to 10% of image size
        scale=(1.0, 1.0),         # scale up/down 10%
        shear=0                  # slight shearing
    ),
    transforms.ToTensor()])

    basic_transform = transforms.Compose([transforms.ToTensor()])
    
    class RandomTranslate:
        def __init__(self, max_offset=25):
            self.max_offset = max_offset
    
        def __call__(self, img):
            off1 = random.randint(-self.max_offset, self.max_offset)
            off2 = random.randint(-self.max_offset, self.max_offset)
            return transforms.functional.affine(img, angle=0.0, scale=1.0, shear=0.0, translate=(off1, off2))
    
    class toTensor:
        def __init__(self):
            pass
    
        def __call__(self, img):
            return torch.from_numpy(img) 
        
    # Example usage in a transformation pipeline
    dvsg_transform = transforms.Compose([
        toTensor(),
        RandomTranslate(max_offset=25)
    ])
    

    train_set = DVS128Gesture(args.datadir, train=True, data_type='frame', split_by='number', frames_number=args.T, transform=dvsg_transform)
    test_set = DVS128Gesture(args.datadir, train=False, data_type='frame', split_by='number', frames_number=args.T)

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model selection
    if args.model_type == "SNN":
        model = SNNGesture(T=args.T, channel_snr=args.train_under_channel_noise, channel_type=args.train_channel_dist).to(device)
    else:
        model = ANNGesture(T = args.T, channel_snr=args.train_under_channel_noise, channel_type=args.train_channel_dist).to(device)

    count_set = DVS128Gesture(args.datadir, train=True, data_type='frame', split_by='number', frames_number=args.T, transform=None)
    count_loader = torch.utils.data.DataLoader(count_set, num_workers=args.num_worker, batch_size=args.batch_size)
    total_spikes = compute_average_spikes_from_loader(count_loader) 

    print(f"The average total number of spikes of an DVSGesture sample is {total_spikes}")      
    criterion = F.mse_loss
    scaler = torch.cuda.amp.GradScaler()
    
    decay_dict = {
    "conv1.weight": 1e-4, # 1e-6, conv layer penalty
    "conv2.weight": 1e-4, # 1e-6   
    "conv3.weight": 1e-6, # 1e-6, conv layer penalty
    "conv4.weight": 1e-6, # 1e-6   
    "conv5.weight": 1e-6, # 1e-6  
    "fc1.weight": 1e-6,   # 1e-6 fc layer penalty
    "fc2.weight": 1e-4, # 1e-3
    "fc3.weight": 1e-6 # 2e-4
    }
    l2_decay_dict = {k: v * args.l2_reg for k, v in decay_dict.items()}
    print('l2_decay_dict weight is')
    print(l2_decay_dict)

    param_groups = layerwise_param_groups(model, l2_decay_dict)
    optimizer = optim.Adam(param_groups, lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.T_max)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            label_onehot = F.one_hot(labels, 11).float()
            reset_net(model)
                        
            if args.adversarial_training == True:
                permuted_images = spike_attack(
                model, images, labels,
                epsilon=args.epsilon,
                max_step_per_pixel= 50,
                targeted=False
            ).detach()
            reset_net(model)
                    
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if args.model_type == "SNN":
                    outputs, penalty, spiking_rates = model(images, args.noisy_thres)
                    loss = criterion(outputs, label_onehot)
                    
                    if args.adversarial_training:
                        reset_net(model)
                        permuted_outputs, _,  _ = model(permuted_images, args.noisy_thres)
                        loss += criterion(permuted_outputs, label_onehot) # The adersarial sample is treated as a regularization term and the weight is set as 0.5 following previous works
                                 
                    if epoch >= args.epochs_before_penalty:
                        loss += args.penalty_separation_weight * penalty
                else:
                    # For ANN, there's no built-in penalty, just forward with optional layer noise
                    outputs, penalty, spiking_rates = model(images)
                    loss = criterion(outputs, label_onehot)

                    if args.adversarial_training:
                        permuted_outputs, _,  _ = model(permuted_images, args.noisy_thres)
                        loss += criterion(permuted_outputs, label_onehot)
                                    
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).detach().sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if args.parseval_normalization:
                for layer in model.modules():
                    parseval_projection(layer, beta=0.0002)
                
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Train acc: {correct/total: .4f}")
        
        if args.model_type == 'SNN':
            print("Spiking Rates (last batch):", {k: f"{v.item():.4f}" for k, v in spiking_rates.items()})
            
        lr_scheduler.step()
        
        if epoch%args.evaluate_interval == args.evaluate_interval-1 and epoch >=80:
            epsilons = list(range(0, 13))  
            epsilons = [(10)**(db/2.5) for db in epsilons]
            
            accuracies = evaluate_spike_attack_multiple_epsilons(
                model,
                test_loader,
                device,
                epsilons,
                max_step_per_pixel=50,
                targeted=False
            )
            print("All accuracies:", accuracies) 
            

    torch.save(model.state_dict(), 'saved_model_dvs_AT')
    epsilons = list(range(0, 26))  
    epsilons = [(10)**(db/5) for db in epsilons]
    
    accuracies = evaluate_spike_attack_multiple_epsilons(
        model,
        test_loader,
        device,
        epsilons,
        max_step_per_pixel=50,
        targeted=False
    )
    
    print("All accuracies:", accuracies)
