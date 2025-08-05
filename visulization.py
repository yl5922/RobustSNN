# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 14:46:49 2025

@author: yl5922
"""
import os
import torch
import numpy as np
import matplotlib
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.cuda.amp import autocast
import math
from torchvision import transforms
from utils import spike_attack, reset_net

class_map = {
    0: "Hand Clapping",
    1: "Right HW",
    2: "Other",
    3: "Left HW",
    4: "Right Arm CW",
    5: "Right Arm CCW",
    6: "Left Arm CW",
    7: "Left Arm CCW",
    8: "Air Rolls",
    9: "Air Drums",
    10: "Air Guitar" 
    }

def collect_adversarial_events_for_plot(model, dataloader, device, epsilon=3.0, max_step_per_pixel=50, targeted=False, use_amp = True):
    """
    Collect spike events with strength (integrated spikes), for original and adversarial inputs.
    Assumes input shape: [B, T, C, H, W]

    Returns:
        original_events_list: list of Tensors (N, 4)  [x, y, t, count]
        adversarial_events_list: list of Tensors (N, 4)
    """
    model.eval()
    original_events_list = []
    adversarial_events_list = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        adv_inputs = spike_attack(
            model, inputs, labels,
            epsilon=epsilon,
            max_step_per_pixel=max_step_per_pixel,
            targeted=targeted,
            use_amp=use_amp
        ).detach()

        B, T, C, H, W = inputs.shape
        for b in range(B):
            ori_events = []
            adv_events = []
            for t in range(T):
                # ⚠️ Only use first channel (ON spikes)
                ori_frame = inputs[b, t, 0]      # shape: [H, W]
                adv_frame = adv_inputs[b, t, 0]  # shape: [H, W]

                ori_coords = (ori_frame > 0).nonzero(as_tuple=False)  # [N, 2]
                adv_coords = (adv_frame > 0).nonzero(as_tuple=False)

                if ori_coords.numel() > 0:
                    strength = ori_frame[ori_coords[:, 0], ori_coords[:, 1]]
                    t_col = torch.full_like(strength, t)
                    x = ori_coords[:, 1]
                    y = ori_coords[:, 0]
                    evt = torch.stack([x, y, t_col, strength], dim=1)
                    ori_events.append(evt)

                if adv_coords.numel() > 0:
                    strength = adv_frame[adv_coords[:, 0], adv_coords[:, 1]]
                    t_col = torch.full_like(strength, t)
                    x = adv_coords[:, 1]
                    y = adv_coords[:, 0]
                    evt = torch.stack([x, y, t_col, strength], dim=1)
                    adv_events.append(evt)

            ori_tensor = torch.cat(ori_events, dim=0).cpu() if ori_events else torch.zeros((0, 4))
            adv_tensor = torch.cat(adv_events, dim=0).cpu() if adv_events else torch.zeros((0, 4))
            original_events_list.append(ori_tensor)
            adversarial_events_list.append(adv_tensor)

    return original_events_list, adversarial_events_list

def plot_original_event_with_projection(original_event, labels=None, index=0, scale=1.5, view_angle=(22, 18), save_path=None):
    class_str = ""
    if labels is not None:
        label = labels[index].item() if isinstance(labels[index], torch.Tensor) else labels[index]
        class_str = f"{class_map.get(label, 'Unknown')}"

    ori = original_event[index].cpu().numpy()
    max_t = int(np.ceil(ori[:, 2].max()) + 2)  # 增加1给上移后的点
    H = int(np.ceil(ori[:, 1].max()) + 1)
    W = int(np.ceil(ori[:, 0].max()) + 1)

    ori_vox = np.zeros((max_t, H, W), dtype=np.int32)
    for x, y, t, c in ori:
        x, y, t = int(x), int(y), int(t)
        if 0 <= x < W and 0 <= y < H and 0 <= t < max_t:
            ori_vox[t, y, x] += int(c)

    ori_proj = ori_vox.sum(axis=0)
    ori_coords = np.nonzero(ori_vox)
    ori_strength = ori_vox[ori_coords]
    ori_points = np.stack([ori_coords[2], ori_coords[0], ori_coords[1], ori_strength], axis=1)

    ori_points = ori_points[np.random.rand(len(ori_points)) < 0.5]

    ori_points[:, 0] = W - 1 - ori_points[:, 0]
    ori_points[:, 1] += 1  # T + 1 避免遮挡底图
    ori_points[:, 2] = H - 1 - ori_points[:, 2]

    xx, zz = np.meshgrid(np.arange(W), np.arange(H))
    xx = W - 1 - xx
    zz = H - 1 - zz
    yy = np.zeros_like(xx)

    ori_proj_log = np.log1p(ori_proj)  # log(1 + x)，避免 log(0)
    ori_clip = ori_proj_log / ori_proj_log.max()  # 归一化到 [0, 1]
    viridis_cmap = matplotlib.colormaps["viridis"]

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(class_str, fontsize=24, y=0.87)

    ax = fig.add_subplot(111, projection='3d')
    s_ori = np.clip(ori_points[:, 3] * scale, 2, 10)

    ax.scatter(
        ori_points[:, 0], ori_points[:, 1], ori_points[:, 2],
        s=s_ori, c='#32cd32', alpha=0.3, label='Spikes'
    )#original color is #4daf4a

    ax.plot_surface(
        xx, yy, zz,
        facecolors=viridis_cmap(ori_clip),
        rstride=1, cstride=1,
        shade=False,
        antialiased=False
    )

    ax.set_xlim(0, W)
    ax.set_ylim(0, max_t-1)
    ax.set_zlim(0, H)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("Y")
    
    ax.set_xlabel("X", fontsize=20, labelpad=10)
    ax.set_ylabel("T", fontsize=20, labelpad=10)
    ax.set_zlabel("Y", fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='z', which='major', labelsize=15)

    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    import matplotlib.lines as mlines
    custom_legend = [mlines.Line2D([], [], color='#32cd32', marker='o', linestyle='None',
                                markersize=8, label='Spikes', alpha=0.6)]

    fig.legend(handles=custom_legend,
               loc='center',
               bbox_to_anchor=(0.70, 0.81),
               fontsize=18,
               frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path,  bbox_inches='tight', dpi=300)
    plt.show()

def plot_perturbation_event_with_projection(original_event, adversarial_event, index=0, epsilon=None, scale=2.0, view_angle=(22, 18), save_path=None):
    ori = original_event[index].cpu().numpy()
    adv = adversarial_event[index].cpu().numpy()
    max_t = int(np.ceil(max(ori[:, 2].max(), adv[:, 2].max()) + 2))
    H = int(np.ceil(max(ori[:, 1].max(), adv[:, 1].max()) + 1))
    W = int(np.ceil(max(ori[:, 0].max(), adv[:, 0].max()) + 1))

    ori_vox = np.zeros((max_t, H, W), dtype=np.int32)
    adv_vox = np.zeros_like(ori_vox)
    for x, y, t, c in ori:
        x, y, t = int(x), int(y), int(t)
        if 0 <= x < W and 0 <= y < H and 0 <= t < max_t:
            ori_vox[t, y, x] += int(c)
    for x, y, t, c in adv:
        x, y, t = int(x), int(y), int(t)
        if 0 <= x < W and 0 <= y < H and 0 <= t < max_t:
            adv_vox[t, y, x] += int(c)

    delta_vox = adv_vox - ori_vox
    perturb_coords = np.nonzero(delta_vox)
    perturb_values = delta_vox[perturb_coords]
    print(f"Total perturbed spikes: {(perturb_values != 0).sum()}")

    pos_mask = perturb_values > 0
    neg_mask = perturb_values < 0

    pos_points = np.stack([perturb_coords[2][pos_mask], perturb_coords[0][pos_mask], perturb_coords[1][pos_mask], perturb_values[pos_mask]], axis=1)
    neg_points = np.stack([perturb_coords[2][neg_mask], perturb_coords[0][neg_mask], perturb_coords[1][neg_mask], -perturb_values[neg_mask]], axis=1)

    for pts in (pos_points, neg_points):
        pts[:, 0] = W - 1 - pts[:, 0]
        pts[:, 1] += 1  # T + 1
        pts[:, 2] = H - 1 - pts[:, 2]

    adv_proj = np.abs(delta_vox).sum(axis=0)
    adv_proj_log = np.log1p(adv_proj)
    adv_clip = adv_proj_log / adv_proj_log.max()

    xx, zz = np.meshgrid(np.arange(W), np.arange(H))
    xx = W - 1 - xx
    zz = H - 1 - zz
    yy = np.zeros_like(xx)

    viridis_cmap = matplotlib.colormaps["viridis"]

    fig = plt.figure(figsize=(10, 7))
    title_text = f"Perturbation" if epsilon is None else f"Perturbation (ε={epsilon})"
    # fig.suptitle(title_text, fontsize=14, y=0.90)
    ax = fig.add_subplot(111, projection='3d')

    if len(pos_points) > 0:
        s_pos = np.clip(pos_points[:, 3] * scale, 2, 10)
        ax.scatter(
            pos_points[:, 0], pos_points[:, 1], pos_points[:, 2],
            s=s_pos, c='#87cefa', alpha=0.9, label='False Alarm'
        ) #orginal color #984ea3

    if len(neg_points) > 0:
        s_neg = np.clip(neg_points[:, 3] * scale, 2, 10)
        ax.scatter(
            neg_points[:, 0], neg_points[:, 1], neg_points[:, 2],
            s=s_neg, c='#ff6f61', alpha=0.9, label='Missing'
        ) #orginal color #ef3b2c

    ax.plot_surface(
        xx, yy, zz,
        facecolors=viridis_cmap(adv_clip),
        rstride=1, cstride=1,
        shade=False,
        antialiased=False
    )

    ax.set_xlim(0, W)
    ax.set_ylim(0, max_t-1)
    ax.set_zlim(0, H)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("Y")
    ax.set_xlabel("X", fontsize=20, labelpad=10)
    ax.set_ylabel("T", fontsize=20, labelpad=10)
    ax.set_zlabel("Y", fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='z', which='major', labelsize=15)
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    import matplotlib.lines as mlines
    custom_legend = [
        mlines.Line2D([], [], color='#87cefa', marker='o', linestyle='None',
                      markersize=8, label='False Alarm', alpha=0.9),
        mlines.Line2D([], [], color='#ff6f61', marker='o', linestyle='None',
                      markersize=8, label='Missing', alpha=0.9)
    ]
    
    # 添加图例
    fig.legend(
        handles=custom_legend,
        loc='center',
        bbox_to_anchor=(0.70, 0.87),
        fontsize=17,
        frameon=False
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight',dpi=300)
    plt.show()

def visualize_events_over_epsilons(model, model_name, dataloader, device, label_list, epsilons, base_dir="./visualizations", max_per_class=4, use_amp = True):
    """
    For each epsilon, generate event visualizations and save them into class-specific folders under the model_name directory.
    """
    model_name = os.path.splitext(model_name)[0]  # remove ".pt"
    save_dir = os.path.join(base_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    for eps in epsilons:
        print(f"Generating plots for epsilon={eps}...")
        ori_events, adv_events = collect_adversarial_events_for_plot(
            model=model,
            dataloader=dataloader,
            device=device,
            epsilon=eps,
            max_step_per_pixel=50,
            targeted=False,
            use_amp=use_amp
        )

        class_counter = defaultdict(int)

        for i in range(len(ori_events)):
            cls_label = label_list[i] if i < len(label_list) else "unknown"
            class_str = f"class_{cls_label}"
            class_dir = os.path.join(save_dir, class_str)
            os.makedirs(class_dir, exist_ok=True)

            if class_counter[cls_label] >= max_per_class:
                continue

            ori_path = os.path.join(class_dir, f"sample{i:03d}_eps{eps}_original.png")
            pert_path = os.path.join(class_dir, f"sample{i:03d}_eps{eps}_perturbation.png")

            plot_original_event_with_projection(
                original_event=ori_events,
                labels=label_list,
                index=i,
                save_path=ori_path
            )
            plot_perturbation_event_with_projection(
                original_event=ori_events,
                adversarial_event=adv_events,
                index=i,
                epsilon=eps,
                save_path=pert_path
            )

            class_counter[cls_label] += 1

def visualize_events_over_epsilons_success_only(
    model,
    model_name,
    dataloader,
    device,
    label_list,
    epsilons,
    success_samples,
    base_dir="./visualizations",
    max_per_class=4,
    use_amp=True
):
    """
    Only visualize samples from success_samples list, selecting at most max_per_class samples per class.
    For each selected sample, generate visualizations across all epsilons.
    """

    model_name = os.path.splitext(model_name)[0]
    save_dir = os.path.join(base_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Select at most max_per_class samples per class
    selected_by_class = defaultdict(list)
    for idx, true_label, _, _ in success_samples:
        if len(selected_by_class[true_label]) < max_per_class:
            selected_by_class[true_label].append(idx)

    # Flatten selected indices
    selected_indices = []
    for label in selected_by_class:
        for idx in selected_by_class[label]:
            selected_indices.append((idx, label))

    # For each epsilon, generate and save visualizations
    for eps in epsilons:
        print(f"Generating plots for epsilon={eps}...")
        ori_events, adv_events = collect_adversarial_events_for_plot(
            model=model,
            dataloader=dataloader,
            device=device,
            epsilon=eps,
            max_step_per_pixel=50,
            targeted=False,
            use_amp=use_amp
        )

        for idx, cls_label in selected_indices:
            class_str = f"class_{cls_label}"
            class_dir = os.path.join(save_dir, class_str)
            os.makedirs(class_dir, exist_ok=True)

            ori_path = os.path.join(class_dir, f"sample{idx:03d}_eps{eps}_original.png")
            pert_path = os.path.join(class_dir, f"sample{idx:03d}_eps{eps}_perturbation.png")

            plot_original_event_with_projection(
                original_event=ori_events,
                labels=label_list,
                index=idx,
                save_path=ori_path
            )
            plot_perturbation_event_with_projection(
                original_event=ori_events,
                adversarial_event=adv_events,
                index=idx,
                epsilon=eps,
                save_path=pert_path
            )

def visualize_selected_event_samples(
    model,
    model_name,
    dataloader,
    device,
    label_list,
    epsilons,
    selected_samples,
    base_dir="./visualizations",
    use_amp=True
):
    """
    Visualize a fixed set of selected samples (by class/sample index) over a list of epsilons.
    Each sample should be in the form: (class_str, sample_index) such as ("class_0", 4)
    """

    model_name = os.path.splitext(model_name)[0]
    save_dir = os.path.join(base_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert to set of (class, idx) for fast lookup
    selected_set = set(selected_samples)

    for eps in epsilons:
        print(f"Generating plots for epsilon={eps}...")
        ori_events, adv_events = collect_adversarial_events_for_plot(
            model=model,
            dataloader=dataloader,
            device=device,
            epsilon=eps,
            max_step_per_pixel=50,
            targeted=False,
            use_amp=use_amp
        )

        for i in range(len(ori_events)):
            cls_label = label_list[i] if i < len(label_list) else "unknown"
            class_str = f"class_{cls_label}"

            if (class_str, i) not in selected_set:
                continue

            class_dir = os.path.join(save_dir, class_str)
            os.makedirs(class_dir, exist_ok=True)

            ori_path = os.path.join(class_dir, f"sample{i:03d}_eps{eps}_original.png")
            pert_path = os.path.join(class_dir, f"sample{i:03d}_eps{eps}_perturbation.png")

            plot_original_event_with_projection(
                original_event=ori_events,
                labels=label_list,
                index=i,
                save_path=ori_path
            )
            plot_perturbation_event_with_projection(
                original_event=ori_events,
                adversarial_event=adv_events,
                index=i,
                epsilon=eps,
                save_path=pert_path
            )
            
def evaluate_flip_probability(model, dataloader, device, epsilon=3.0, max_step_per_pixel=50, targeted=False, use_amp = True):
    """
    Evaluate model accuracy and average per-layer spike flip probability (scalar in [0, 1]).
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    flip_sum = {}     # Total number of flips per layer
    elem_sum = {}     # Total number of elements compared per layer

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        adv_inputs = spike_attack(model, inputs, labels,
                                  epsilon=epsilon,
                                  max_step_per_pixel=max_step_per_pixel,
                                  targeted=targeted,use_amp=use_amp).detach()

        with torch.no_grad():
            reset_net(model)
            outputs_clean, _, spikes_clean = model(inputs)
            reset_net(model)
            outputs_adv, _, spikes_adv = model(adv_inputs)

            preds_adv = outputs_adv.argmax(dim=1)

        total_correct += (preds_adv == labels).sum().item()
        total_samples += labels.size(0)

        for k in spikes_clean:
            clean = spikes_clean[k]         # [B, T, ...]
            adv = spikes_adv[k]             # same shape
            flips = (clean != adv).float()  # [B, T, ...]
            flip_sum[k] = flip_sum.get(k, 0.0) + flips.sum().item()
            elem_sum[k] = elem_sum.get(k, 0.0) + flips.numel()

    flip_probabilities = {k: flip_sum[k] / elem_sum[k] for k in flip_sum}  # value ∈ [0, 1]
    accuracy = total_correct / total_samples
    print(f"Accuracy under spike-based PGD attack: {accuracy * 100:.2f}%")

    return accuracy, flip_probabilities

def collect_flip_rates_over_epsilons(model, dataloader, device, epsilon_list, max_step_per_pixel=50, targeted=False, use_amp = True):
    """
    Returns:
        {epsilon: {layer_name: flip_rate}}
    """
    results = {}
    for eps in epsilon_list:
        print(f"[ε = {eps}] Evaluating...")
        _, flip_rates = evaluate_flip_probability(
            model, dataloader, device,
            epsilon=eps,
            max_step_per_pixel=max_step_per_pixel,
            targeted=targeted,
            use_amp=use_amp
        )
        results[eps] = flip_rates
    return results

def plot_flip_rate_heatmap(
    flip_rate_dict,
    firing_rate_dict=None,
    normalize=False,
    log_scale=False,
    vmin=None,
    vmax=None,
    title="Flip Rate Heatmap",
    base_dir="./visualizations"
):
    """
    flip_rate_dict: dict[epsilon][layer] = float
    firing_rate_dict: optional, same structure
    normalize: if True, normalize flip rate using firing rate
    log_scale: if True, apply log10 to flip rates
    vmin/vmax: fixed colorbar range (after log if log_scale=True)
    
    Returns: DataFrame used for plotting
    """
    # Normalize if needed
    if normalize:
        assert firing_rate_dict is not None, "You must provide firing_rate_dict if normalize=True"
        normalized = {}
        for eps in flip_rate_dict:
            normalized[eps] = {
                layer: flip_rate_dict[eps][layer] / (firing_rate_dict.get(layer, 1e-10) + 1e-10)
                for layer in flip_rate_dict[eps]
            }
        data = normalized
    else:
        data = flip_rate_dict

    df = pd.DataFrame.from_dict(data, orient="index")
    # Rename epsilon index to dB scale
    epsilons = df.index.to_numpy(dtype=float)
    eps_dB = [f"{10 * np.log10(e + 1e-10):.0f}" for e in epsilons]
    df.index = eps_dB
    df.index.name = r"$\epsilon$ (dB)"

    layer_display_names = {
        "layer1": "conv1",
        "layer2": "conv2",
        "layer3": "conv3",
        "layer4": "conv4",
        "layer5": "conv5",
        "layer6": "fc1",
        "layer7": "fc2"
    }
    df.rename(columns=layer_display_names, inplace=True)
    
    # Apply log scale if requested
    if log_scale:
        df = df.applymap(lambda x: np.log10(x + 1e-6))  # prevent log(0)

    # Plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f" if log_scale else ".3f",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        # cbar_kws={'label': 'log10(Flip Rate)' if log_scale else ('Normalized Flip Rate' if normalize else 'Flip Rate')}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=17)
    for text in ax.texts:
        text.set_fontsize(17)
    # ax.set_title(title + (" (Normalized)" if normalize else "") + (" (log10)" if log_scale else ""),fontsize=14)
    ax.set_xlabel("Layer", fontsize=20)
    ax.set_ylabel(r"$\epsilon$ (dB)", fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    
    # Save figure
    save_dir = os.path.join(base_dir, "flip_rate")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title.replace(" ", "_").replace("(", "").replace(")", "") + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return df

def evaluate_firing_rate(model, dataloader, device):
    """
    Evaluate average per-layer firing rate over dataset.
    
    Returns:
        firing_rates: dict[layer_name: float] ∈ [0, 1]
    """
    model.eval()
    spike_sum = {}   # 放电总和
    elem_sum = {}    # 总元素个数

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            reset_net(model)
            _, _, spikes = model(inputs)  # spikes: dict[layer_name: Tensor[B, T, ...]]

        for k in spikes:
            s = spikes[k]                        # [B, T, ...]
            spike_sum[k] = spike_sum.get(k, 0.0) + s.sum().item()
            elem_sum[k] = elem_sum.get(k, 0.0) + s.numel()

    firing_rates = {k: spike_sum[k] / elem_sum[k] for k in spike_sum}
    return firing_rates

def compute_confusion_matrix_under_attack(
    model, dataloader, device,
    epsilon=3.0, max_step_per_pixel=50, targeted=False,
    num_classes=11, use_amp = True
):
    """
    计算模型在对抗攻击下的混淆矩阵（未可视化）。
    返回：
        confusion_matrix: ndarray (num_classes, num_classes)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        adv_inputs = spike_attack(model, inputs, labels,
                                  epsilon=epsilon,
                                  max_step_per_pixel=max_step_per_pixel,
                                  targeted=targeted, use_amp=use_amp).detach()

        with torch.no_grad():
            reset_net(model)
            outputs, _, _ = model(adv_inputs)
            preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    cm = confusion_matrix(
        all_targets, all_preds, labels=list(range(num_classes))
    )
    return cm

def plot_feature_maps_comparison(
    model,
    input_tensor,
    adv_tensor,
    top_k=5,
    use_log=True,
    layer_names=None,
    save_dir=None
):
    """
    Plot 2x6 grid per layer: 1 row for original input + topk features, 1 row for adversarial input + features.
    Optionally save the figure to `save_dir` if provided.
    """
    model.eval()
    with torch.no_grad():
        reset_net(model)
        _, _, spikes_clean = model(input_tensor)
        reset_net(model)
        _, _, spikes_adv = model(adv_tensor)

    if layer_names is None:
        layer_names = [f"layer{i+1}" for i in range(5)]

    input_proj_clean = input_tensor[0].sum(dim=1).cpu().numpy().sum(axis=0)
    input_proj_adv = adv_tensor[0].sum(dim=1).cpu().numpy().sum(axis=0)

    if use_log:
        input_proj_clean = np.log1p(input_proj_clean)
        input_proj_adv = np.log1p(input_proj_adv)

    for lname in layer_names:
        spike_clean = spikes_clean[lname][0]  # [T, C, H, W]
        spike_adv = spikes_adv[lname][0]      # same shape

        spike_sum_per_channel = spike_clean.sum(dim=(0, 2, 3))  # [C]
        _, topk_indices = torch.topk(spike_sum_per_channel, k=top_k)

        fig, axes = plt.subplots(2, top_k+1, figsize=(20, 6.3))
        axes = axes.reshape(2, top_k+1)

        # Clean row
        axes[0, 0].imshow(input_proj_clean, cmap="viridis")
        axes[0, 0].set_title("Input", fontsize=15)
        axes[0, 0].axis("off")

        for i, idx in enumerate(topk_indices):
            fmap = spike_clean[:, idx].sum(dim=0).cpu().numpy()
            if use_log:
                fmap = np.log1p(fmap)
            axes[0, i+1].imshow(fmap, cmap="plasma")
            axes[0, i+1].set_title(f"Ch {idx.item()}", fontsize=15)
            axes[0, i+1].axis("off")

        # Adversarial row
        axes[1, 0].imshow(input_proj_adv, cmap="viridis")
        axes[1, 0].set_title("Adv Input", fontsize=15)
        axes[1, 0].axis("off")

        for i, idx in enumerate(topk_indices):
            fmap = spike_adv[:, idx].sum(dim=0).cpu().numpy()
            if use_log:
                fmap = np.log1p(fmap)
            axes[1, i+1].imshow(fmap, cmap="plasma")
            axes[1, i+1].set_title(f"Ch {idx.item()}", fontsize=15)
            axes[1, i+1].axis("off")

        # plt.suptitle(f"{lname} Feature Maps (Top {top_k})", fontsize=16)
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{lname}_feature_maps.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()
        
def find_successful_adversarial_samples(model, dataloader, device, epsilon=3e4, batch_size=16, max_step_per_pixel=50, use_amp=False):
    """
    返回所有对抗攻击成功的样本索引和预测变化（idx, true_label, clean_pred, adv_pred）
    """
    model.eval()
    successful_samples = []

    for idx, (inputs_np, labels) in enumerate(dataloader):
        input_tensor = inputs_np.to(device)
        label = labels.to(device)

        with torch.no_grad():
            reset_net(model)
            output_clean, _, _ = model(input_tensor)
            pred_clean = output_clean.argmax(dim=1)

        adv_inputs = spike_attack(
            model, input_tensor, label,
            epsilon=epsilon,
            max_step_per_pixel=max_step_per_pixel,
            targeted=None,
            use_amp=use_amp
        ).detach()

        with torch.no_grad():
            reset_net(model)
            output_adv, _, _ = model(adv_inputs)
            pred_adv = output_adv.argmax(dim=1)

        for i in range(len(label)):
            if pred_clean[i] != pred_adv[i]:
                successful_samples.append((idx * batch_size + i, label[i].item(), pred_clean[i].item(), pred_adv[i].item()))

    print("Successful adversarial samples:")
    for idx, true_label, clean_pred, adv_pred in successful_samples:
        print(f"Sample {idx}: True={true_label}, Clean Pred={clean_pred}, Adv Pred={adv_pred}")

    return successful_samples

def visualize_event_frames(x, pause=0.05, title_prefix="Time step"):
    """
    可视化 [T, 2, H, W] 的双极性事件张量。
    极性1 映射为绿色，极性2 映射为蓝色。

    参数:
    - x: numpy.ndarray 或 torch.Tensor，形状为 [T, 2, H, W]
    - pause: 每帧暂停时间（秒）
    - title_prefix: 显示在每帧图像上标题的前缀
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).float()
    else:
        x = x.float()

    to_img = transforms.ToPILImage()
    T, C, H, W = x.shape

    assert C == 2, f"Expected 2 polarity channels, got {C}"

    # 构建 RGB 图像，R=0, G=polarity1, B=polarity2
    img_tensor = torch.zeros((T, 3, H, W))
    img_tensor[:, 1] = x[:, 0]  # 极性1 -> Green
    img_tensor[:, 2] = x[:, 1]  # 极性2 -> Blue

    for t in range(T):
        img = to_img(torch.clamp(img_tensor[t], 0, 1))  # 避免非法值
        plt.imshow(img)
        plt.title(f"{title_prefix} {t}")
        plt.axis('off')
        plt.pause(pause)