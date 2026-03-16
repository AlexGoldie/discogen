import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple


def evaluate_model(
    model,
    val_data,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model and return metrics dict."""
    model.eval()

    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_metrics = {
        'minADE': [],
        'minFDE': [],
        'miss_rate': [],
        'brier_minFDE': []
    }

    miss_threshold = 2.0  # meters

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            batch = {'input_dict': batch}
            predictions = model(batch)

            # Get predictions and ground truth
            pred_trajs = predictions['predicted_trajectory']  # (B, K, T, 5)
            pred_probs = predictions['predicted_probability']  # (B, K)

            # Extract ground truth
            if 'input_dict' in batch:
                gt_trajs = batch['input_dict']['center_gt_trajs']  # (B, T, 2+)
                gt_mask = batch['input_dict']['center_gt_trajs_mask']  # (B, T)
            else:
                gt_trajs = batch['center_gt_trajs']
                gt_mask = batch['center_gt_trajs_mask']

            # Compute metrics for this batch
            batch_metrics = compute_batch_metrics(
                pred_trajs=pred_trajs[:, :, :, :2],  # Only x, y
                pred_probs=pred_probs,
                gt_trajs=gt_trajs[:, :, :2],  # Only x, y
                gt_mask=gt_mask,
                miss_threshold=miss_threshold
            )

            for key in all_metrics:
                all_metrics[key].extend(batch_metrics[key])

    # Aggregate metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            final_metrics[key] = np.mean(values)
        else:
            final_metrics[key] = 0.0

    return final_metrics


def compute_batch_metrics(
    pred_trajs: torch.Tensor,
    pred_probs: torch.Tensor,
    gt_trajs: torch.Tensor,
    gt_mask: torch.Tensor,
    miss_threshold: float = 2.0
) -> Dict[str, List[float]]:
    """Compute minADE, minFDE, miss_rate, brier_minFDE for a batch."""
    batch_size = pred_trajs.shape[0]
    num_modes = pred_trajs.shape[1]

    metrics = {
        'minADE': [],
        'minFDE': [],
        'miss_rate': [],
        'brier_minFDE': []
    }

    pred_trajs = pred_trajs.cpu().numpy()
    pred_probs = pred_probs.cpu().numpy()
    gt_trajs = gt_trajs.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    for b in range(batch_size):
        # Get valid timesteps
        valid_mask = gt_mask[b] > 0.5
        if valid_mask.sum() == 0:
            continue

        gt = gt_trajs[b]  # (T, 2)

        # Compute ADE and FDE for each mode
        ades = []
        fdes = []

        for k in range(num_modes):
            pred = pred_trajs[b, k]  # (T, 2)

            # ADE: average displacement over valid timesteps
            displacements = np.linalg.norm(pred - gt, axis=-1)  # (T,)
            ade = (displacements * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            ades.append(ade)

            # FDE: displacement at final valid timestep
            last_valid_idx = np.where(valid_mask)[0][-1] if valid_mask.any() else -1
            fde = displacements[last_valid_idx] if last_valid_idx >= 0 else 0.0
            fdes.append(fde)

        # MinADE and MinFDE (best mode)
        min_ade_idx = np.argmin(ades)
        min_fde_idx = np.argmin(fdes)

        metrics['minADE'].append(ades[min_ade_idx])
        metrics['minFDE'].append(fdes[min_fde_idx])

        # Miss rate (FDE > threshold for best mode)
        miss = 1.0 if fdes[min_fde_idx] > miss_threshold else 0.0
        metrics['miss_rate'].append(miss)

        # Brier-minFDE (minFDE weighted by probability of best mode)
        best_prob = pred_probs[b, min_fde_idx]
        brier_fde = fdes[min_fde_idx] + (1 - best_prob)
        metrics['brier_minFDE'].append(brier_fde)

    return metrics


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in batch to the specified device."""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = move_batch_to_device(value, device)
        else:
            moved_batch[key] = value
    return moved_batch
