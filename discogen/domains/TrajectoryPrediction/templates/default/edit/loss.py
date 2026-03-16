import torch
from typing import Dict, Any


def compute_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    config: Dict[str, Any]
) -> torch.Tensor:
    """Compute the trajectory prediction loss.
    Args:
        predictions: Model output containing:
            - 'predicted_trajectory': (B, K, T, 5) distribution parameters
            - 'predicted_probability': (B, K) mode probabilities
        batch: Contains 'input_dict' with:
            - 'center_gt_trajs': (B, T, 2) ground truth future positions
            - 'center_gt_trajs_mask': (B, T) validity mask
        config: Configuration dictionary with loss hyperparameters
    Returns:
        Scalar loss tensor with gradient
    """
    """Fill in your loss calculation here."""
    loss = ...
    return loss
