import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from networks import TrajectoryPredictionModel
from loss import compute_loss
from optim import create_optimizer


def train_model(
    train_data,
    val_data,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[TrajectoryPredictionModel, Dict[str, List[float]]]:
    """Train the model and return it along with training history.

    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        config: Configuration dictionary containing:
            - batch_size (int): Batch size for data loaders
            - num_epochs (int): Number of training epochs
            - max_grad_norm (float): Maximum gradient norm for clipping
        device: Device to train on.

    Returns:
        Tuple of (trained model, training history dict with keys
        'train_loss', 'val_loss', 'lr' each mapping to a list of per-epoch values).
        You must NOT print or log scores/metrics; they are recorded via the history dict.
    """
    """Fill in your training loop here."""
    model = ...
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    return model, history
