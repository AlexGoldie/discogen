from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
    """Calculate the loss for the model's outputs against the true labels.

    Args:
        outputs (torch.Tensor): The model's outputs, typically containing logits.
        labels (torch.Tensor): The true labels for the batch.

    Returns:
        torch.FloatTensor: The computed loss value.
    """

    """Fill in your loss calculation here."""
    loss = ...

    return loss
