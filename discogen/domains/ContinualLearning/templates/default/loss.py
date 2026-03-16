"""Classification loss."""

import torch
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
