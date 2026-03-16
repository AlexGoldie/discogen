import torch.nn.functional as F
import torch
import torch.nn as nn


def compute_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, num_items_in_batch: Optional[int] = None) -> torch.FloatTensor:
    """Calculate the loss for the model's outputs against the true labels.

    Args:
        outputs (dict): The model's outputs, typically containing logits {"logits": logits}.
        labels (torch.Tensor): The true labels for the batch.
        num_items_in_batch (int, optional): The number of items in the batch. Defaults to None.

    Returns:
        torch.FloatTensor: The computed loss value.
    """

    """Fill in your loss calculation here."""
    loss = ...

    return loss
