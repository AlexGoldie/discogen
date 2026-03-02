import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for next token prediction in language modeling. Make sure to ignore padding tokens,
    i.e. when the target token is -1.

    Args:
        logits (torch.Tensor): Model output logits of shape [batch_size, seq_len, vocab_size]
        targets (torch.Tensor): Target tokens of shape [batch_size, seq_len]

    Returns:
        torch.Tensor: The computed loss value
    """

    """Fill in your loss logic here."""
    loss = ...

    return loss
