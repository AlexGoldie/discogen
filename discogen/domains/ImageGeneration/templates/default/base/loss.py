import torch
import torch.nn.functional as F
from einops import reduce


def compute_loss(model_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the error loss between model outputs and targets.

    Args:
        model_out (torch.Tensor): The model's output tensor of shape
            `(B, ...)`, where `B` is the batch size and `...` represents
            arbitrary additional dimensions.
        target (torch.Tensor): The target tensor with the same shape as
            `model_out`.

    Returns:
        torch.Tensor: A tensor of shape `(B,)` containing the mean loss
        value for each element in the batch.
    """
    loss = F.mse_loss(model_out, target, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss
