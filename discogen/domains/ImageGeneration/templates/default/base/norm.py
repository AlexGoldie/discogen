import torch


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an input tensor from the [0, 1] range to the [-1, 1] range.

    Args:
        img (torch.Tensor): Input tensor with values in the range [0, 1].

    Returns:
        torch.Tensor: Tensor of the same shape as `img` with values
        rescaled to the range [-1, 1].
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from the [-1, 1] range back to the [0, 1] range.

    Args:
        t (torch.Tensor): Input tensor with values in the range [-1, 1].

    Returns:
        torch.Tensor: Tensor of the same shape as `t` with values
        rescaled to the range [0, 1].
    """
    return (t + 1) * 0.5
