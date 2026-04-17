from typing import Any

import torch
from torch import nn
from torch import optim


def get_optim(model: nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    """
    Create and return an optimizer for the given model.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        config (dict): Configuration dictionary containing optimizer settings.
            Expected to have an `"optimizer"` key whose value is a dictionary
            of keyword arguments accepted by `torch.optim.Adam`
            (e.g., learning rate, betas, weight decay).

    Returns:
        torch.optim.Optimizer: An initialized Adam optimizer configured
        according to `config`.
    """
    optimizer = ...
    return optimizer
