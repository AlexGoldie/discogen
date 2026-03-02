from torch import optim
from typing import Dict
import torch


def create_optimizer(model: torch.nn.Module, config: Dict[str, float]) -> "Optimizer":
    """
    Create an optimizer for the given model and configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): Configuration dictionary containing optimizer parameters.
            - learning_rate (float): The learning rate for the optimizer.
            - weight_decay (float): The weight decay (L2 regularization) factor.

    Returns:
        Optimizer: An optimizer instance with functions including step and zero_grad. A subclass of torch.optim.Optimizer might be a good choice.
    """

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no decay for 1D params (norm/bias) and explicit bias tensors
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
         lr=learning_rate
    )

    return optimizer
