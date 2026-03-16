import torch
from typing import Dict, Any, Tuple, Optional


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Create an optimizer and optional learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): Configuration dictionary containing optimizer parameters.
            - learning_rate (float): The learning rate for the optimizer.
            - weight_decay (float): The weight decay factor.

    Returns:
        Tuple of (optimizer, scheduler or None).
    """

    """Fill in your optimizer creation logic here."""
    optimizer = ...
    scheduler = ...
    return optimizer, scheduler
