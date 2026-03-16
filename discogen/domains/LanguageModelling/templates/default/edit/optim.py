from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    # Add your optimizer (and scheduler) hyperparameters here
    ...


def create_optimizers(
    model: nn.Module, config: OptimizerConfig
) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Creates and returns optimizers and schedulers for different parts of the model.

    Args:
        model (nn.Module): The model with weights to optimize (Look at the networks.py file for the model structure).
        config (OptimizerConfig): The configuration (i.e. hyperparameters) for the optimizers.

    Returns:
        optimizers (List[torch.optim.Optimizer]): The optimizers for the different parts of the model.
        schedulers (List[torch.optim.lr_scheduler.LRScheduler]): The schedulers for the different parts of the model.
    """

    # Create optimizers for different parts of the model
    optimizers = ...

    # Create learning rate scheduler
    schedulers = ...

    return optimizers, schedulers
