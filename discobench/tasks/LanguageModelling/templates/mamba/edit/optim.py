from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    # Add your optimizer (and scheduler) hyperparameters here
    # Common settings for SSM models:
    # - learning_rate: typically 3e-4 to 1e-3
    # - weight_decay: typically 0.1
    # - warmup_iters: number of warmup iterations
    ...


def create_optimizers(
    model: nn.Module, config: OptimizerConfig
) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Creates and returns optimizers and schedulers for the Mamba model.

    Args:
        model (nn.Module): The model with weights to optimize (Look at the networks.py file for the model structure).
        config (OptimizerConfig): The configuration (i.e. hyperparameters) for the optimizers.

    Returns:
        optimizers (List[torch.optim.Optimizer]): The optimizers for the different parts of the model.
        schedulers (List[torch.optim.lr_scheduler.LRScheduler]): The schedulers for the different parts of the model.

    Note:
        For SSM models like Mamba, AdamW with weight decay is commonly used.
        Consider separating parameters into groups:
        - Embeddings, biases, norms: no weight decay
        - Other parameters: with weight decay
    """

    # Create optimizers for the model
    optimizers = ...

    # Create learning rate scheduler (e.g., cosine with warmup)
    schedulers = ...

    return optimizers, schedulers
