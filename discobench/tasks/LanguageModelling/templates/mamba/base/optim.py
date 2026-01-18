import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    # AdamW settings
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # Learning rate schedule settings
    num_iterations: int = 500
    warmup_iters: int = 50
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio


def create_optimizers(
    model: nn.Module, config: OptimizerConfig
) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Creates and returns optimizers and schedulers for the Mamba model.

    Uses AdamW with weight decay applied to most parameters except embeddings,
    biases, and normalization weights.

    Args:
        model (nn.Module): The Mamba model to optimize.
        config (OptimizerConfig): The configuration for the optimizer.

    Returns:
        optimizers (List[torch.optim.Optimizer]): List containing the AdamW optimizer.
        schedulers (List[torch.optim.lr_scheduler.LRScheduler]): List containing the LR scheduler.
    """
    # Separate parameters into those with weight decay and those without
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for embeddings, biases, and normalization weights
        if "embedding" in name or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        fused=True,
    )

    # Cosine learning rate schedule with warmup
    def get_lr(it):
        # Linear warmup
        if it < config.warmup_iters:
            return (it + 1) / config.warmup_iters
        # Cosine decay after warmup
        if it >= config.num_iterations:
            return config.min_lr_ratio
        decay_ratio = (it - config.warmup_iters) / (config.num_iterations - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr_ratio + coeff * (1.0 - config.min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    return [optimizer], [scheduler]
