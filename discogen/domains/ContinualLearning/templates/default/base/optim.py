"""Optimizer module: builds a torch optimizer from config."""

from dataclasses import dataclass
from typing import Iterable, Tuple
import torch.optim as torch_optim


@dataclass
class OptimizerConfig:
    name: str = "sgd"  # "sgd", "adam", "adamw"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    nesterov: bool = False


def build_optimizer(parameters: Iterable, cfg: OptimizerConfig):
    n = cfg.name.lower()
    if n == "sgd":
        return torch_optim.SGD(
            parameters, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=cfg.nesterov
        )
    if n == "adam":
        return torch_optim.Adam(
            parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
    if n == "adamw":
        return torch_optim.AdamW(
            parameters, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer {cfg.name}")
