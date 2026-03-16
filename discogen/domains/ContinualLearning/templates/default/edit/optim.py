from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class OptimizerConfig:
    name: str = ...
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def build_optimizer(parameters: Iterable, cfg: OptimizerConfig):
    """Return a torch optimizer. You may use the config above, and edit any rows.
    """
    ...
