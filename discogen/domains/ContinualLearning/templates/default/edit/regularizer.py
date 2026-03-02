from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn


@dataclass
class RegularizerConfig:
    importance_estimation_batches: int = 10
    importance_decay: float = 1.0
    max_val_batches: int = 50
    eps: float = 1e-8


class Regularizer:
    def __init__(self, cfg: RegularizerConfig, seed: int):
        """Initialize any buffers; do not modify model params."""
        ...

    def on_task_start(self, task_id: int, model: nn.Module) -> None:
        """Prepare per-task state. No param changes."""
        ...

    def compute_penalty(self, model: nn.Module, step: int) -> torch.Tensor:
        """Return scalar penalty tensor on model device; keep cheap per step."""
        ...

    def on_task_end(self, task_id: int, model: nn.Module, val_loader) -> None:
        """Estimate per-parameter importance from val_loader; snapshot old params."""
        ...

    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...
