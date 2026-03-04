from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch


@dataclass
class ReplayConfig:
    capacity: int = 2000
    sample_size: int = 0


class ReplayBuffer:
    def __init__(self, cfg: ReplayConfig, seed: int, device: torch.device):
        """Initialize storage and RNG using seed. Store tensors for 'x','y','task_id'."""
        ...

    def add(self, batch):
        """Insert current-task batch using an O(1) amortized policy."""
        ...

    def sample(self, k: int) -> Optional[dict]:
        """Return a batch dict or None if insufficient data."""
        ...

    def size(self) -> int: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...
