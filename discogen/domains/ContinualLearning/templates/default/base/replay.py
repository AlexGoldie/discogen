"""Reservoir replay buffer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import random
import torch


@dataclass
class ReplayConfig:
    capacity: int = 2000
    sample_size: int = 0


class ReplayBuffer:
    def __init__(self, cfg: ReplayConfig, seed: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(seed)
        self._data: List[Dict[str, torch.Tensor]] = []
        self._seen = 0

    def add(self, batch: Dict[str, torch.Tensor]) -> None:
        x, y, task_id = batch["x"], batch["y"], batch["task_id"]
        for i in range(x.size(0)):
            item = {"x": x[i].detach().to(self.device), "y": y[i].detach().to(self.device), "task_id": task_id[i].detach().to(self.device)}
            self._seen += 1
            if len(self._data) < self.cfg.capacity:
                self._data.append(item)
            else:
                j = self.rng.randint(0, self._seen - 1)
                if j < self.cfg.capacity:
                    self._data[j] = item

    def sample(self, k: int) -> Optional[Dict[str, torch.Tensor]]:
        n = len(self._data)
        if n == 0 or k <= 0:
            return None
        k = min(k, n)
        idx = [self.rng.randrange(n) for _ in range(k)]
        xs = torch.stack([self._data[i]["x"] for i in idx], dim=0)
        ys = torch.stack([self._data[i]["y"] for i in idx], dim=0)
        ts = torch.stack([self._data[i]["task_id"] for i in idx], dim=0)
        return {"x": xs, "y": ys, "task_id": ts}

    def size(self) -> int:
        return len(self._data)

    def state_dict(self) -> Dict[str, Any]:
        return {"seen": self._seen, "data": self._data, "cfg": self.cfg.__dict__}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._seen = int(state.get("seen", 0))
        self._data = state.get("data", [])
