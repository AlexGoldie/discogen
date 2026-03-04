"""Batch mixing policy."""

from dataclasses import dataclass
from typing import Dict
import math
import torch


@dataclass
class SamplerConfig:
    replay_ratio: float = 0.5
    min_replay_after: int = 1000


class BatchMixer:
    def __init__(self, cfg: SamplerConfig, seed: int):
        self.cfg = cfg
        self.rng = torch.Generator().manual_seed(seed)

    def mix(self, current: Dict[str, torch.Tensor], replay, final_batch_size: int):
        b = current["x"].size(0)
        k_rep = 0
        if replay.size() >= self.cfg.min_replay_after and self.cfg.replay_ratio > 0:
            k_rep = min(int(math.floor(final_batch_size * self.cfg.replay_ratio)), replay.size())
        k_cur = min(final_batch_size - k_rep, b)

        # Deterministic truncation of current
        idx_c = torch.arange(b, device=current["x"].device)[:k_cur]
        cur = {k: v.index_select(0, idx_c) for k, v in current.items()}

        # Sample from replay
        rep = replay.sample(k_rep) if k_rep > 0 else None
        if rep is None:
            return cur

        x = torch.cat([cur["x"], rep["x"]], dim=0)
        y = torch.cat([cur["y"], rep["y"]], dim=0)
        t = torch.cat([cur["task_id"], rep["task_id"]], dim=0)
        return {"x": x, "y": y, "task_id": t}
