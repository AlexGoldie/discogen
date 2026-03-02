"""Step-based scheduler with warmup and cosine/step decay."""

from dataclasses import dataclass
import math


@dataclass
class SchedulerConfig:
    kind: str = "cosine"  # "cosine" or "step"
    warmup_steps: int = 500
    total_steps: int = 10000
    step_size: int = 3000
    gamma: float = 0.1


class _WarmupCosine:
    def __init__(self, optimizer, warmup: int, total: int):
        self.opt = optimizer
        self.warmup = max(1, warmup)
        self.total = max(self.warmup + 1, total)
        self.step_idx = 0
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        s = self.step_idx
        if s < self.warmup:
            factor = s / float(self.warmup)
        else:
            s2 = s - self.warmup
            t2 = max(1, self.total - self.warmup)
            factor = 0.5 * (1 + math.cos(math.pi * s2 / t2))
        for lr, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = lr * factor
        self.step_idx += 1


class _WarmupStep:
    def __init__(self, optimizer, warmup: int, step_size: int, gamma: float):
        self.opt = optimizer
        self.warmup = max(1, warmup)
        self.step_size = max(1, step_size)
        self.gamma = gamma
        self.step_idx = 0
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        s = self.step_idx
        if s < self.warmup:
            factor = s / float(self.warmup)
        else:
            k = (s - self.warmup) // self.step_size
            factor = (self.gamma ** k)
        for lr, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = lr * factor
        self.step_idx += 1


def build_scheduler(optimizer, cfg: SchedulerConfig):
    if cfg.kind == "cosine":
        return _WarmupCosine(optimizer, cfg.warmup_steps, cfg.total_steps)
    if cfg.kind == "step":
        return _WarmupStep(optimizer, cfg.warmup_steps, cfg.step_size, cfg.gamma)
    return None
