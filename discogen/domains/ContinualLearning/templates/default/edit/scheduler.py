from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    kind: str = "cosine"
    warmup_steps: int = 500
    total_steps: int = 10000
    step_size: int = 3000
    gamma: float = 0.1


def build_scheduler(optimizer, cfg: SchedulerConfig):
    """Return object with .step() called every step. No metric dependence."""
    ...
