from dataclasses import dataclass


@dataclass
class SamplerConfig:
    replay_ratio: float = 0.5
    min_replay_after: int = 1000


class BatchMixer:
    def __init__(self, cfg: SamplerConfig, seed: int):
        """Store config and RNG."""
        ...

    def mix(self, current, replay, final_batch_size: int):
        """Combine 'current' with samples from 'replay' to reach 'final_batch_size'."""
        ...
