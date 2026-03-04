"""Helpers for seeding, logging, and checkpoint IO."""

import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    # DataLoader worker seeding
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        self.f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def save_json(path: str, obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
