from typing import List, Dict, Any
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class _PermutedMNIST(Dataset):
    def __init__(self, base: Dataset, permutation: np.ndarray):
        self.base = base
        self.permutation = permutation
        self.transform = getattr(base, "transform", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # Ensure tensor
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        c, h, w = x.shape
        flat = x.view(-1)
        flat = flat[self.permutation]
        x = flat.view(c, h, w)
        # Allow downstream transforms (resize/normalize) to be applied later by data.py
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def _subset(ds: Dataset, size: int, seed: int) -> Dataset:
    if size is None or size <= 0 or size >= len(ds):
        return ds
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=size, replace=False)
    return Subset(ds, idx.tolist())


def _train_val_split(ds: Dataset, val_size: int, seed: int):
    if not val_size or val_size <= 0:
        return ds, _subset(ds, 0, seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


def build_task_sequence(
    data_root: str,
    seed: int,
    n_tasks: int,
    split_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    perm_seed_base = int(split_cfg.get("perm_seed_base", 0))
    train_size = split_cfg.get("train_size")
    val_size = split_cfg.get("val_size")
    test_size = split_cfg.get("test_size")

    tasks: List[Dict[str, Any]] = []
    for t in range(n_tasks):
        rng = np.random.RandomState(perm_seed_base + t)
        perm = rng.permutation(28 * 28)
        try:
            base_train = datasets.MNIST(data_root, train=True, download=True)
            base_test = datasets.MNIST(data_root, train=False, download=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to download/load MNIST. On macOS, run the Python 'Install Certificates.command' for your Python, or set SSL_CERT_FILE to certifi.where()."
            ) from e

        train_full = _PermutedMNIST(base_train, perm)
        test_full = _PermutedMNIST(base_test, perm)

        # Split val from train deterministically, then optional subsample train
        split_seed = seed + 11 * (t + 1)
        train_split, val_ds = _train_val_split(train_full, int(val_size or 0), split_seed)
        train_seed = seed + 7 * (t + 1)
        test_seed = seed + 17 * (t + 1)
        train_ds = _subset(train_split, train_size, train_seed)
        test_ds = _subset(test_full, test_size, test_seed)

        tasks.append(
            {
                "task_id": t,
                "train": train_ds,
                "val": val_ds,
                "test": test_ds,
                "n_classes_this_task": 10,
                "class_ids": list(range(10)),
            }
        )
    return tasks


def download_dataset(cache_dir) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    # Use torchvision's downloader into the provided cache dir
    datasets.MNIST(cache_dir, train=True, download=True)
    datasets.MNIST(cache_dir, train=False, download=True)


def default_image_size():
    return 28


def get_split_cfg():
    return {"perm_seed_base": 17}
