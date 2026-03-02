"""Data utilities: build task sequences and make data loaders."""

from typing import Any, Dict, List

import importlib
import importlib.util
import os
import torch
from torch.utils.data import DataLoader, Dataset
from wrappers import seed_worker

import data_transforms as dt
import make_dataset

class TransformedDataset(Dataset):
    """
    A wrapper dataset that applies a transform to the output of another dataset.
    """
    def __init__(self, dataset: Dataset, transform: Any):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Any:
        # Get the original (image, label) tuple
        # This will be (PIL Image, int) from your Subset
        img, label = self.dataset[idx]

        # Apply the transform to the image
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self) -> int:
        return len(self.dataset)


def default_image_size(dataset_name: str, model_class_name: str) -> int:
    """Return a sensible default image size for a dataset/model combo."""
    if model_class_name.lower().startswith("clvit"):
        return 224
    return {"PermutedMNIST": 28, "SplitCIFAR100": 32, "TinyImageNetSplit": 64}[dataset_name]


def get_task_sequence(
    name: str,
    data_root: str,
    seed: int,
    n_tasks: int,
    split_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:

    return make_dataset.build_task_sequence(data_root, seed, n_tasks, split_cfg)



def apply_transforms(tasks: List[Dict[str, Any]], image_size: int) -> None:
    for t in tasks:
        t["train"] = TransformedDataset(
            t["train"],
            transform=dt.build_transforms(image_size, train=True)
        )
        t["val"] = TransformedDataset(
            t["val"],
            transform=dt.build_transforms(image_size, train=False)
        )
        t["test"] = TransformedDataset(
            t["test"],
            transform=dt.build_transforms(image_size, train=False)
        )

def make_loaders(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
