from typing import List, Dict, Any
import numpy as np
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets
import torch


def _class_indices(ds: TensorDataset, class_ids: List[int]) -> List[int]:
    if isinstance(ds, TensorDataset):
        targets = ds.tensors[1].tolist()
    else:
        targets = ds.targets if hasattr(ds, "targets") else ds.labels
    allowed = set(class_ids)
    return [i for i, y in enumerate(targets) if y in allowed]


def _partition_classes(n_classes: int, n_tasks: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    order = np.arange(n_classes)
    rng.shuffle(order)
    return np.array_split(order.tolist(), n_tasks)


def build_task_sequence(
    data_root: str,
    seed: int,
    n_tasks: int,
    split_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    order_seed = int(split_cfg.get("class_order_seed", 0))
    class_splits = _partition_classes(100, n_tasks, order_seed)

    try:
        base_train = datasets.CIFAR100(data_root, train=True, download=True)
        base_test = datasets.CIFAR100(data_root, train=False, download=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to download/load CIFAR-100. On macOS, run the Python 'Install Certificates.command' for your Python, or set SSL_CERT_FILE to certifi.where()."
        ) from e

    tasks: List[Dict[str, Any]] = []
    for t, cls_ids in enumerate(class_splits):
        cls_ids = list(map(int, cls_ids))
        train_idx = _class_indices(base_train, cls_ids)
        test_idx = _class_indices(base_test, cls_ids)
        train_ds = Subset(base_train, train_idx)
        # Use a small validation subset from train deterministically
        val_size = min(len(train_idx) // 10, 500)
        rng = np.random.RandomState(seed + 101 * (t + 1))
        rng.shuffle(train_idx)
        val_idx = train_idx[:val_size]
        tr_idx = train_idx[val_size:]
        train_ds = Subset(base_train, tr_idx)
        val_ds = Subset(base_train, val_idx)
        test_ds = Subset(base_test, test_idx)

        tasks.append(
            {
                "task_id": t,
                "train": train_ds,
                "val": val_ds,
                "test": test_ds,
                "n_classes_this_task": len(cls_ids),
                "class_ids": cls_ids,
            }
        )
    return tasks


def download_dataset(cache_dir) -> None:
    os = __import__("os")
    os.makedirs(cache_dir, exist_ok=True)
    datasets.CIFAR100(cache_dir, train=True, download=True)
    datasets.CIFAR100(cache_dir, train=False, download=True)


def default_image_size():
    return 32


def get_split_cfg():
    return {"class_order_seed": 23}
