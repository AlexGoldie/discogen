from typing import List, Dict, Any, Tuple
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, extract_archive


class TinyImageNetTrain(Dataset):
    def __init__(self, root: str, wnids: List[str], allowed_wnids: List[str]):
        self.root = root
        self.wnids = wnids
        self.allowed = set(allowed_wnids)
        self.transform = None
        self.samples: List[Tuple[str, int]] = []
        train_dir = os.path.join(root, "train")
        for wnid in wnids:
            if wnid not in self.allowed:
                continue
            img_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                path = os.path.join(img_dir, fname)
                label = wnids.index(wnid)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, root: str, wnids: List[str], allowed_wnids: List[str]):
        self.root = root
        self.wnids = wnids
        self.allowed = set(allowed_wnids)
        self.transform = None
        self.samples: List[Tuple[str, int]] = []
        val_dir = os.path.join(root, "val")
        ann_path = os.path.join(val_dir, "val_annotations.txt")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(
                f"TinyImageNet val annotations not found at {ann_path}. Please download tiny-imagenet-200."
            )
        with open(ann_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                fname, wnid = row[0], row[1]
                if wnid not in self.allowed:
                    continue
                path = os.path.join(val_dir, "images", fname)
                label = wnids.index(wnid)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _load_wnids(root: str) -> List[str]:
    wnids_path = os.path.join(root, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(
            f"TinyImageNet not found under {root}. Expected wnids.txt. Download tiny-imagenet-200."
        )
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return wnids


def _partition_classes(n: int, n_tasks: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    order = np.arange(n)
    rng.shuffle(order)
    return np.array_split(order.tolist(), n_tasks)


def build_task_sequence(
    data_root: str,
    seed: int,
    n_tasks: int,
    split_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    root = os.path.join(data_root, "tiny-imagenet-200")
    wnids = _load_wnids(root)
    class_splits = _partition_classes(len(wnids), n_tasks, int(split_cfg.get("class_order_seed", 0)))

    tasks: List[Dict[str, Any]] = []
    for t, cls_ids in enumerate(class_splits):
        cls_ids = list(map(int, cls_ids))
        allowed = [wnids[i] for i in cls_ids]
        train_ds = TinyImageNetTrain(root, wnids, allowed)
        val_ds = TinyImageNetVal(root, wnids, allowed)
        test_ds = TinyImageNetVal(root, wnids, allowed)
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
    os.makedirs(cache_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    download_url(url, cache_dir, filename=filename)
    zip_path = os.path.join(cache_dir, filename)
    extract_archive(zip_path, cache_dir)

def default_image_size():
    return 64

def get_split_cfg():
    return {"class_order_seed": 23}
