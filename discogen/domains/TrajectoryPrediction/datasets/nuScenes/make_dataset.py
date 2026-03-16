"""nuScenes dataset loader for trajectory prediction."""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset


DATASET_NAME = "nuScenes"


def download_dataset(dest_loc: str):
    """Download nuScenes trajectory data from HuggingFace."""
    from huggingface_hub import snapshot_download

    dest_path = Path(dest_loc)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_NAME} dataset from HuggingFace...")
    snapshot_download(
        repo_id="saeedrmd/trajectory-prediction-nuscenes",
        repo_type="dataset",
        local_dir=str(dest_path),
    )

    pkl_count = len(list(dest_path.glob("*.pkl")))
    print(f"Downloaded {pkl_count} samples to {dest_path}")


class TrajectoryDataset(Dataset):
    """Trajectory dataset that loads pkl files."""

    def __init__(self, data_files: list):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], 'rb') as f:
            return pickle.load(f)


def _get_remote_count(repo_id: str) -> int:
    """Return the number of .pkl files in the HuggingFace dataset repo."""
    try:
        from huggingface_hub import list_repo_files
        return sum(1 for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".pkl"))
    except Exception:
        return 0


def load_dataset(src_loc: str = "./nuScenes/data") -> Dict[str, Dataset]:
    """Load nuScenes dataset with 80/20 train/test split."""
    data_path = Path(src_loc)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    # Get all pkl files
    pkl_files = sorted(data_path.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {data_path}")

    expected = _get_remote_count("saeedrmd/trajectory-prediction-nuscenes")
    if expected > 0 and len(pkl_files) < expected:
        print(
            f"WARNING: Only {len(pkl_files)} of {expected} samples found locally. "
            "The dataset may be incomplete due to an interrupted download. "
            "To fix: delete the cache/nuScenes directory and re-run create-task."
        )

    # Split 80/20 for train/test
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(pkl_files))
    split_idx = int(len(pkl_files) * 0.8)

    train_files = [pkl_files[i] for i in indices[:split_idx]]
    test_files = [pkl_files[i] for i in indices[split_idx:]]

    print(f"Loaded {DATASET_NAME}: {len(train_files)} train, {len(test_files)} test samples")

    return {
        "train": TrajectoryDataset(train_files),
        "test": TrajectoryDataset(test_files),
    }
