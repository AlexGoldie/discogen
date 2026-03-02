"""Dataset loader for MaunaLoaN2O-2015 training data."""

from pathlib import Path

import numpy as np


def download_dataset(dest_loc: str) -> None:
    """Copy the dataset to the destination location.

    Args:
        dest_loc: Destination directory where the dataset should be saved.
    """
    # Dataset is already in the repository, just copy it
    train_source_file = Path(__file__).parent / "n2o_train.npy"
    train_dest_file = Path(dest_loc) / "train_data.npy"
    train_dest_file.parent.mkdir(parents=True, exist_ok=True)


    test_source_file = Path(__file__).parent / "n2o_test.npy"
    test_dest_file = Path(dest_loc) / "test_data.npy"
    test_dest_file.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(train_source_file, train_dest_file)
    shutil.copy2(test_source_file, test_dest_file)


def load_train_dataset(src_loc: str = "./data") -> np.ndarray:
    """Load the training dataset.

    Args:
        src_loc: Path to the data directory (default: ./data)
                 When called from task_src/MaunaLoaN2O-2015/, this resolves to task_src/MaunaLoaN2O-2015/data

    Returns:
        numpy array with shape (N, 5) containing N2O data from 2001-2014
    """
    data_file = Path(src_loc) / "n2o_train.npy"
    return np.load(data_file)

def load_test_dataset(src_loc: str = "./data") -> np.ndarray:
    """Load the test dataset.

    Args:
        src_loc: Path to the data directory (default: ./data)
                 When called from task_src/MaunaLoaN2O-2025/, this resolves to task_src/MaunaLoaN2O-2025/data

    Returns:
        numpy array with shape (N, 5) containing N2O data from 2015-2025
    """
    data_file = Path(src_loc) / "n2o_test.npy"
    return np.load(data_file)
