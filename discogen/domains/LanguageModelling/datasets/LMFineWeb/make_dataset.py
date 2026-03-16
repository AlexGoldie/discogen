import datasets
import numpy as np
import os
from typing import Iterable, List
from transformers import AutoTokenizer

import os
import sys
from huggingface_hub import hf_hub_download
from pathlib import Path


def download_dataset(dest_loc: str):
    # Download the GPT-2 tokens of Fineweb10B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    def get(fname):
        local_dir = os.path.join(dest_loc, 'fineweb10B')
        if not os.path.exists(os.path.join(local_dir, fname)):
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                            repo_type="dataset", local_dir=local_dir)
    get("fineweb_val_%06d.bin" % 0)
    num_chunks = 103 # full fineweb10B. Each chunk is 100M tokens
    for i in range(1, num_chunks+1):
        get("fineweb_train_%06d.bin" % i)


def load_dataset(src_loc: Path = Path(__file__).parent / "data"):
    """
    Load dataset and return filename patterns for DistributedDataLoader.

    Args:
        src_loc: Path to the data directory (default: task_src/LMFineWeb/data)
                 When called from task_src/LMFineWeb/, this resolves to task_src/LMFineWeb/data

    Returns:
        Dictionary with "train", "val", and "vocab_size" keys
    """
    # Construct paths relative to src_loc
    # Data is stored in fineweb10B subdirectory
    base_path = os.path.join(src_loc, 'fineweb10B')

    train_pattern = os.path.join(base_path, 'fineweb_train_*.bin')
    val_pattern = os.path.join(base_path, 'fineweb_val_*.bin')

    # GPT-2 has 50257 unique tokens; extended to nearest multiple of 128 for efficiency
    vocab_size = 50304

    return {
        "train": train_pattern,
        "val": val_pattern,
        "vocab_size": vocab_size
    }


if __name__ == "__main__":
    # Accept destination path from command line, or use default
    dest_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/LMFineWeb"
    download_dataset(dest_path)
