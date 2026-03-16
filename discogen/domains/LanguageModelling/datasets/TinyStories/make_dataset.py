import datasets
import numpy as np
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path


def download_dataset(dest_loc: str):
    """
    Download and preprocess the TinyStories dataset into binary files.

    This function:
    1. Downloads TinyStories from HuggingFace
    2. Tokenizes the text using a GPT-2 compatible tokenizer
    3. Saves tokenized data as .bin files with proper headers
    """
    # Create output directory
    output_dir = os.path.join(dest_loc, 'tinystories')
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset from HuggingFace
    print("Loading TinyStories dataset from HuggingFace...")
    ds = datasets.load_dataset("roneneldan/TinyStories")

    # Initialize tokenizer - using TinyStories-specific tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")

    # Process each split
    for split_name in ["train", "validation"]:
        if split_name not in ds:
            continue

        split = ds[split_name]
        print(f"\nProcessing {split_name} split with {len(split)} examples...")

        # Tokenize all texts
        all_tokens = []
        for example in tqdm(split, desc=f"Tokenizing {split_name}"):
            text = example["text"]
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # Convert to numpy array
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        print(f"Total tokens in {split_name}: {len(all_tokens):,}")

        # Determine output filename
        if split_name == "train":
            output_filename = os.path.join(output_dir, "tinystories_train_000001.bin")
        else:
            output_filename = os.path.join(output_dir, "tinystories_val_000000.bin")

        # Write to binary file with header
        print(f"Writing to {output_filename}...")
        with open(output_filename, "wb") as f:
            # Write header (256 int32 values)
            header = np.zeros(256, dtype=np.int32)
            header[0] = 20240520  # magic number
            header[1] = 1         # version
            header[2] = len(all_tokens)  # number of tokens
            f.write(header.tobytes())

            # Write tokens
            f.write(all_tokens.tobytes())

        print(f"Successfully wrote {len(all_tokens):,} tokens to {output_filename}")


def load_dataset(src_loc: Path = Path(__file__).parent / "data"):
    """
    Load dataset and return filename patterns for DistributedDataLoader.

    Args:
        src_loc: Path to the data directory (default: task_src/TinyStories/data)
                 When called from task_src/TinyStories/, this resolves to task_src/TinyStories/data

    Returns:
        Dictionary with "train", "val", and "vocab_size" keys
    """
    # Construct paths relative to src_loc
    # Data is stored in tinystories subdirectory
    base_path = os.path.join(src_loc, 'tinystories')

    train_pattern = os.path.join(base_path, 'tinystories_train_*.bin')
    val_pattern = os.path.join(base_path, 'tinystories_val_*.bin')

    # GPT-2 has 50257 unique tokens; extended to nearest multiple of 128 for efficiency
    vocab_size = 50304

    return {
        "train": train_pattern,
        "val": val_pattern,
        "vocab_size": vocab_size
    }

if __name__ == "__main__":
    import sys
    # Accept destination path from command line, or use default
    dest_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/TinyStories"
    download_dataset(dest_path)
