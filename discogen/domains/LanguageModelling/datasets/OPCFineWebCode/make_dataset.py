import datasets
import numpy as np
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path


def download_dataset(dest_loc: str):
    """
    Download and preprocess the OPC-FineWeb-Code dataset into binary files.

    This function:
    1. Downloads OPC-FineWeb-Code from HuggingFace
    2. Tokenizes the text using GPT-2 tokenizer
    3. Randomly samples a fixed validation set (half a parquet file) from all data
    4. Uses remaining data for training (validation examples excluded)
    5. Saves tokenized data as .bin files with proper headers
    6. Skips any unsafe/corrupted files during processing

    Args:
        dest_loc: Destination directory for the dataset
    """
    # Create output directory
    output_dir = os.path.join(dest_loc, 'opcfinewebcode')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading OPC-FineWeb-Code dataset from HuggingFace...")

    try:
        from huggingface_hub import HfFileSystem

        # Get list of parquet files without downloading
        print("Fetching dataset file list...")
        fs = HfFileSystem()
        all_files = fs.ls("datasets/OpenCoder-LLM/opc-fineweb-code-corpus/data", detail=False)
        parquet_files = sorted([f.split('/')[-1] for f in all_files if f.endswith('.parquet')])

        total_files = len(parquet_files)

        # First, load one file to determine validation set size (half a parquet file)
        print(f"\nLoading first file to determine validation set size...")
        first_file_data = {"train": [f"data/{parquet_files[0]}"]}
        ds_first = datasets.load_dataset(
            "OpenCoder-LLM/opc-fineweb-code-corpus",
            data_files=first_file_data,
            verification_mode="no_checks"
        )
        examples_per_file = len(ds_first["train"])
        fixed_val_size = examples_per_file // 2  # Half of one parquet file
        print(f"Examples per parquet file: {examples_per_file:,}")
        print(f"Fixed validation size: {fixed_val_size:,} (half a parquet file)")

        # Now load all files for training
        num_files_to_download = 1  # Change this to download more files
        selected_files = parquet_files[:num_files_to_download]

        print(f"\nTotal parquet files available: {total_files}")
        print(f"Downloading {num_files_to_download} file(s) ({100*num_files_to_download/total_files:.2f}%)")

        # Load all selected files
        print(f"Loading data from {num_files_to_download} file(s)...")
        data_files = {"train": [f"data/{f}" for f in selected_files]}
        ds_full = datasets.load_dataset(
            "OpenCoder-LLM/opc-fineweb-code-corpus",
            data_files=data_files,
            verification_mode="no_checks"
        )
        full_dataset = ds_full["train"]
        total_size = len(full_dataset)

        # Randomly sample fixed_val_size examples for validation
        print(f"\nTotal examples loaded: {total_size:,}")
        print(f"Randomly sampling {fixed_val_size:,} examples for validation...")

        # Create shuffled indices
        import random
        random.seed(42)  # For reproducibility
        all_indices = list(range(total_size))
        random.shuffle(all_indices)

        # Split indices
        val_indices = sorted(all_indices[:fixed_val_size])
        train_indices = sorted(all_indices[fixed_val_size:])

        train_size = len(train_indices)
        val_percentage = (fixed_val_size / total_size) * 100

        print(f"Train size: {train_size:,} examples ({100-val_percentage:.2f}%)")
        print(f"Validation size: {fixed_val_size:,} examples ({val_percentage:.2f}%)")

        # Create train/val splits using the shuffled indices
        train_dataset = full_dataset.select(train_indices)
        val_dataset = full_dataset.select(val_indices)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative loading method...")
        raise

    # Initialize tokenizer - using GPT-2 tokenizer for consistency
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Process each split
    splits = [
        ("train", train_dataset, "opcfinewebcode_train_000001.bin"),
        ("validation", val_dataset, "opcfinewebcode_val_000000.bin")
    ]

    for split_name, split_data, output_filename in splits:
        print(f"\nProcessing {split_name} split with {len(split_data)} examples...")

        # Tokenize all texts
        all_tokens = []
        skipped_count = 0

        for idx, example in enumerate(tqdm(split_data, desc=f"Tokenizing {split_name}")):
            try:
                text = example["text"]
                # Skip if text is None or empty
                if text is None or len(text.strip()) == 0:
                    skipped_count += 1
                    continue

                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            except Exception as e:
                # Skip unsafe/corrupted files
                skipped_count += 1
                if skipped_count % 1000 == 0:
                    print(f"Skipped {skipped_count} corrupted/unsafe files so far...")
                continue

        if skipped_count > 0:
            print(f"Skipped {skipped_count} corrupted/unsafe files in {split_name}")

        # Convert to numpy array
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        print(f"Total tokens in {split_name}: {len(all_tokens):,}")

        # Determine output path
        output_path = os.path.join(output_dir, output_filename)

        # Write to binary file with header
        print(f"Writing to {output_path}...")
        with open(output_path, "wb") as f:
            # Write header (256 int32 values)
            header = np.zeros(256, dtype=np.int32)
            header[0] = 20240520  # magic number
            header[1] = 1         # version
            header[2] = len(all_tokens)  # number of tokens
            f.write(header.tobytes())

            # Write tokens
            f.write(all_tokens.tobytes())

        print(f"Successfully wrote {len(all_tokens):,} tokens to {output_path}")


def load_dataset(src_loc: Path = Path(__file__).parent / "data"):
    """
    Load dataset and return filename patterns for DistributedDataLoader.

    Args:
        src_loc: Path to the data directory (default: task_src/OPCFineWebCode/data)
                 When called from task_src/OPCFineWebCode/, this resolves to task_src/OPCFineWebCode/data

    Returns:
        Dictionary with "train", "val", and "vocab_size" keys
    """
    # Construct paths relative to src_loc
    # Data is stored in opcfinewebcode subdirectory
    base_path = os.path.join(src_loc, 'opcfinewebcode')

    train_pattern = os.path.join(base_path, 'opcfinewebcode_train_*.bin')
    val_pattern = os.path.join(base_path, 'opcfinewebcode_val_*.bin')

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
    dest_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/OPCFineWebCode"
    download_dataset(dest_path)
