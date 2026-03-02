#!/usr/bin/env python3
"""
Script to toggle attn_implementation setting in all Unlearning dataset main_config.yaml files.

Usage:
    python toggle_attn_implementation.py --platform mac     # Add sdpa setting (for Mac)
    python toggle_attn_implementation.py --platform server # Remove sdpa setting (for CUDA-enabled server with flash-attention)
"""

import argparse
import re
from pathlib import Path


def find_config_files(base_dir: Path):
    """Find all main_config.yaml files in dataset directories."""
    config_files = []
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                config_file = dataset_dir / "main_config.yaml"
                if config_file.exists():
                    config_files.append(config_file)
    return config_files


def toggle_attn_implementation(config_file: Path, enable: bool):
    """
    Toggle the attn_implementation setting in a config file.

    Args:
        config_file: Path to the main_config.yaml file
        enable: If True, add the sdpa setting. If False, remove it completely.
    """
    with open(config_file, 'r') as f:
        content = f.read()

    # Pattern to match the model block (without comment)
    pattern = r'model:\s*\n\s*model_args:\s*\n\s*attn_implementation:\s*sdpa\s*\n'

    modified = False

    if enable:
        # Check if it already exists
        if re.search(pattern, content):
            # Already present, no change needed
            return False

        # Doesn't exist, add it at the end of the file
        # Ensure there's a newline before adding if the file doesn't end with one
        if content and not content.endswith('\n'):
            content += '\n'
        content += '\nmodel:\n  model_args:\n    attn_implementation: sdpa\n'
        modified = True
    else:
        # Remove the block if it exists
        if re.search(pattern, content):
            content = re.sub(pattern, '', content)
            modified = True

    if modified:
        with open(config_file, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Toggle attn_implementation setting in all Unlearning dataset config files"
    )
    parser.add_argument(
        "--platform",
        choices=["mac", "server"],
        required=True,
        help="Platform: 'mac' adds sdpa setting, 'server' removes it (for flash-attention)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base directory containing the datasets folder (default: parent of utils/)"
    )

    args = parser.parse_args()

    config_files = find_config_files(args.base_dir)

    if not config_files:
        print(f"No main_config.yaml files found in {args.base_dir / 'datasets'}")
        return

    enable_sdpa = (args.platform == "mac")
    action = "added" if enable_sdpa else "removed"

    print(f"Toggling attn_implementation for platform: {args.platform}")
    print(f"Setting will be {action} in {len(config_files)} config file(s)\n")

    modified_count = 0
    for config_file in config_files:
        dataset_name = config_file.parent.name
        if toggle_attn_implementation(config_file, enable_sdpa):
            print(f"✓ Modified: {config_file}")
            modified_count += 1
        else:
            print(f"⚠ No changes needed: {config_file}")

    print(f"\n{modified_count} file(s) modified.")


if __name__ == "__main__":
    main()
