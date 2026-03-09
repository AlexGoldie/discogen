"""Visualization for MNISTInpainting dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize(state_final, targets, config, output_dir: str = ".") -> dict | None:
    """Visualize inpainting results.

    Args:
        state_final: Final NCA states of shape (num_samples, H, W, channels).
        targets: Target images of shape (num_samples, H, W, 1).
        config: Dataset configuration dictionary.
        output_dir: Directory to save output images.

    Returns:
        None (no additional metrics for this visualization type).
    """
    # Extract first sample - reconstruction is in last channel
    reconstruction = np.array(state_final[0, ..., -1])
    reconstruction = np.clip(reconstruction, 0, 1)
    target = np.array(targets[0, ..., 0])

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.imshow(target, cmap="gray")
    ax1.set_title("Target (Original)")
    ax1.axis("off")

    ax2.imshow(reconstruction, cmap="gray")
    ax2.set_title("NCA Reconstruction")
    ax2.axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "nca_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image to: {comparison_path}")

    return None
