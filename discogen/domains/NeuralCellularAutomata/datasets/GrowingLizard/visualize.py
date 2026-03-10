"""Visualization for GrowingLizard dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize(state_final, targets, config, output_dir: str = ".") -> dict | None:
    """Visualize NCA output vs target and save comparison image.

    Args:
        state_final: Final NCA states of shape (num_samples, H, W, channels).
        targets: Target images of shape (num_samples, H, W, 4) in RGBA format.
        config: Dataset configuration dictionary.
        output_dir: Directory to save output images.

    Returns:
        None (no additional metrics for this visualization type).
    """
    # Extract RGBA channels from first sample
    nca_rgba = np.array(state_final[0, ..., -4:])
    nca_rgba = np.clip(nca_rgba, 0, 1)
    target_np = np.array(targets[0])

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.imshow(target_np)
    ax1.set_title("Target")
    ax1.axis("off")
    ax2.imshow(nca_rgba)
    ax2.set_title("NCA Output")
    ax2.axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "growing_lizard_nca_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image to: {comparison_path}")

    return None
