"""Visualization for MatrixOperations dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize(state_final, targets, config, output_dir: str = ".") -> dict | None:
    """Visualize matrix operation result as heatmap.

    Args:
        state_final: Final NCA states of shape (num_samples, H, W, channels).
        targets: Target matrices of shape (num_samples, H, W, 1).
        config: Dataset configuration dictionary.
        output_dir: Directory to save output images.

    Returns:
        None (no additional metrics for this visualization type).
    """
    matrix_config = config["matrix"]
    output_ch = matrix_config["output_channel"]
    input_channels = matrix_config["input_channels"]
    operations = matrix_config["operations"]
    num_ops = len(operations)

    num_viz = min(4, state_final.shape[0])

    fig, axes = plt.subplots(num_viz, 4, figsize=(12, 3 * num_viz))

    for row, i in enumerate(range(num_viz)):
        sample = state_final[i]
        A = np.array(sample[..., input_channels[0]])
        B = np.array(sample[..., input_channels[1]])
        pred = np.array(sample[..., output_ch])
        target_np = np.array(targets[i, ..., 0])

        # Recover operation from one-hot in last num_ops channels
        op_onehot = np.array(sample[0, 0, -num_ops:])
        op_name = operations[int(np.argmax(op_onehot))]

        # Shared colour scale across all four panels in this row
        all_vals = np.concatenate([A.ravel(), B.ravel(), target_np.ravel(), pred.ravel()])
        vmin, vmax = all_vals.min(), all_vals.max()

        b_label = "Input B (unused)" if op_name in ("transpose", "negate") else "Input B"
        col_titles = ["Input A", b_label, "Target", "NCA Output"]
        for j, (ax, data, title) in enumerate(zip(axes[row], [A, B, target_np, pred], col_titles)):
            ax.imshow(data, cmap="RdBu", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if j == 0 and row == 0:
                ax.set_title(f"{title}\n[{op_name}]")
            elif row == 0:
                ax.set_title(title)
            elif j == 0:
                ax.set_title(f"[{op_name}]")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "matrix_operations_nca_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image to: {comparison_path}")

    return None
