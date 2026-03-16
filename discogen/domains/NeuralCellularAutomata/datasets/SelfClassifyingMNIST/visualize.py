"""Visualization for SelfClassifyingMNIST dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize(state_final, targets, config, output_dir: str = ".") -> dict | None:
    """Visualize classification results and compute accuracy.

    Args:
        state_final: Final NCA states of shape (num_samples, H, W, channels).
        targets: Target one-hot labels of shape (num_samples, H, W, num_classes).
        config: Dataset configuration dictionary.
        output_dir: Directory to save output images.

    Returns:
        Dictionary with accuracy metrics.
    """
    num_classes = config["nca"]["num_classes"]

    # Extract first sample
    logits = np.array(state_final[0, ..., :num_classes])
    target = np.array(targets[0])
    mnist_input = np.array(state_final[0, ..., -1])

    # Get predictions and true labels
    pred_class = np.argmax(logits, axis=-1)
    true_class = np.argmax(target, axis=-1)

    # Mask for digit pixels (where target is non-zero)
    digit_mask = np.any(target > 0, axis=-1)

    # Compute accuracy on digit pixels only
    correct = (pred_class == true_class) & digit_mask
    accuracy = correct.sum() / digit_mask.sum() if digit_mask.sum() > 0 else 0.0

    # Compute accuracy across all samples
    all_logits = np.array(state_final[..., :num_classes])
    all_targets = np.array(targets)
    all_preds = np.argmax(all_logits, axis=-1)
    all_true = np.argmax(all_targets, axis=-1)
    all_mask = np.any(all_targets > 0, axis=-1)
    all_correct = (all_preds == all_true) & all_mask
    total_accuracy = all_correct.sum() / all_mask.sum() if all_mask.sum() > 0 else 0.0

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # MNIST input
    axes[0].imshow(mnist_input, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    # True class (show as colored image)
    axes[1].imshow(true_class, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("True Class")
    axes[1].axis("off")

    # Predicted class
    axes[2].imshow(pred_class, cmap="tab10", vmin=0, vmax=9)
    axes[2].set_title(f"Predicted (acc={accuracy:.1%})")
    axes[2].axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "selfclassifying_mnist_nca_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image to: {comparison_path}")

    return {"accuracy": float(total_accuracy)}
