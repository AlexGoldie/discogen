import json
import os
import sys
import torch
import numpy as np
import random

from config import get_config
from train import train_model
from evaluate import evaluate_model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Run training and evaluation."""
    # Load configuration
    config = get_config()
    config_dict = config.to_dict()

    # Set random seed
    set_seed(config.seed)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Import dataset loader
    from make_dataset import load_dataset

    # Load datasets
    print("Loading datasets...")
    dataset = load_dataset()
    train_data, test_data = dataset["train"], dataset["test"]

    # Train model
    print("Starting training...")
    model, history = train_model(
        train_data=train_data,
        val_data=test_data,
        config=config_dict,
        device=device
    )

    # Log training progress (scores kept in fixed file, not in module)
    num_epochs = len(history.get('train_loss', []))
    for epoch in range(num_epochs):
        train_loss = history['train_loss'][epoch]
        val_loss = history['val_loss'][epoch]
        lr = history['lr'][epoch]
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"LR = {lr:.6f}")

    # Evaluate model
    print("Starting evaluation...")
    metrics = evaluate_model(
        model=model,
        val_data=test_data,
        config=config_dict,
        device=device
    )

    # Print final metrics
    print(json.dumps({k: float(v) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
