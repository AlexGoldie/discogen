import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from networks import TrajectoryPredictionModel
from loss import compute_loss
from optim import create_optimizer


def train_model(
    train_data,
    val_data,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[TrajectoryPredictionModel, Dict[str, List[float]]]:
    """Train the model and return it along with training history.

    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Tuple of (trained model, training history dict with keys
        'train_loss', 'val_loss', 'lr' each mapping to a list of per-epoch values).
    """
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = TrajectoryPredictionModel(config)
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)

    # Training loop
    num_epochs = config['num_epochs']
    best_val_loss = float('inf')
    best_model_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move batch to device and wrap in input_dict
            batch = move_batch_to_device(batch, device)
            batch = {'input_dict': batch}

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch)

            # Compute loss
            loss = compute_loss(
                predictions=predictions,
                batch=batch,
                config=config
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['max_grad_norm']
                )

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                batch = {'input_dict': batch}
                predictions = model(batch)
                loss = compute_loss(
                    predictions=predictions,
                    batch=batch,
                    config=config
                )
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in batch to the specified device."""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = move_batch_to_device(value, device)
        else:
            moved_batch[key] = value
    return moved_batch
