import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Dict, Any, Tuple, Optional


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Tuple[optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    # Get hyperparameters from config
    learning_rate = config.get('learning_rate', 0.00075)
    lr_milestones = [10, 20, 30, 40, 50]

    # Create Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        eps=0.0001
    )

    # Create MultiStepLR scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=0.5
    )

    return optimizer, scheduler


def create_optimizer_with_separate_groups(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Tuple[optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Create optimizer with weight-decay parameter groups and scheduler."""
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    # Create optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=learning_rate,
        eps=1e-4
    )

    # Create scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[10, 20, 30],
        gamma=0.5
    )

    return optimizer, scheduler
