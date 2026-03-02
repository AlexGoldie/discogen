import optax
from typing import Any

def build_surrogate_optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    """
    Builds the optimizer for the surrogate model.
    Args:
        config: Configuration dictionary. [Must contain 'surrogate_optimizer_lr']
    Returns:
        Optimizer for the surrogate model.
    """
    return optax.sgd(learning_rate=config['surrogate_optimizer_lr'])
