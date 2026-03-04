import optax
from typing import Any

def build_surrogate_optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    """
    Builds the optimizer for the surrogate model.
    Args:
        config: Configuration dictionary.
    Returns:
        Optimizer for the surrogate model.
    """
    # --- Fill in your construction of the gradient optimizer here. ---
    return optimizer # noqa
