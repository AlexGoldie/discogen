import optax


def create_optimizer(learning_rate: float, eps: float = 1e-8):
    # Inputs:
    # - learning_rate: Learning rate for the optimizer.
    # - eps: Epsilon value for numerical stability.

    """Fill in your optimizer creation logic here."""

    # Your function must return an optax optimizer that can be used with .init() and .update().
    optax_object = ...
    return optax_object


def create_optimizer_with_schedule(
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    eps: float = 1e-8,
):
    # Inputs:
    # - learning_rate: Peak learning rate.
    # - total_steps: Total number of training steps.
    # - warmup_steps: Number of warmup steps.
    # - eps: Epsilon value for numerical stability.

    """Fill in your optimizer with schedule logic here."""

    # Your function must return an optax optimizer with schedule.
    optax_object_with_scheduler = ...
    return optax_object_with_scheduler
