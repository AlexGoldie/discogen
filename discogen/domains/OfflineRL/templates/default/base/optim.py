import optax


def create_optimizer(learning_rate: float, eps: float = 1e-8):
    """Create an Adam optimizer.

    Args:
        learning_rate: Learning rate for the optimizer.
        eps: Epsilon value for numerical stability.

    Returns:
        An optax optimizer.
    """
    return optax.adam(learning_rate=learning_rate, eps=eps)


def create_optimizer_with_schedule(
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    eps: float = 1e-8,
):
    """Create an Adam optimizer with optional learning rate schedule.

    Args:
        learning_rate: Peak learning rate.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (linear ramp).
        eps: Epsilon value for numerical stability.

    Returns:
        An optax optimizer with schedule.
    """
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=learning_rate * 0.1,
        )
    else:
        schedule = learning_rate

    return optax.adam(learning_rate=schedule, eps=eps)
