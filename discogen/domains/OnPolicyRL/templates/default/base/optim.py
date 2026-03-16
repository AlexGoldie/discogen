import optax


def scale_by_optimizer(eps: float = 1e-5):
    """Factory for Adam-style scaling with custom eps."""
    return optax.scale_by_adam(eps=eps)
