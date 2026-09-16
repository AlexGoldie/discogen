import optax


def make_optimizer(config):
    def scale_by_optimizer(eps: float = 1e-5):
        """Factory for Adam-style scaling with custom eps and gradient clipping."""

        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.scale_by_adam(eps=eps)
        )

    return scale_by_optimizer
