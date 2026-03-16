import jax
import jax.numpy as jnp


def compute_actor_loss(
    actor_apply_fn,
    critic_apply_fn,
    config,
    actor_params,
    critic_params,
    batch,
    rng,
):
    # Inputs:
    # - actor_apply_fn: Function to apply actor network.
    #     Usage: actor_apply_fn({'params': params}, obs)
    # - critic_apply_fn: Function to apply critic network.
    # - config: Configuration dictionary
    # - actor_params: Current actor parameters (for gradient computation).
    # - critic_params: Critic parameters (no gradient).
    # - batch: Dictionary with keys:
    #     'observations', 'actions'
    # - rng: JAX random key.

    """Fill in your actor loss logic here."""

    total_loss = ...
    info = ...

    # Your function must return a tuple of (actor_loss, info_dict).
    return total_loss, info
