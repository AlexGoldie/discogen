import jax
import jax.numpy as jnp


def compute_critic_loss(
    critic_apply_fn,
    target_critic_apply_fn,
    target_actor_apply_fn,
    config,
    critic_params,
    target_critic_params,
    target_actor_params,
    batch,
    rng,
):
    # Inputs:
    # - critic_apply_fn: Function to apply critic network.
    #     Usage: critic_apply_fn({'params': params}, obs, actions=actions)
    # - target_critic_apply_fn: Function to apply target critic network.
    # - target_actor_apply_fn: Function to apply target actor network.
    #     Usage: target_actor_apply_fn({'params': params}, obs).mode()
    # - config: Configuration dictionary with hyperparameters:
    #     'discount', 'actor_noise', 'actor_noise_clip', 'alpha_critic'
    # - critic_params: Current critic parameters (for gradient computation).
    # - target_critic_params: Target critic parameters (no gradient).
    # - target_actor_params: Target actor parameters (no gradient).
    # - batch: Dictionary with keys:
    #     'observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'masks'
    # - rng: JAX random key.

    """Fill in your critic loss logic here."""

    critic_loss = jnp.array(0.0)
    info = {
        'critic_loss': critic_loss,
        'q_mean': jnp.array(0.0),
    }

    # Your function must return a tuple of (critic_loss, info_dict).
    return critic_loss, info
