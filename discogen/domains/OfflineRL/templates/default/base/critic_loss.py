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
    """Compute the critic loss.

    The critic loss is TD error with a behavior cloning penalty on the next actions.
    This encourages the critic to be conservative on out-of-distribution actions.

    Args:
        critic_apply_fn: Function to apply critic network.
        target_critic_apply_fn: Function to apply target critic network.
        target_actor_apply_fn: Function to apply target actor network.
        config: Configuration dictionary with hyperparameters.
        critic_params: Current critic parameters (for gradient computation).
        target_critic_params: Target critic parameters.
        target_actor_params: Target actor parameters.
        batch: Batch of transitions with keys: observations, actions, rewards,
               next_observations, next_actions, masks.
        rng: JAX random key.

    Returns:
        Tuple of (critic_loss, info_dict).
    """
    rng, sample_rng = jax.random.split(rng)

    # Get next actions from target actor
    next_dist = target_actor_apply_fn({'params': target_actor_params}, batch['next_observations'])
    next_actions = next_dist.mode()

    # Add clipped noise for target policy smoothing (TD3-style)
    noise = jnp.clip(
        jax.random.normal(sample_rng, next_actions.shape) * config['actor_noise'],
        -config['actor_noise_clip'],
        config['actor_noise_clip'],
    )
    next_actions = jnp.clip(next_actions + noise, -1, 1)

    # Compute target Q-values (minimum over ensemble)
    next_qs = target_critic_apply_fn(
        {'params': target_critic_params},
        batch['next_observations'],
        actions=next_actions
    )
    next_q = next_qs.min(axis=0)

    # Apply BC penalty on critic targets
    mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
    next_q = next_q - config['alpha_critic'] * mse

    # Compute TD target
    target_q = batch['rewards'] + config['discount'] * batch['masks'] * next_q

    # Compute current Q-values
    q = critic_apply_fn(
        {'params': critic_params},
        batch['observations'],
        actions=batch['actions']
    )

    # MSE loss over ensemble
    critic_loss = jnp.square(q - target_q).mean()

    info = {
        'critic_loss': critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    }

    return critic_loss, info
