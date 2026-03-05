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
    """Compute the actor loss.

    The actor loss combines Q-value maximization with a behavior cloning penalty.
    The Q-loss is normalized by its absolute mean to make the BC coefficient
    scale-invariant.

    Args:
        actor_apply_fn: Function to apply actor network.
        critic_apply_fn: Function to apply critic network.
        config: Configuration dictionary with hyperparameters.
        actor_params: Current actor parameters (for gradient computation).
        critic_params: Critic parameters (no gradient).
        batch: Batch of transitions.
        rng: JAX random key.

    Returns:
        Tuple of (total_loss, info_dict).
    """
    # Get actions from current policy
    dist = actor_apply_fn({'params': actor_params}, batch['observations'])
    actions = dist.mode()

    # Compute Q-values for policy actions
    qs = critic_apply_fn({'params': critic_params}, batch['observations'], actions=actions)
    q = jnp.min(qs, axis=0)

    # Behavior cloning loss (MSE to dataset actions)
    mse = jnp.square(actions - batch['actions']).sum(axis=-1)

    # Normalize Q-values by absolute mean (makes alpha scale-invariant)
    lam = jax.lax.stop_gradient(1 / (jnp.abs(q).mean() + 1e-8))
    actor_loss = -(lam * q).mean()

    # BC loss with coefficient
    bc_loss = (config['alpha_actor'] * mse).mean()

    total_loss = actor_loss + bc_loss

    # Get std for logging
    if config['tanh_squash']:
        action_std = dist._distribution.stddev()
    else:
        action_std = dist.stddev()

    info = {
        'total_loss': total_loss,
        'actor_loss': actor_loss,
        'bc_loss': bc_loss,
        'std': action_std.mean(),
        'mse': mse.mean(),
        'lambda': lam,
    }

    return total_loss, info
