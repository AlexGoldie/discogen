import jax
import jax.numpy as jnp

from actor_loss import compute_actor_loss
from critic_loss import compute_critic_loss


def target_update(params, target_params, tau):
    # Inputs:
    # - params: Current network parameters.
    # - target_params: Target network parameters.
    # - tau: Polyak averaging coefficient (0 < tau <= 1).
    #     Typical values: 0.005 for slow updates, 1.0 for hard updates.

    """Fill in your target update logic here."""

    # Your function must return the updated target parameters.
    return target_params


def make_train_step(
    config,
    actor_def,
    critic_def,
    actor_opt,
    critic_opt,
):
    # Inputs:
    # - config: Configuration dictionary with hyperparameters ('tau', etc.)
    # - actor_def: Actor network definition (flax module).
    # - critic_def: Critic network definition (flax module).
    # - actor_opt: Actor optimizer (optax).
    # - critic_opt: Critic optimizer (optax).

    def critic_loss_fn(critic_params, train_state, batch, rng):
        # Inputs:
        # - critic_params: Current critic parameters (for gradient computation).
        # - train_state: Dictionary containing target params and other state.
        # - batch: Batch of transitions.
        # - rng: JAX random key.

        # This function wraps compute_critic_loss for use with jax.grad.
        # Use critic_def.apply and actor_def.apply as the apply functions.

        """Fill in your critic loss wrapper here."""

        # Your function must return the output of compute_critic_loss.
        return compute_critic_loss(
            critic_apply_fn=critic_def.apply,
            target_critic_apply_fn=critic_def.apply,
            target_actor_apply_fn=actor_def.apply,
            config=config,
            critic_params=critic_params,
            target_critic_params=train_state['target_critic_params'],
            target_actor_params=train_state['target_actor_params'],
            batch=batch,
            rng=rng,
        )

    def actor_loss_fn(actor_params, critic_params, batch, rng):
        # Inputs:
        # - actor_params: Current actor parameters (for gradient computation).
        # - critic_params: Updated critic parameters (no gradient).
        # - batch: Batch of transitions.
        # - rng: JAX random key.

        """Fill in your actor loss wrapper here."""

        # Your function must return the output of compute_actor_loss.
        return compute_actor_loss(
            actor_apply_fn=actor_def.apply,
            critic_apply_fn=critic_def.apply,
            config=config,
            actor_params=actor_params,
            critic_params=critic_params,
            batch=batch,
            rng=rng,
        )

    def _train_step(train_state, batch, rng, full_update=True):
        # Inputs:
        # - train_state: Dictionary containing:
        #     'actor_params', 'actor_opt_state', 'critic_params', 'critic_opt_state',
        #     'target_actor_params', 'target_critic_params', 'step'
        # - batch: Dictionary of batched transitions.
        # - rng: JAX random key.
        # - full_update: If True, update actor and targets. If False, only update critic.

        # Update the actor and critic loss function gradients
        # Use them to update the actor and critic parameters
        # Update the target networks as needed
        # Return the updated train_state and info_dict

        """Fill in your training step logic here."""

        new_train_state = {
            'actor_params': train_state['actor_params'],
            'actor_opt_state': train_state['actor_opt_state'],
            'critic_params': train_state['critic_params'],
            'critic_opt_state': train_state['critic_opt_state'],
            'target_actor_params': train_state['target_actor_params'],
            'target_critic_params': train_state['target_critic_params'],
            'step': train_state['step'] + 1,
        }
        info = {}

        # Your function must return a tuple of (updated_train_state, info_dict).
        return new_train_state, info

    # JIT compile with static argument for full_update
    return jax.jit(_train_step, static_argnames=('full_update',))


def sample_actions(actor_def, actor_params, observations, rng, config, temperature=1.0):
    # Inputs:
    # - actor_def: Actor network definition.
    # - actor_params: Actor parameters.
    # - observations: Input observations.
    # - rng: JAX random key.
    # - config: Configuration dictionary.
    # - temperature: Sampling temperature (0 = deterministic).

    """Fill in your action sampling logic here."""

    actions = ...

    # Your function must return sampled actions clipped to [-1, 1].
    return jnp.clip(actions, -1, 1)
