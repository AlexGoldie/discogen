import jax
import jax.numpy as jnp

from actor_loss import compute_actor_loss
from critic_loss import compute_critic_loss


def target_update(params, target_params, tau):
    """Polyak averaging update for target networks.

    Args:
        params: Current network parameters.
        target_params: Target network parameters.
        tau: Polyak averaging coefficient (0 < tau <= 1).

    Returns:
        Updated target parameters.
    """
    return jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        params,
        target_params,
    )


def make_train_step(
    config,
    actor_def,
    critic_def,
    actor_opt,
    critic_opt,
):
    """Create the JIT-compiled training step function.

    Args:
        config: Configuration dictionary with hyperparameters.
        actor_def: Actor network definition.
        critic_def: Critic network definition.
        actor_opt: Actor optimizer (optax).
        critic_opt: Critic optimizer (optax).

    Returns:
        A function that performs one training step.
    """

    def _train_step(train_state, batch, rng, full_update=True):
        """Perform one training step.

        Args:
            train_state: Dictionary containing all network states.
            batch: Batch of transitions.
            rng: JAX random key.
            full_update: Whether to update the actor (delayed updates).

        Returns:
            Tuple of (updated_train_state, info_dict).
        """
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        info = {}

        # --- Critic Update ---
        def critic_loss_fn(critic_params):
            return compute_critic_loss(
                critic_apply_fn=critic_def.apply,
                target_critic_apply_fn=critic_def.apply,
                target_actor_apply_fn=actor_def.apply,
                config=config,
                critic_params=critic_params,
                target_critic_params=train_state['target_critic_params'],
                target_actor_params=train_state['target_actor_params'],
                batch=batch,
                rng=critic_rng,
            )

        critic_grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(
            train_state['critic_params']
        )

        # Apply critic gradients
        critic_updates, new_critic_opt_state = critic_opt.update(
            critic_grads,
            train_state['critic_opt_state'],
            train_state['critic_params'],
        )
        new_critic_params = jax.tree_util.tree_map(
            lambda p, u: p + u,
            train_state['critic_params'],
            critic_updates,
        )

        info.update({f'critic/{k}': v for k, v in critic_info.items()})

        # --- Actor Update (conditional) ---
        if full_update:
            def actor_loss_fn(actor_params):
                return compute_actor_loss(
                    actor_apply_fn=actor_def.apply,
                    critic_apply_fn=critic_def.apply,
                    config=config,
                    actor_params=actor_params,
                    critic_params=new_critic_params,  # Use updated critic
                    batch=batch,
                    rng=actor_rng,
                )

            actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(
                train_state['actor_params']
            )

            # Apply actor gradients
            actor_updates, new_actor_opt_state = actor_opt.update(
                actor_grads,
                train_state['actor_opt_state'],
                train_state['actor_params'],
            )
            new_actor_params = jax.tree_util.tree_map(
                lambda p, u: p + u,
                train_state['actor_params'],
                actor_updates,
            )

            info.update({f'actor/{k}': v for k, v in actor_info.items()})

            # --- Target Network Updates ---
            new_target_critic_params = target_update(
                new_critic_params,
                train_state['target_critic_params'],
                config['tau'],
            )
            new_target_actor_params = target_update(
                new_actor_params,
                train_state['target_actor_params'],
                config['tau'],
            )
        else:
            # No actor update
            new_actor_params = train_state['actor_params']
            new_actor_opt_state = train_state['actor_opt_state']
            new_target_critic_params = train_state['target_critic_params']
            new_target_actor_params = train_state['target_actor_params']

        # Build updated train state
        new_train_state = {
            'actor_params': new_actor_params,
            'actor_opt_state': new_actor_opt_state,
            'critic_params': new_critic_params,
            'critic_opt_state': new_critic_opt_state,
            'target_actor_params': new_target_actor_params,
            'target_critic_params': new_target_critic_params,
            'step': train_state['step'] + 1,
        }

        return new_train_state, info

    return jax.jit(_train_step, static_argnames=('full_update',))


def sample_actions(actor_def, actor_params, observations, rng, config, temperature=1.0):
    """Sample actions from the actor policy.

    Args:
        actor_def: Actor network definition.
        actor_params: Actor parameters.
        observations: Input observations.
        rng: JAX random key.
        config: Configuration dictionary.
        temperature: Sampling temperature.

    Returns:
        Sampled actions clipped to [-1, 1].
    """
    dist = actor_def.apply({'params': actor_params}, observations, temperature=temperature)
    actions = dist.mode()

    # Add exploration noise
    noise = jnp.clip(
        jax.random.normal(rng, actions.shape) * config['actor_noise'] * temperature,
        -config['actor_noise_clip'],
        config['actor_noise_clip'],
    )
    actions = jnp.clip(actions + noise, -1, 1)

    return actions
