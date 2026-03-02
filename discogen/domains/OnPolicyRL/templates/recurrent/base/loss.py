from typing import Any, Callable

import flax
import jax.numpy as jnp


def loss_actor_and_critic(
    params, init_hstate, traj_batch, gae, targets, network, config
):
    # RERUN NETWORK
    _, pi, value = network.apply(
        params, init_hstate[0], (traj_batch.obs, traj_batch.done)
    )
    log_prob = pi.log_prob(traj_batch.action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -config["CLIP_EPS"], config["CLIP_EPS"]
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)
