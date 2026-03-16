import jax.numpy as jnp

def loss_actor_and_critic(params, init_hstate, traj_batch, gae, targets, network, config):
    # Inputs:
    if config.get("GET_AVAIL_ACTIONS", False):
        net_ins = (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
    else:
        net_ins = (traj_batch.obs, traj_batch.done)
    # RERUN NETWORK
    _, pi, value = network.apply(
        params,
        init_hstate.squeeze(),
        net_ins,
    )
    log_prob = pi.log_prob(traj_batch.action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(
        value_losses, value_losses_clipped
    ).mean()

    # CALCULATE ACTOR LOSS
    logratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(logratio)
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

    # debug
    approx_kl = ((ratio - 1) - logratio).mean()
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)
