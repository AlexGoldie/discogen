import jax.numpy as jnp

def q_loss_fn(params, target_params, network, learn_batch, config):

    # used for action selection
    q_next_online = network.apply(
        params, learn_batch.second.obs
    )
    next_action = jnp.argmax(q_next_online, axis=-1)


    q_next_target = network.apply(
        target_params, learn_batch.second.obs
    )

    q_target_val = jnp.take_along_axis(
        q_next_target,
        jnp.expand_dims(next_action, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)


    target = (
        learn_batch.first.reward
        + (1 - learn_batch.first.done) * config["GAMMA"] * q_target_val
    )

    q_online = network.apply(
        params, learn_batch.first.obs
    )

    chosen_action_qvals = jnp.take_along_axis(
        q_online,
        jnp.expand_dims(learn_batch.first.action, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)

    # Calculate loss
    loss = jnp.mean((chosen_action_qvals - target) ** 2)
    return loss
