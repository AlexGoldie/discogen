import jax
import jax.numpy as jnp


def get_targets(traj_batch, last_val, config):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        value, reward, next_done = (
            transition.value,
            transition.reward,
            transition.next_done,
        )
        delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value
