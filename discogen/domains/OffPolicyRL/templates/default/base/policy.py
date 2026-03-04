import jax
import jax.numpy as jnp

def explore(rng, q_vals, t, config):

    rng_a, rng_e = jax.random.split(
        rng, 2
    )  # a key for sampling random actions and one for picking
    eps = jnp.clip(  # get epsilon
        (
            (config["EPSILON_FINISH"] - config["EPSILON_START"])
            / config["EPSILON_ANNEAL_TIME"]
        )
        * t
        + config["EPSILON_START"],
        config["EPSILON_FINISH"],
    )
    greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape)
        < eps,  # pick the actions that should be random
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),  # sample random actions,
        greedy_actions,
    )
    return chosen_actions

def exploit(rng, q_vals, t, config):
    greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
    return greedy_actions
