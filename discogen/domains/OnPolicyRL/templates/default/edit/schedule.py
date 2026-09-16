import jax
import jax.numpy as jnp

def make_schedule_fn(config, lr):
    # lr is negative to implement gradient DESCENT under optax syntax.
    def schedule_fn(count):
        return ...

    return schedule_fn
