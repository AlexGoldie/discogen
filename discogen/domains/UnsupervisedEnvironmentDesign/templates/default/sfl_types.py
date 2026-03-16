import jax.numpy as jnp
from typing import NamedTuple, Any
class Transition(NamedTuple):
    next_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

Level = Any

EnvParams = Any
Environment = Any
