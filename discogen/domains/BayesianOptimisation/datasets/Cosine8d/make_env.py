from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Cosine8d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def cosine8d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2'], params['x3'],
                    params['x4'], params['x5'], params['x6'], params['x7'], params['x8']])
        return -(jnp.sum(0.1 * jnp.cos(5 * jnp.pi * x) - x**2))

    domain = {"x1": Real(-1.0, 1.0), "x2": Real(-1.0, 1.0), "x3": Real(-1.0, 1.0), "x4": Real(-1.0, 1.0), "x5": Real(-1.0, 1.0), "x6": Real(-1.0, 1.0), "x7": Real(-1.0, 1.0), "x8": Real(-1.0, 1.0)}
    return cosine8d, domain
