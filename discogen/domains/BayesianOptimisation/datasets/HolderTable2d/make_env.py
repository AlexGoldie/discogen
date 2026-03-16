from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Griewank5d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def holdertable2d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2']])
        term = jnp.abs(1 - jnp.linalg.norm(x) / jnp.pi)
        return jnp.abs(jnp.sin(x[0]) * jnp.cos(x[1]) * jnp.exp(term))

    domain = {"x1": Real(-10.0, 10.0), "x2": Real(-10.0, 10.0)}
    return holdertable2d, domain
