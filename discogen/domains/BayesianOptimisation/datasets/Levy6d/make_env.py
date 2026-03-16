from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Levy6d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def levy6d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2'], params['x3'], params['x4'], params['x5'], params['x6']])
        w = 1.0 + (x - 1.0) / 4.0
        part1 = jnp.sin(jnp.pi * w[0])**2
        part2 = jnp.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w[:-1] + 1.0)**2))
        part3 = (w[-1] - 1.0)**2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[-1])**2)
        return -(part1 + part2 + part3)

    domain = {"x1": Real(-10.0, 10.0), "x2": Real(-10.0, 10.0), "x3": Real(-10.0, 10.0), "x4": Real(-10.0, 10.0), "x5": Real(-10.0, 10.0), "x6": Real(-10.0, 10.0)}
    return levy6d, domain
