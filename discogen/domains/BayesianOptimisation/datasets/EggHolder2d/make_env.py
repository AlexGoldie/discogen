from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) EggHolder2d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def dropwave2d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2']])
        x1, x2 = x[0], x[1]
        part1 = -(x2 + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * jnp.sin(jnp.sqrt(jnp.abs(x1 - (x2 + 47.0))))
        return -(part1 + part2)

    domain = {"x1": Real(-512.0, 512.0), "x2": Real(-512.0, 512.0)}
    return dropwave2d, domain
