from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) DropWave2d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def dropwave2d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2']])
        norm = jnp.linalg.norm(x)
        part1 = 1.0 + jnp.cos(12.0 * norm)
        part2 = 0.5 * norm**2 + 2.0
        return part1 / part2

    domain = {"x1": Real(-5.12, 5.12), "x2": Real(-5.12, 5.12)}
    return dropwave2d, domain
