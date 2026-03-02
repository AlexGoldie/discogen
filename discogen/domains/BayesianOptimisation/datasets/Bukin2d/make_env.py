from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Branin2d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def bukin2d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2']])
        part1 = 100.0 * jnp.sqrt(jnp.abs(x[1] - 0.01 * x[0]**2))
        part2 = 0.01 * jnp.abs(x[0] + 10.0)
        return -(part1 + part2)

    domain = {"x1": Real(-15.0, -5.0), "x2": Real(-3.0, 3.0)}
    return bukin2d, domain
