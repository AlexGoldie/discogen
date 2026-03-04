from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Griewank5d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def griewank5d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2'], params['x3'],
                    params['x4'], params['x5']])
        part1 = jnp.sum(x**2 / 4000.0)
        d = x.shape[-1]
        part2 = -(jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, d + 1)))))
        return -(part1 + part2 + 1.0)

    domain = {"x1": Real(-600.0, 600.0), "x2": Real(-600.0, 600.0), "x3": Real(-600.0, 600.0), "x4": Real(-600.0, 600.0), "x5": Real(-600.0, 600.0)}
    return griewank5d, domain
