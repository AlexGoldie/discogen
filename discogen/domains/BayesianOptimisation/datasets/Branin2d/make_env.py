from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Branin2d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def branin2d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2']])
        t1 = (
            x[1]
            - 5.1 / (4 * jnp.pi**2) * x[0]**2
            + 5 / jnp.pi * x[0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * jnp.pi)) * jnp.cos(x[0])
        return -(t1**2 + t2 + 10)

    domain = {"x1": Real(-5.0, 10.0), "x2": Real(0.0, 15.0)}
    return branin2d, domain
