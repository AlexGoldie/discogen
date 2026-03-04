from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) 1D Ackley function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """
    def ackley1d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1']])
        a, b, c = 20, 0.2, 2 * jnp.pi
        part1 = -a * jnp.exp(-jnp.linalg.norm(x) * b / jnp.sqrt(2))
        part2 = -(jnp.exp(jnp.mean(jnp.cos(c * x))))
        return -(part1 + part2 + a + jnp.exp(1))

    domain = {"x1": Real(-32.768, 32.768)}
    return ackley1d, domain
