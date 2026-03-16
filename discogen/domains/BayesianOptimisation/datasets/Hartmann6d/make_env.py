from typing import Callable
import jax.numpy as jnp
from domain import Real


def make_env() -> tuple[Callable[[dict[str, jnp.ndarray]], jnp.ndarray], dict[str, Real]]:
    """
    Makes the (negative) Hartmann6d function to maximise and its parameter space.

    Returns:
        A tuple containing the function to maximise and the parameter space.
    """

    def hartmann6d(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.array([params['x1'], params['x2'], params['x3'],
                    params['x4'], params['x5'], params['x6']])
        alpha = jnp.array([1.0, 1.2, 3.0, 3.2])
        A = jnp.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ])
        P = jnp.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381.0],
        ])
        inner = jnp.sum(A * (x - 0.0001 * P) ** 2, axis=1)
        return jnp.sum(alpha * jnp.exp(-inner))

    domain = {"x1": Real(0.0, 1.0), "x2": Real(0.0, 1.0), "x3": Real(0.0, 1.0), "x4": Real(0.0, 1.0), "x5": Real(0.0, 1.0), "x6": Real(0.0, 1.0)}
    return hartmann6d, domain
