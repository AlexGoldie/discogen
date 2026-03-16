import jax.numpy as jnp
from jax.scipy import special as jsp_special
from typing import Any

from surrogate import Surrogate

def acq_fn(
    X_test: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    surrogate: Surrogate,
    surrogate_params: dict[str, Any],
    config: dict[str, Any],
) -> jnp.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.
    Args:
        X_test: Test points.
        X: Training points.
        y: Training values.
        surrogate: Surrogate model.
        surrogate_params: Surrogate model parameters.
        config: Configuration dictionary. [Must contain 'acq_kappa']
    Returns:
        UCB Acquisition Function values.
    """
    mu, var = surrogate.apply(
        surrogate_params, X_test=X_test, X=X, y=y, method=type(surrogate).predict
    )
    sigma = jnp.sqrt(jnp.clip(var, a_min=0.0)) + 1e-9
    kappa = float(config.get("acq_kappa", 2.0))
    return jnp.squeeze(mu + kappa * sigma)


def expected_improvement(
    X_test: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    surrogate: Surrogate,
    surrogate_params: dict[str, Any],
    config: dict[str, Any],
) -> jnp.ndarray:
    """
    Expected Improvement (EI) acquisition function.
    Args:
        X_test: Test points.
        X: Training points.
        y: Training values.
        surrogate: Surrogate model.
        surrogate_params: Surrogate model parameters.
        config: Configuration dictionary. [Must contain 'acq_xi']
    Returns:
        EI Acquisition Function values.
    """

    def _normal_pdf(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)

    def _normal_cdf(z: jnp.ndarray) -> jnp.ndarray:
        # stable cdf via erf
        return 0.5 * (1.0 + jsp_special.erf(z / jnp.sqrt(2.0)))

    ymax = jnp.max(y)
    mu, var = surrogate.apply(
        surrogate_params, X_test=X_test, X=X, y=y, method=type(surrogate).predict
    )
    sigma = jnp.sqrt(jnp.clip(var, a_min=0.0)) + 1e-9

    a = mu - ymax - config['acq_xi']
    z = a / sigma

    ei = a * _normal_cdf(z) + sigma * _normal_pdf(z)
    return jnp.maximum(ei, 0.0).squeeze()
