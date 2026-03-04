import jax.numpy as jnp
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
    Acquisition function to be maximised.
    The acquisition function determines the next point at which to evaluate the objective function. It embeds the surrogate model prediction (mean) and uncertainty (variance) within it.
    Args:
        X_test: Candidate test points at which to evaluate the acquisition function.
        X: Training inputs to condition the surrogate model on (in case the surrogate model is non-parametric).
        y: Training targets to condition the surrogate model on (in case the surrogate model is non-parametric).
        surrogate: Surrogate model.
        surrogate_params: Tuned surrogate model parameters (already fitted to the training data).
        config: Configuration dictionary.
    Returns:
        Acquisition Function values at the candidate test points.
    """
    mu, var = surrogate.apply(
        surrogate_params, X_test=X_test, X=X, y=y, method=type(surrogate).predict
    )
    # --- Fill in your acquisition function here, using some combination of the surrogate model mean and variance predictions.
    # Ensure that the acquisition function is differentiable, so that it can be optimised further locally.
    # Also, ensure that the acquisition function returns a scalar value for each point. acq_fn_values should then be a 1D array of length X_test.shape[0].
    return acq_fn_values # noqa
