import optax
from typing import Any, Callable
import jax
import jax.numpy as jnp
from functools import partial

from surrogate import Surrogate


def build_acq_fn_gradient_optimizer(config:dict[str, Any]) -> optax.GradientTransformation:
    """
    Builds the gradient optimizer for the acquisition function.
    Args:
        config: Configuration dictionary.
    Returns:
        Gradient optimizer for the acquisition function.
    """
    return optax.lbfgs()


def gradient_acq_fn_optimizer(sample_point: jnp.ndarray,
                  acq_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, Surrogate, dict[str, Any], dict[str, Any]], jnp.ndarray],
                  acq_gradient_optimizer: optax.GradientTransformation,
                  config: dict[str, Any]) -> jnp.ndarray:
    """
    Function to locally optimise the acquisition function at a single point.
    Args:
        sample_point: Single point to optimize.
        acq: Acquisition function to maximize. Must return a scalar value and support automatic differentiation.
        acq_gradient_optimizer: Optimizer to use for gradient optimization.
        config: Configuration dictionary.
    Returns:
        The optimized point after `config['acq_gradient_max_iter']` iterations.
    """
    # --- optimise the acquisition function i.e. minimise negative acquisition function ---
    value_and_grad_fun = optax.value_and_grad_from_state(lambda x: -acq_fn(x))

    def step(carry: tuple, _: None) -> tuple[tuple, None]:
        sample_point, state = carry
        value, grad = value_and_grad_fun(sample_point, state=state)
        updates, state = acq_gradient_optimizer.update(grad, state, sample_point, value=value, grad=grad, value_fn=acq_fn)
        sample_point = optax.apply_updates(sample_point, updates)
        sample_point = jnp.clip(sample_point, -1e6, 1e6)
        return (sample_point, state), None

    init_carry = (sample_point, acq_gradient_optimizer.init(sample_point))
    (final_sample_point, _), __ = jax.lax.scan(step, init_carry, None, length=config['acq_gradient_max_iter'])

    return final_sample_point


def acq_fn_optimizer(candidate_samples: jnp.ndarray,
            acq_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, Surrogate, dict[str, Any], dict[str, Any]], jnp.ndarray],
            acq_fn_gradient_optimizer: optax.GradientTransformation,
            X: jnp.ndarray,
            y: jnp.ndarray,
            surrogate: Surrogate,
            surrogate_params: dict[str, Any],
            config: dict[str, Any]) -> dict[str, Any]:
    """
    Function to globally optimise the acquisition function over a set of candidate points.
    Further optimises the query by implementing local, gradient-based optimisation of the acquisition function at the top N points.
    Args:
        candidate_samples: Candidate points to optimize.
        acq_fn: Acquisition function to maximize. Must return a scalar value and support automatic differentiation.
        acq_fn_gradient_optimizer: Optimizer to use for gradient optimization.
        X: Training points.
        y: Training values.
        surrogate: Surrogate model.
        surrogate_params: Surrogate model parameters.
        config: Configuration dictionary.
    Returns:
        The sorted candidate samples and corresponding sorted acquisition function values.
    """

    # --- compute acquisition function values for sampled points ---
    partial_acq_fn = partial(acq_fn, X=X, y=y, surrogate=surrogate, surrogate_params=surrogate_params, config=config)
    acq_vals = partial_acq_fn(candidate_samples)

    # --- find best N points and optimize acquisition function locally ---
    top_idxs = jnp.argsort(acq_vals)[-config['acq_top_n_samples']:]
    init_points = candidate_samples[top_idxs]
    opt_candidate_samples = jax.vmap(lambda x: gradient_acq_fn_optimizer(x[None, :], partial_acq_fn, acq_fn_gradient_optimizer, config=config).squeeze(0))(init_points)
    opt_acq_vals = partial_acq_fn(opt_candidate_samples)

    # --- concatenate acquisition function values and sampled points ---
    candidate_samples = jnp.concatenate([candidate_samples, opt_candidate_samples], axis=0)
    candidate_acq_fn_vals = jnp.concatenate([acq_vals, opt_acq_vals], axis=0)

    # --- return candidate samples and corresponding acquisition function values ---
    return candidate_samples, candidate_acq_fn_vals
