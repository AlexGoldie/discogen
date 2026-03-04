from typing import Any
import jax.numpy as jnp

def next_queries(obs_samples: jnp.ndarray,
                 obs_values: jnp.ndarray,
                 candidate_samples: jnp.ndarray,
                 candidate_acq_fn_vals: jnp.ndarray,
                 remaining_budget: int,
                 config: dict[str, Any]) -> jnp.ndarray:
    """
    Function that chooses the next queries for the objective function based on the acquisition function values.
    Args:
        obs_samples: The previously observed samples. (We ignore these here, but they could be used to inform next query selection.)
        obs_values: The previously observed values corresponding to the observed samples. (We ignore these here, but they could be used to inform next query selection.)
        candidate_samples: The candidate samples.
        candidate_acq_fn_vals: The corresponding acquisition function values for the candidate samples.
        remaining_budget: The remaining budget for the objective function queries. (We ignore this here, but it could be used to inform next query selection.)
        config: Configuration dictionary. [Must contain 'next_queries_batch_size']
    Returns:
        The next objective function query(ies).
    """
        # --- sort by acquisition function values ---
    sorted_idxs = jnp.argsort(candidate_acq_fn_vals, descending=True)
    sorted_candidate_samples = candidate_samples[sorted_idxs]
    sorted_candidate_acq_fn_vals = candidate_acq_fn_vals[sorted_idxs]

    return sorted_candidate_samples[:config['next_queries_batch_size']] # noqa
