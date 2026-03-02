from typing import Any
import jax.numpy as jnp

def next_queries(obs_samples: jnp.ndarray,
                 obs_values: jnp.ndarray,
                 candidate_samples: jnp.ndarray,
                 candidate_acq_fn_vals: jnp.ndarray,
                 remaining_budget: int,
                 config: dict[str, Any]) -> jnp.ndarray:
    """
    Function that chooses the next objective function queries based on the acquisition function values.
    Args:
        obs_samples: The previously observed samples.
        obs_values: The previously observed values corresponding to the observed samples.
        candidate_samples: The candidate samples.
        candidate_acq_fn_vals: The corresponding acquisition function values for the candidate samples.
        remaining_budget: The remaining budget for the objective function queries.
        config: Configuration dictionary.
    Returns:
        The next objective function query(ies).
    """
    # --- Fill in your choice of next queries here. This can be a single query (just the maximum acquisition function value), or a batch of queries. ---
    # If a batch of queries, you may want to take into account the location of the other queries in the batch, the fixed budget etc.
    return next_queries # noqa
