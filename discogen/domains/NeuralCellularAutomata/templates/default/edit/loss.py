"""Loss function for Neural Cellular Automata training.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from config import config

NCA = Any
State = Any


def compute_sample_loss(state: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    # Inputs:
    # - state: cell state with shape (*spatial_dims, channel_size)
    # - target: target image with shape (*spatial_dims, target_channels)

    """Fill in per-sample loss logic here."""

    # Must return a scalar loss value
    return loss


def compute_loss(
    nca: NCA,
    state: State,
    target: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, State]:
    # Inputs:
    # - nca: the NCA model (call with nca(state, num_steps=N) to run N steps)
    # - state: batch of states with shape (batch, *spatial_dims, channel_size)
    # - target: batch of targets with shape (batch, *spatial_dims, target_channels)
    # - key: PRNG key

    """Fill in batch loss logic here."""

    # Must return (scalar loss, final state batch)
    return loss, final_state
