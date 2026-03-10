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

    if "matrix" in config:
        output_ch = config["matrix"]["output_channel"]
        pred = state[..., output_ch : output_ch + 1]
        loss = ...
    else:
        target_channels = target.shape[-1]

        if target_channels == 10: # classifying
            pred_logits = state[..., :target_channels]
            loss = ...
        else:
            pred = state[..., -target_channels:]
            loss = ...


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
