"""Update module for Neural Cellular Automata.
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from config import config

State = Any
Perception = Any
Input = Any


class NCAUpdate(nnx.Module):
    """Update module that computes state transitions based on perception."""

    def __init__(self, *, rngs: nnx.Rngs):
        # rngs: rng key for initialization

        """Fill in initialization here."""
        pass

    def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
        # Inputs:
        # - state: current state with shape (*spatial_dims, channel_size)
        # - perception: output from Perceive with shape (*spatial_dims, perception_size)
        # - input: optional input (used in classification tasks)

        """Fill in update logic here."""

        # Must return next state with shape (*spatial_dims, channel_size)
        return next_state


def create_update(rngs: nnx.Rngs) -> nnx.Module:
    return NCAUpdate(rngs=rngs)
