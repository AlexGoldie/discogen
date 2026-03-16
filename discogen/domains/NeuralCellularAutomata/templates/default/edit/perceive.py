"""Perception module for Neural Cellular Automata.
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from config import config

State = Any
Perception = Any


class Perceive(nnx.Module):
    """Perception module that maps state to perception."""

    def __init__(self, *, rngs: nnx.Rngs):
        # rngs: rng key for initialization

        """Fill in initialization here."""
        pass

    def __call__(self, state: State) -> Perception:
        # Input: state with shape (*spatial_dims, channel_size)

        """Fill in perception logic here."""

        # Must return perception with shape (*spatial_dims, perception_size)
        # where perception_size is determined by your implementation
        return perception


def create_perceive(rngs: nnx.Rngs) -> nnx.Module:
    return Perceive(rngs=rngs)
