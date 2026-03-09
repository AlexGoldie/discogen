"""Optimizer configuration for Neural Cellular Automata.
"""

import optax
from flax import nnx

from config import config


def create_optimizer(nca: nnx.Module) -> nnx.Optimizer:
    # Input: nca - the NCA model to optimize

    """Fill in optimizer configuration here. You can use your own optimizer logic."""
    # Only optimize update module parameters

    # Must return an nnx.Optimizer wrapping an optax optimizer
