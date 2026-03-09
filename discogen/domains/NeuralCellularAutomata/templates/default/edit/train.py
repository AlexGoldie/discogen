"""Training loop for Neural Cellular Automata.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from config import config
from loss import compute_loss, compute_sample_loss
from make_dataset import sample_state
from pool import Pool

NCA = Any
Optimizer = Any


def create_train_step(nca: NCA, optimizer: Optimizer) -> Callable:
    # Inputs:
    # - nca: the NCA model
    # - optimizer: the nnx.Optimizer

    @nnx.jit
    def train_step(
        nca: NCA,
        optimizer: Optimizer,
        pool: Pool,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, Pool]:
        # Inputs:
        # - nca: the NCA model
        # - optimizer: the nnx.Optimizer
        # - pool: contains {"state": states, "target": targets} arrays
        # - key: PRNG key

        """Fill in training step logic here."""

        # Must return (scalar loss, updated pool)
        return loss, pool

    return train_step
