"""Training loop for Neural Cellular Automata."""

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
    """Create a JIT-compiled training step function.

    Uses pool-based training with curriculum learning.

    Args:
        nca: The NCA system to train.
        optimizer: The optimizer.

    Returns:
        JIT-compiled function: (nca, optimizer, pool, key) -> (loss, pool)
    """
    train_config = config["train"]
    batch_size = train_config["batch_size"]

    update_params = nnx.All(nnx.Param, nnx.PathContains("update"))

    @nnx.jit
    def train_step(
        nca: NCA,
        optimizer: Optimizer,
        pool: Pool,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, Pool]:
        sample_key, loss_key, new_sample_key = jax.random.split(key, 3)

        # Sample batch from pool
        pool_idx, batch = pool.sample(sample_key, batch_size=batch_size)
        current_state = batch["state"]
        current_target = batch["target"]

        # Sort by descending loss (curriculum learning)
        losses = jax.vmap(compute_sample_loss, in_axes=(0, 0))(current_state, current_target)
        sort_idx = jnp.argsort(losses, descending=True)
        pool_idx = pool_idx[sort_idx]
        current_state = current_state[sort_idx]
        current_target = current_target[sort_idx]

        # Replace worst sample with fresh seed
        new_state, new_target = sample_state(config, new_sample_key)
        current_state = current_state.at[0].set(new_state)
        current_target = current_target.at[0].set(new_target)

        # Compute loss and gradients
        (loss, current_state), grads = nnx.value_and_grad(
            compute_loss,
            has_aux=True,
            argnums=nnx.DiffState(0, update_params),
        )(nca, current_state, current_target, loss_key)

        # Update parameters
        optimizer.update(nca, grads)

        # Update pool with evolved states
        pool = pool.update(pool_idx, {"state": current_state, "target": current_target})

        return loss, pool

    return train_step
