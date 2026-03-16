"""Optimizer configuration for Neural Cellular Automata."""

import optax
from flax import nnx

from config import config


def create_optimizer(nca: nnx.Module) -> nnx.Optimizer:
    """Create optimizer for NCA training.

    Args:
        nca: The NCA system to optimize.

    Returns:
        Configured optimizer.
    """
    opt_config = config["optimizer"]

    lr_schedule = optax.linear_schedule(
        init_value=opt_config["learning_rate"],
        end_value=opt_config["learning_rate"] * opt_config["lr_decay_factor"],
        transition_steps=opt_config["lr_decay_steps"],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(opt_config["max_grad_norm"]),
        optax.adam(learning_rate=lr_schedule),
    )

    # Only optimize update module parameters
    update_params = nnx.All(nnx.Param, nnx.PathContains("update"))
    return nnx.Optimizer(nca, optimizer, wrt=update_params)
