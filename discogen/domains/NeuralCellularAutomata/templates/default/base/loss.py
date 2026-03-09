"""Loss function for Neural Cellular Automata training."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from config import config

NCA = Any
State = Any


def compute_sample_loss(state: State, target: jnp.ndarray) -> jnp.ndarray:
    """Compute loss for a single sample (state, target) pair."""
    # Matrix operations task: MSE on designated output channel
    if "matrix" in config:
        output_ch = config["matrix"]["output_channel"]
        pred = state[..., output_ch : output_ch + 1]
        return jnp.mean(jnp.square(pred - target))

    target_channels = target.shape[-1]

    if target_channels == 10: # classifying
        pred_logits = state[..., :target_channels]
        return jnp.mean(
            jnp.sum(-target * jax.nn.log_softmax(pred_logits, axis=-1), axis=-1)
        )
    else:
        return jnp.mean(jnp.square(pred - target))


def compute_loss(
    nca: NCA,
    state: State,
    target: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, State]:
    """Compute training loss for NCA."""
    train_config = config["train"]
    batch_size = train_config["batch_size"]
    num_steps = train_config["num_steps"]

    # Run NCA with intermediate state capture
    state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})

    nnx.split_rngs(splits=batch_size)(
        nnx.vmap(
            lambda nca, s: nca(s, num_steps=num_steps, sow=True),
            in_axes=(state_axes, 0),
        )
    )(nca, state)

    # Get intermediate states
    intermediates = nnx.pop(nca, nnx.Intermediate)
    all_steps_states = jnp.stack(intermediates["state"])

    # Sample random timestep from second half
    idx = jax.random.randint(key, (batch_size,), num_steps // 2, num_steps)
    final_state = all_steps_states[idx, jnp.arange(batch_size), :, :, :]

    # Compute loss
    losses = jax.vmap(compute_sample_loss)(final_state, target)
    loss = jnp.mean(losses)

    return loss, final_state
