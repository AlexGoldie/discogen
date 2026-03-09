"""Update module for Neural Cellular Automata.

The update module is the neural network that computes how each cell's state
changes based on its perception. This is the core "rule" of the NCA.
"""

from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import default_kernel_init

from config import config

State = Any
Perception = Any
Input = Any


def state_to_alive(state: State) -> State:
    """Extract the 'alive' channel from state (last channel)."""
    return state[..., -1:]


class NCAUpdate(nnx.Module):
    """Neural Cellular Automata update module.

    Implements a residual MLP update with:
    - Stochastic cell dropout for robustness
    - Optional alive masking to prevent dead cells from updating
    - Zero-initialized final layer for stable training
    """

    def __init__(
        self,
        channel_size: int,
        perception_size: int,
        hidden_layer_sizes: tuple[int, ...],
        *,
        activation_fn=nnx.relu,
        step_size: float = 1.0,
        cell_dropout_rate: float = 0.0,
        use_alive_masking: bool = True,
        alive_threshold: float = 0.1,
        zeros_init: bool = True,
        rngs: nnx.Rngs,
    ):
        """Initialize NCA update module.

        Args:
            channel_size: Number of state channels.
            perception_size: Size of perception input.
            hidden_layer_sizes: Sizes of hidden layers in MLP.
            activation_fn: Activation function.
            step_size: Multiplier for residual update.
            cell_dropout_rate: Dropout rate for cell updates.
            use_alive_masking: Whether to apply alive masking.
            alive_threshold: Threshold for alive masking.
            zeros_init: Whether to zero-initialize final layer.
            rngs: Flax NNX random number generators.
        """
        self.step_size = step_size
        self.use_alive_masking = use_alive_masking
        self.alive_threshold = alive_threshold
        self.activation_fn = activation_fn

        # Dropout with spatial broadcasting (same mask for all channels at each position)
        self.dropout = nnx.Dropout(
            rate=cell_dropout_rate,
            broadcast_dims=(-1,),
            rngs=rngs,
        )

        # Max pool for alive masking
        self.pool = partial(nnx.max_pool, window_shape=(3, 3), padding="SAME")

        # Build MLP layers as 1x1 convolutions
        in_features = (perception_size,) + hidden_layer_sizes
        out_features = hidden_layer_sizes + (channel_size,)

        # Use zeros init for final layer, default for others
        kernel_inits = [default_kernel_init] * len(hidden_layer_sizes)
        kernel_inits.append(initializers.zeros_init() if zeros_init else default_kernel_init)

        self.layers = nnx.List([])
        for in_f, out_f, k_init in zip(in_features, out_features, kernel_inits):
            self.layers.append(
                nnx.Conv(
                    in_features=in_f,
                    out_features=out_f,
                    kernel_size=(1, 1),
                    kernel_init=k_init,
                    rngs=rngs,
                )
            )

    def get_alive_mask(self, state: State) -> jnp.ndarray:
        """Compute mask of alive cells based on neighborhood.

        A cell is alive if any cell in its 3x3 neighborhood has
        alive channel > threshold.

        Args:
            state: Current state.

        Returns:
            Boolean mask with same spatial shape as state.
        """
        alive = state_to_alive(state)
        return self.pool(alive) > self.alive_threshold

    def __call__(
        self,
        state: State,
        perception: Perception,
        input: Input | None = None,
    ) -> State:
        """Compute state update.

        Args:
            state: Current state with shape (*spatial_dims, channel_size).
            perception: Perception with shape (*spatial_dims, perception_size).
            input: Optional external input (concatenated to perception).

        Returns:
            Next state with same shape as input state.
        """
        # Compute pre-update alive mask (if using alive masking)
        if self.use_alive_masking:
            pre_alive = self.get_alive_mask(state)

        # Concatenate input if provided
        x = perception
        if input is not None:
            x = jnp.concatenate([x, input], axis=-1)

        # MLP forward pass
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        update = self.layers[-1](x)

        # Apply dropout and residual connection
        update = self.dropout(update)
        next_state = state + self.step_size * update

        # Apply alive mask (intersection of pre and post alive)
        if self.use_alive_masking:
            post_alive = self.get_alive_mask(next_state)
            alive_mask = pre_alive & post_alive
            next_state = alive_mask * next_state

        return next_state


def create_update(rngs: nnx.Rngs) -> nnx.Module:
    """Factory function to create an update module."""
    nca_config = config["nca"]
    perception_size = nca_config.get("num_kernels", 3) * nca_config["channel_size"]

    return NCAUpdate(
        channel_size=nca_config["channel_size"],
        perception_size=perception_size,
        hidden_layer_sizes=(nca_config.get("hidden_size", 128),),
        cell_dropout_rate=nca_config.get("cell_dropout_rate", 0.5),
        use_alive_masking=nca_config.get("use_alive_masking", True),
        alive_threshold=nca_config.get("alive_threshold", 0.1),
        zeros_init=True,
        rngs=rngs,
    )
