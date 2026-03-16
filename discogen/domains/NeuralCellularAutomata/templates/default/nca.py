"""Neural Cellular Automata system definition."""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from config import config

# Type aliases
State = Any
Input = Any
Perception = Any


def rgba_to_rgb(rgba: jnp.ndarray) -> jnp.ndarray:
    """Convert RGBA to RGB with white background."""
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def clip_and_uint8(x: jnp.ndarray) -> jnp.ndarray:
    """Clip values to [0, 1] and convert to uint8."""
    return (jnp.clip(x, 0.0, 1.0) * 255).astype(jnp.uint8)


class NCA(nnx.Module):
    """Neural Cellular Automata system.

    Combines a perception module and update module to form a complete NCA.
    The system can be run for multiple steps and supports intermediate state capture.

    Supports two modes:
    - Growing mode (preserve_channels=0): State evolves freely
    - Classifying mode (preserve_channels>0): Input channels are preserved each step
    """

    def __init__(self, perceive: nnx.Module, update: nnx.Module):
        """Initialize NCA with perceive and update modules."""
        self.perceive = perceive
        self.update = update
        self.preserve_channels = config["nca"].get("preserve_channels", 0)

    def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
        """Perform a single NCA step.

        Args:
            state: Current grid state with shape (*spatial_dims, channel_size).
            input: Optional external input (unused in standard NCA).
            sow: Whether to store intermediate states.

        Returns:
            Next state with same shape as input state.
        """
        # Preserve input channels if configured (for classifying tasks)
        if self.preserve_channels > 0:
            preserved = state[..., -self.preserve_channels:]

        perception = self.perceive(state)
        next_state = self.update(state, perception, input)

        # Restore preserved channels
        if self.preserve_channels > 0:
            next_state = next_state.at[..., -self.preserve_channels:].set(preserved)

        if sow:
            self.sow(nnx.Intermediate, "state", next_state)

        return next_state

    def __call__(
        self,
        state: State,
        num_steps: int,
        input: Input | None = None,
        *,
        sow: bool = False,
    ) -> State:
        """Run the NCA for multiple steps.

        Args:
            state: Initial state with shape (*spatial_dims, channel_size).
            num_steps: Number of steps to run.
            input: Optional external input.
            sow: Whether to store intermediate states.

        Returns:
            Final state after num_steps iterations.
        """
        for _ in range(num_steps):
            state = self._step(state, input, sow=sow)
        return state

    @nnx.jit
    def render(self, state: State) -> jnp.ndarray:
        """Render state to RGB image.

        Extracts the last 4 channels as RGBA and converts to RGB.

        Args:
            state: NCA state with shape (*spatial_dims, channel_size).

        Returns:
            RGB image with shape (*spatial_dims, 3) as uint8.
        """
        rgba = state[..., -4:]
        rgb = rgba_to_rgb(rgba)
        return clip_and_uint8(rgb)

    @nnx.jit
    def render_rgba(self, state: State) -> jnp.ndarray:
        """Render state to RGBA image.

        Extracts the last 4 channels as RGBA.

        Args:
            state: NCA state with shape (*spatial_dims, channel_size).

        Returns:
            RGBA image with shape (*spatial_dims, 4) as uint8.
        """
        rgba = state[..., -4:]
        return clip_and_uint8(rgba)
