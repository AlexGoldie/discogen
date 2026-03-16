"""Perception module for Neural Cellular Automata."""

from itertools import product
from typing import Any

import jax.numpy as jnp
from flax import nnx

from config import config

State = Any
Perception = Any


def identity_kernel(ndim: int = 2) -> jnp.ndarray:
    """Create an identity kernel (selects center cell only)."""
    kernel = jnp.zeros((3,) * ndim)
    center_idx = (1,) * ndim
    kernel = kernel.at[center_idx].set(1.0)
    return jnp.expand_dims(kernel, axis=-1)


def grad_kernel(ndim: int = 2, normalize: bool = True) -> jnp.ndarray:
    """Create Sobel gradient kernels for edge detection."""
    grad = jnp.array([-1.0, 0.0, 1.0])
    smooth = jnp.array([1.0, 2.0, 1.0])

    kernels = []
    for i in range(ndim):
        kernel = jnp.ones((3,) * ndim)
        for j in range(ndim):
            axis_kernel = grad if i == j else smooth
            shape = [1] * ndim
            shape[j] = 3
            kernel = kernel * axis_kernel.reshape(shape)
        kernels.append(kernel)

    if normalize:
        kernels = [k / jnp.sum(jnp.abs(k)) for k in kernels]

    return jnp.stack(kernels, axis=-1)


class ConvPerceive(nnx.Module):
    """Convolutional perception module with fixed Sobel + identity kernels."""

    def __init__(self, *, rngs: nnx.Rngs):
        nca_config = config["nca"]
        channel_size = nca_config["channel_size"]
        num_kernels = nca_config.get("num_kernels", 3)
        perception_size = num_kernels * channel_size

        self.conv = nnx.Conv(
            in_features=channel_size,
            out_features=perception_size,
            kernel_size=(3, 3),
            padding="CIRCULAR",
            feature_group_count=channel_size,
            use_bias=False,
            rngs=rngs,
        )

        kernel = jnp.concatenate([identity_kernel(ndim=2), grad_kernel(ndim=2)], axis=-1)
        kernel = jnp.expand_dims(jnp.concatenate([kernel] * channel_size, axis=-1), axis=-2)
        self.conv.kernel.value = kernel

    def __call__(self, state: State) -> Perception:
        return self.conv(state)


def create_perceive(rngs: nnx.Rngs) -> nnx.Module:
    return ConvPerceive(rngs=rngs)
