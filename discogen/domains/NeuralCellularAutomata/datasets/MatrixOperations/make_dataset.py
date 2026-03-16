"""Dataset utilities for MatrixOperations NCA task."""

from pathlib import Path

import jax
import jax.numpy as jnp


def download_dataset(cache_dir: Path) -> None:
    """No download needed - matrices are generated procedurally."""
    cache_dir.mkdir(parents=True, exist_ok=True)


def _compute_operation(A: jnp.ndarray, B: jnp.ndarray, op_idx: int) -> jnp.ndarray:
    """Compute the result of a matrix operation.

    Operations:
        0: transpose(A)
        1: negate(A) = -A
        2: add(A, B) = A + B
        3: multiply(A, B) = A @ B
    """
    return jax.lax.switch(
        op_idx,
        [
            lambda: A.T,
            lambda: -A,
            lambda: A + B,
            lambda: A @ B,
        ],
    )


_GAUSSIAN_KERNEL_SIZE = 3
_SPARSITY = 0.15  # Fraction of non-zero elements in sparse matrices


def _gaussian_kernel(size: int) -> jnp.ndarray:
    """Create a normalised 2D Gaussian smoothing kernel."""
    coords = jnp.arange(size) - size // 2
    g = jnp.exp(-0.5 * coords**2)
    kernel = jnp.outer(g, g)
    return kernel / kernel.sum()


def _sample_matrix(key: jax.Array, size: int, dist_idx: int) -> jnp.ndarray:
    """Sample a matrix from one of four distributions.

    Distributions:
        0: Uniform in [-1, 1]
        1: Gaussian (standard normal, clipped to [-3, 3])
        2: Spatially correlated (smoothed Gaussian noise)
        3: Sparse (Gaussian values, ~15% non-zero)
    """
    k1, k2 = jax.random.split(key)

    def uniform(_):
        return jax.random.uniform(k1, (size, size), minval=-1.0, maxval=1.0)

    def gaussian(_):
        return jnp.clip(jax.random.normal(k1, (size, size)), -3.0, 3.0) / 3.0

    def spatially_correlated(_):
        noise = jax.random.normal(k1, (size, size))
        kernel = _gaussian_kernel(_GAUSSIAN_KERNEL_SIZE)
        kernel_4d = kernel[None, None, :, :]
        noise_4d = noise[None, None, :, :]
        smoothed = jax.lax.conv(noise_4d, kernel_4d, window_strides=(1, 1), padding="SAME")
        smoothed = smoothed[0, 0]
        return smoothed / (jnp.std(smoothed) + 1e-6)

    def sparse(_):
        values = jax.random.normal(k1, (size, size))
        mask = jax.random.bernoulli(k2, _SPARSITY, shape=(size, size))
        return values * mask

    return jax.lax.switch(dist_idx, [uniform, gaussian, spatially_correlated, sparse], None)


def sample_state(config: dict, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random matrix operation task.

    Args:
        config: Configuration dictionary.
        key: JAX PRNG key.

    Returns:
        state: Shape (size, size, channel_size) with input matrices and op encoding
        target: Shape (size, size, 1) with expected output matrix
    """
    matrix_config = config["matrix"]
    nca_config = config["nca"]

    size = matrix_config["size"]
    num_ops = len(matrix_config["operations"])
    channel_size = nca_config["channel_size"]
    input_channels = matrix_config["input_channels"]
    preserve_channels = nca_config["preserve_channels"]

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Generate random input matrices with diverse distributions
    num_distributions = 4
    dist_idx_A = jax.random.randint(k4, (), 0, num_distributions)
    dist_idx_B = jax.random.randint(k5, (), 0, num_distributions)
    A = _sample_matrix(k1, size, dist_idx_A)
    B = _sample_matrix(k2, size, dist_idx_B)

    # Random operation
    op_idx = jax.random.randint(k3, (), 0, num_ops)

    # Zero out B for unary operations (transpose, negate only use A)
    operations = matrix_config["operations"]
    is_unary = jnp.array([op in ("transpose", "negate") for op in operations])[op_idx]
    B = jnp.where(is_unary, jnp.zeros_like(B), B)

    # Compute target output
    result = _compute_operation(A, B, op_idx)

    # Build state tensor
    state = jnp.zeros((size, size, channel_size))

    # Place input matrices in designated channels
    state = state.at[..., input_channels[0]].set(A)
    state = state.at[..., input_channels[1]].set(B)

    # Place operation one-hot in last `num_ops` channels (within preserved range)
    op_onehot = jax.nn.one_hot(op_idx, num_ops)
    # Broadcast one-hot to all spatial positions
    op_broadcast = jnp.broadcast_to(op_onehot, (size, size, num_ops))
    state = state.at[..., -num_ops:].set(op_broadcast)

    # Target is the result matrix (single channel)
    target = result[..., None]

    return state, target
