"""Dataset utilities for SelfClassifyingMNIST task."""

from pathlib import Path

import jax
import jax.numpy as jnp

_mnist_cache = {}


def download_dataset(cache_dir: Path) -> None:
    """Download MNIST dataset using torchvision."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mnist_path = cache_dir / "mnist_train.npz"
    if not mnist_path.exists():
        print("Downloading MNIST dataset...")
        import torchvision
        ds_train = torchvision.datasets.MNIST(root=str(cache_dir), train=True, download=True)
        spatial_dims = (28, 28)
        x_train = jnp.array([x.resize(spatial_dims) for x, _ in ds_train])[..., None] / 255
        y_train = jnp.array([y for _, y in ds_train], dtype=jnp.int32)
        jnp.savez(mnist_path, x=x_train, y=y_train)


def _load_mnist(config: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load MNIST data, cached."""
    if "mnist" not in _mnist_cache:
        data_path = Path("data/mnist_train.npz")
        if data_path.exists():
            data = jnp.load(data_path)
            x_train = data["x"]
            y_train = data["y"]
        else:
            import torchvision
            spatial_dims = (28, 28)
            ds_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
            x_train = jnp.array([x.resize(spatial_dims) for x, _ in ds_train])[..., None] / 255
            y_train = jnp.array([y for _, y in ds_train], dtype=jnp.int32)
        _mnist_cache["mnist"] = (x_train, y_train)
    return _mnist_cache["mnist"]


def _compute_target(x: jnp.ndarray, y_integer: int) -> jnp.ndarray:
    """Compute per-pixel classification target.

    For digit pixels (x >= 0.1), target is one-hot encoding of the digit class.
    For background pixels, target is all zeros.
    """
    mask = x >= 0.1
    return jnp.where(mask, jax.nn.one_hot(y_integer, 10), 0.0)


def sample_state(config: dict, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a state with a random MNIST digit and its classification target.

    Returns:
        state: Shape (28, 28, channel_size) with MNIST image in last channel
        target: Shape (28, 28, 10) with one-hot classification for digit pixels
    """
    x_train, y_train = _load_mnist(config)

    x_idx = jax.random.choice(key, x_train.shape[0])
    x = x_train[x_idx]
    y_integer = y_train[x_idx]

    channel_size = config["nca"]["channel_size"]
    spatial_dims = x.shape[:2]

    state = jnp.zeros((*spatial_dims, channel_size))
    state = state.at[..., -1:].set(x)

    target = _compute_target(x, y_integer)

    return state, target
