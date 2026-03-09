"""Dataset utilities for MNISTInpainting task."""

from pathlib import Path

import jax
import jax.numpy as jnp

_mnist_cache = {}

DEFAULT_MASK_RATIO = 0.5


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


def sample_state(config: dict, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a state with a randomly masked MNIST digit.

    Returns:
        state: Shape (28, 28, channel_size) with masked MNIST image in last channel
        target: Shape (28, 28, 1) with the full (unmasked) MNIST image
    """
    x_train, _ = _load_mnist(config)
    mask_ratio = config.get("mask_ratio", DEFAULT_MASK_RATIO)

    sample_key, mask_key = jax.random.split(key)
    x_idx = jax.random.choice(sample_key, x_train.shape[0])
    x = x_train[x_idx]

    spatial_dims = x.shape[:2]
    channel_size = config["nca"]["channel_size"]

    mask = jax.random.bernoulli(mask_key, 1 - mask_ratio, shape=spatial_dims)
    mask = jnp.expand_dims(mask, axis=-1)
    x_masked = x * mask

    state = jnp.zeros((*spatial_dims, channel_size))
    state = state.at[..., -1:].set(x_masked)

    target = x

    return state, target
