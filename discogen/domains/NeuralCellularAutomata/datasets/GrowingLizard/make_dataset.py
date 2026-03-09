"""Dataset utilities for GrowingLizard task."""

import io
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import PIL.Image

DEFAULT_EMOJI = "\U0001F98E"  # Lizard emoji

# Cache for loaded target
_target_cache = {}


def get_image_from_url(url: str) -> PIL.Image.Image:
    """Fetch an image from a URL."""
    with urlopen(url) as response:
        image_data = response.read()
    return PIL.Image.open(io.BytesIO(image_data))


def get_emoji(emoji: str) -> PIL.Image.Image:
    """Fetch an emoji as a PIL Image from Google's Noto Emoji repository."""
    code = hex(ord(emoji))[2:].lower()
    url = f"https://raw.githubusercontent.com/googlefonts/noto-emoji/refs/heads/main/png/128/emoji_u{code}.png"
    return get_image_from_url(url)


def download_dataset(cache_dir: Path) -> None:
    """Download the emoji image to the cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    emoji_path = cache_dir / "target.png"

    if not emoji_path.exists():
        print(f"Downloading emoji {DEFAULT_EMOJI}...")
        img = get_emoji(DEFAULT_EMOJI)
        img.save(emoji_path)
        print(f"Saved to {emoji_path}")


def _load_target(config: dict) -> jnp.ndarray:
    """Load and cache the target image."""
    cache_key = (config["target_size"], config["pad_width"])

    if cache_key not in _target_cache:
        target_size = config["target_size"]
        pad_width = config["pad_width"]

        # Try to load from data directory first
        data_path = Path("data/target.png")
        if not data_path.exists():
            print("Target not found in data/, downloading...")
            img = get_emoji(DEFAULT_EMOJI)
        else:
            img = PIL.Image.open(data_path)

        img = img.convert("RGBA")
        img = img.resize((target_size, target_size), resample=PIL.Image.Resampling.LANCZOS)
        target = jnp.array(img, dtype=jnp.float32) / 255.0
        target = jnp.pad(
            target,
            ((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        _target_cache[cache_key] = target

    return _target_cache[cache_key]


def sample_state(config: dict, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create an initial seed state and load the target.

    For growing tasks, the initial state is a single alive cell in the center,
    and the target is the same for all samples (the emoji image).

    Args:
        config: Configuration dictionary.
        key: JAX PRNG key (unused for this task, but required for interface).

    Returns:
        Tuple of (initial_state, target).
    """
    target = _load_target(config["target"])
    spatial_dims = target.shape[:2]
    channel_size = config["nca"]["channel_size"]

    # Create seed state: all zeros with center cell alive
    state = jnp.zeros((*spatial_dims, channel_size))
    center_h = spatial_dims[0] // 2
    center_w = spatial_dims[1] // 2
    state = state.at[center_h, center_w, -1].set(1.0)

    return state, target
