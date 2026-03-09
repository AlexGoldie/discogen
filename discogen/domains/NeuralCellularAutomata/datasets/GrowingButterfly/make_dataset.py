"""Dataset utilities for GrowingButterfly task."""

import io
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import PIL.Image

DEFAULT_EMOJI = "\U0001F98B"  # Butterfly emoji

_target_cache = {}


def get_image_from_url(url: str) -> PIL.Image.Image:
    with urlopen(url) as response:
        image_data = response.read()
    return PIL.Image.open(io.BytesIO(image_data))


def get_emoji(emoji: str) -> PIL.Image.Image:
    code = hex(ord(emoji))[2:].lower()
    url = f"https://raw.githubusercontent.com/googlefonts/noto-emoji/refs/heads/main/png/128/emoji_u{code}.png"
    return get_image_from_url(url)


def download_dataset(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emoji_path = cache_dir / "target.png"
    if not emoji_path.exists():
        print(f"Downloading emoji {DEFAULT_EMOJI}...")
        img = get_emoji(DEFAULT_EMOJI)
        img.save(emoji_path)


def _load_target(config: dict) -> jnp.ndarray:
    cache_key = (config["target_size"], config["pad_width"])
    if cache_key not in _target_cache:
        target_size = config["target_size"]
        pad_width = config["pad_width"]
        data_path = Path("data/target.png")
        if not data_path.exists():
            img = get_emoji(DEFAULT_EMOJI)
        else:
            img = PIL.Image.open(data_path)
        img = img.convert("RGBA")
        img = img.resize((target_size, target_size), resample=PIL.Image.Resampling.LANCZOS)
        target = jnp.array(img, dtype=jnp.float32) / 255.0
        target = jnp.pad(target, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
        _target_cache[cache_key] = target
    return _target_cache[cache_key]


def sample_state(config: dict, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    target = _load_target(config["target"])
    spatial_dims = target.shape[:2]
    channel_size = config["nca"]["channel_size"]
    state = jnp.zeros((*spatial_dims, channel_size))
    center_h, center_w = spatial_dims[0] // 2, spatial_dims[1] // 2
    state = state.at[center_h, center_w, -1].set(1.0)
    return state, target
