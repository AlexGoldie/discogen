from config import config
import jax
import jax.numpy as jnp

def eval_loss(state, target):
    target_channels = target.shape[-1]

    pred = state[..., -target_channels:]
    return jnp.mean(jnp.square(pred - target))
