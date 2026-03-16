from config import config
import jax
import jax.numpy as jnp

def eval_loss(state, target):
    output_ch = config["matrix"]["output_channel"]
    pred = state[..., output_ch : output_ch + 1]
    return jnp.mean(jnp.square(pred - target))
