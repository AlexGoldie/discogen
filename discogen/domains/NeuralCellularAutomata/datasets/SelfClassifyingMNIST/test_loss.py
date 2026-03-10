from config import config
import jax
import jax.numpy as jnp

def eval_loss(state, target):
    target_channels = target.shape[-1]

    pred_logits = state[..., :target_channels]
    return jnp.mean(
        jnp.sum(-target * jax.nn.log_softmax(pred_logits, axis=-1), axis=-1)
    )
