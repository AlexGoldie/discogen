import jax.numpy as jnp

def q_loss_fn(params, target_params, network, learn_batch, config):
    """
    Fill in your Q-loss function here.
    Inputs:
    - params: the parameters of the Q-network.
    - target_params: the target parameters of the Q-network.
    - learn_batch: batch containing:
        -- first: (obs, action, reward, done)
        -- second: (obs, action)
    - config: the config, defined in `config.py`, which provides some hyperparameters.

    Returns:
    - loss: the loss value.
    """


    return loss
