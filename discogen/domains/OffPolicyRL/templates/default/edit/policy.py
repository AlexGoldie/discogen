import jax.numpy as jnp
import jax

def explore(rng, q_vals, t, config):
    """
    Fill in your exploration logic here. (used during training)
    Inputs:
    - rng: the random key.
    - q_vals: the Q-values (shape: (batch_size, ...)).
    - t: the timestep.
    - config: the config, defined in `config.py`, which provides some hyperparameters.

    Returns:
    - actions: the actions to take (shape: (batch_size,)).
    """
    actions = ... # fill in your exploration logic here
    return actions # shape (batch_size)

def exploit(rng, q_vals, t, config):
    """
    Fill in your exploitation logic here. (used during testing)
    Inputs:
    - rng: the random key.
    - q_vals: the Q-values (shape: (batch_size, ...)).
    - t: the timestep.
    - config: the config, defined in `config.py`, which provides some hyperparameters.

    Returns:
    - actions: the actions to take (shape: (batch_size,)).
    """

    actions = ... # fill in your exploitation logic here
    return actions # shape (batch_size)
