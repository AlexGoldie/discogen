from typing import Any, Callable

import flax
import jax.numpy as jnp


def loss_actor_and_critic(params, init_hstate, traj_batch, gae, targets, network, config):
    # Inputs:
    # - params: the model parameters.
    # - init_hstate: a hidden state for the recurrent module.
    # - traj_batch: the various data collected from the environment. It is a Transition object.
    # - gae: the generalized advantage estimate.
    # - targets: TD targets.
    # - network: the actor-critic network from `network.py`.
    # - config: the config, defined in `config.py`, which provides some hyperparameters.

    # Estimate the current value and get the current policy from the actor critic architecture
    pi, value_pred = network.apply(params, init_hstate, (traj_batch.obs, traj_batch.done))

    """Fill in your loss logic here."""

    # Your function must return a loss from which we can calculate current gradients.
    return total_loss, (aux1, aux2, ...)
