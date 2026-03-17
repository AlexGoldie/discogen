from typing import Sequence, Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class RecurrentModule(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )

    @nn.compact
    def __call__(self, carry, x):
        """Insert your recurrent logic here."""
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Add initialize_carry function.
        return initialized_carry

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: dict
    activation: Callable

    @nn.compact
    def __call__(self, hidden, x):
        """Insert your network logic here. You should call your recurrent module."""
        # Input = x. x is the environment observation.
        obs, dones = x

        # You can use self.activation(x) here.

        # Some environments have continuous action spaces, and some have discrete action spaces. This is denoted by config["CONTINUOUS"}]
        if self.config.get("CONTINUOUS", False):
            pi = ...
        else:
            pi = ...

        # You must somehow produce pi and v.
        return hidden, pi, jnp.squeeze(critic, axis=-1)
