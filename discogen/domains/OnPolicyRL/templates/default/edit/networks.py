from typing import Sequence, Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: dict
    activation: Callable

    @nn.compact
    def __call__(self, x):
        """Insert your network logic here."""
        # Input = x. x is the environment observation.

        # You can use self.activation(x) here.

        # Some environments have continuous action spaces, and some have discrete action spaces. This is denoted by config["CONTINUOUS"]
        if self.config.get("CONTINUOUS", False):
            pi = ...
        else:
            pi = ...

        # You must somehow produce pi and v.
        return pi, jnp.squeeze(v, axis=-1)
