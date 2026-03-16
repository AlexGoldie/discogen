from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    return ...


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    tanh_squash: bool = True
    const_std: bool = True
    final_fc_init_scale: float = 0.01

    @nn.compact
    def __call__(self, observations, temperature=1.0):
        # Input = observations. observations is the input of shape (batch, obs_dim).
        # temperature scales the standard deviation of the output distribution.

        # The actor should output an action distribution

        """Fill in your actor network logic here."""

        # Your function must return a distribution object with .mode() and .sample() methods.
        distribution = distrax.MultivariateNormalDiag()
        return distribution


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2

    @nn.compact
    def __call__(self, observations, actions=None):
        # Inputs:
        # - observations: Input observations of shape (batch, obs_dim).
        # - actions: Actions of shape (batch, action_dim).

        # The critic computes Q(s, a) values. Should support ensemble of critics

        """Fill in your critic network logic here."""

        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        # use the value net to compute the Q-values
        v = ...

        return v
