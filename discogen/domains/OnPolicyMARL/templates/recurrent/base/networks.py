import functools
from typing import Dict, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Callable

GRU_HIDDEN_DIM = 128
FC_DIM_SIZE = 128

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
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size=GRU_HIDDEN_DIM):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation: Callable

    @nn.compact
    def __call__(self, hidden, x):
        if self.config.get("GET_AVAIL_ACTIONS", False):
            obs, dones, avail_actions = x
        else:
            obs, dones = x
        embedding = nn.Dense(
            FC_DIM_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = self.activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = RecurrentModule()(hidden, rnn_in)

        actor_mean = nn.Dense(GRU_HIDDEN_DIM, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = self.activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        if self.config.get("GET_AVAIL_ACTIONS", False):
            unavail_actions = 1 - avail_actions
            action_logits = actor_mean - (unavail_actions * 1e10)
        else:
            action_logits = actor_mean

        if self.config.get("CONTINUOUS", False):
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(FC_DIM_SIZE, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = self.activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)
