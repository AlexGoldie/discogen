from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    activation: callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # Self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=orthogonal(np.sqrt(2)),
        )(y, y)
        x = x + y

        # Feedforward
        y = nn.LayerNorm()(x)
        y = nn.Dense(
            int(self.embed_dim * self.mlp_ratio), kernel_init=orthogonal(np.sqrt(2))
        )(y)
        y = self.activation(y)
        y = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(y)
        x = x + y
        return x


class ActorCritic(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, x):
        # Pull activation from config
        activation_name = self.config.get("ACTIVATION", "tanh")
        if activation_name == "relu":
            activation_fn = nn.relu
        elif activation_name == "gelu":
            activation_fn = nn.gelu
        else:
            activation_fn = nn.tanh

        # Use HSIZE as the transformer embedding dim
        embed_dim = self.config.get("HSIZE", 64)
        continuous = self.config.get("CONTINUOUS", False)

        num_layers = 2
        num_heads = 4

        if x.ndim == 1:
            # Add batch dimension
            x = x[None, ...]  # (1, obs_dim)
        if x.ndim == 2:
            # Shape (batch, features) → add seq dim
            x = x[:, None, :]  # (batch, seq=1, features)

        # Create multiple "views" of the observation
        num_patches = self.config.get("NUM_PATCHES", 8)

        # Replicate and project
        x_expanded = jnp.tile(
            x[:, None, :], (1, num_patches, 1)
        )  # (B, num_patches, obs_dim)
        # Add positional embeddings (this is key!)
        x = nn.Dense(
            embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        pos_embed = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, embed_dim),
        )
        x = x + pos_embed
        # Transformer stack
        for _ in range(num_layers):
            x = TransformerEncoderBlock(embed_dim, num_heads, activation=activation_fn)(
                x
            )

        # Pool sequence dimension (mean)
        x = x.mean(axis=1)

        actor_mean = nn.LayerNorm()(x)
        actor_mean = nn.Dense(embed_dim, kernel_init=orthogonal(np.sqrt(2)))(actor_mean)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if continuous:
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.LayerNorm()(x)
        critic = nn.Dense(embed_dim, kernel_init=orthogonal(np.sqrt(2)))(critic)
        critic = activation_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
