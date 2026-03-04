import flax.linen as nn
import jax.numpy as jnp

class QNetwork(nn.Module):
    action_dim: int
    width: int
    depth: int
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width)(x)
            x = nn.relu(x)
        q = nn.Dense(self.action_dim)(x)
        return q
