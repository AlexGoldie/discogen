import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Initialize the positional embedding module.

        Args:
            dim (int): The dimensionality of the positional embedding vector.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embeddings for the given positions.

        Args:
            x (torch.Tensor): A 1D tensor of shape `(N,)` containing scalar
                position values (e.g., time steps or token indices). The
                tensor's device and dtype are used to construct the output
                embeddings.

        Returns:
            torch.Tensor: A tensor of shape `(N, dim)` containing the
                sinusoidal positional embeddings corresponding to each
                input position.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
