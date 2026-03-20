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
        emb = ...
        return emb
