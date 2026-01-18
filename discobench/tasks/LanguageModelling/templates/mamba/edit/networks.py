from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    d_model: int = 768
    n_layer: int = 12
    # Add other SSM-specific hyperparameters as needed (d_state, d_conv, expand, etc.)


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize your Mamba/SSM model here.
        # Typical components:
        # - self.embedding: Token embeddings
        # - self.layers: Stack of Mamba/SSM blocks
        # - self.norm_f: Final normalization layer
        # - self.lm_head: Output projection to vocab_size

    def forward(self, idx):
        """
        Forward pass of the Mamba model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
                where each element is an integer in the range [0, vocab_size-1]

        Returns:
            torch.Tensor: Logits of shape (batch_size, sequence_length, vocab_size)
                representing the model's predictions for each token position
        """

        # Implement forward pass of the model here.
        # 1. Embed tokens: x = self.embedding(idx)
        # 2. Pass through SSM layers
        # 3. Apply final normalization
        # 4. Project to vocabulary: logits = self.lm_head(x)
        logits = ...

        return logits

    def get_config(self):
        return self.config
