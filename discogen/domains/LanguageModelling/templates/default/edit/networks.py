from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 50304

    # possibly add other hyperparameters


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize your model here.

    def forward(self, idx):
        """
        Forward pass of the model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
                where each element is an integer in the range [0, vocab_size-1]

        Returns:
            torch.Tensor: Logits of shape (batch_size, sequence_length, vocab_size)
                representing the model's predictions for each token position
        """

        # Implement forward pass of the model here.
        logits = ...

        return logits

    def get_config(self):
        return self.config
