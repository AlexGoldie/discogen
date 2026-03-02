from typing import Dict, List, Optional

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, config: Dict[str, int | bool | List[int]]):
        """Initialize the model with the given configuration.
        Args:
            config (dict): Configuration dictionary containing model parameters.
                - num_classes (int): Number of output classes.
                - num_channels (int): Number of input channels.
                - channels (list): List of channel sizes for each stage.
                - blocks_per_stage (int): Number of blocks per stage.
                - use_residual (bool): Whether to use residual connections.
                - dropout (float): Dropout rate before the classifier.
        """
        super(Model, self).__init__()
        """Fill in your network architecture here."""

    def forward(self, pixel_values, labels=None):
        del labels
        """Fill in your forward pass here."""

        return {{"logits": logits}}
