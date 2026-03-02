from typing import Dict, Any
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with the given configuration.
        Args:
            config (dict): Configuration object containing model parameters.
                - input_dim (int): Number of channels/features in the input tensor (usually SENSORS_SPEECH_MASK)
                - model_dim (int): Dimensionality for the intermediate model representation.
                - kernel_size (int): Size of the convolutional kernel.
                - padding (int): Padding size for the convolutional layer.
                - dropout_rate (float): Dropout probability applied after convolutional and LSTM layers.
                - lstm_layers (int): Number of layers in the LSTM module.
                - bi_directional (bool): If True, uses a bidirectional LSTM; otherwise, a unidirectional LSTM.
                - batch_norm (bool): Indicates whether to use batch normalization.

        """
        super(Model, self).__init__()
        """Fill in your network architecture here."""


    def forward(self, x):
        """Fill in your forward pass here."""
        ...
        return x
