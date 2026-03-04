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
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config['input_dim'],
            out_channels=config['model_dim'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
        )
        self.lstm_layers = config['lstm_layers']
        self.batch_norm = (
            nn.BatchNorm1d(num_features=config['model_dim'])
            if config['batch_norm']
            else nn.Identity()
        )
        self.conv_dropout = nn.Dropout(p=config['dropout_rate'])
        self.lstm = nn.LSTM(
            input_size=config['model_dim'],
            hidden_size=config['model_dim'],
            num_layers=self.lstm_layers,
            dropout=config['dropout_rate'],
            batch_first=True,
            bidirectional=config['bi_directional'],
        )
        self.lstm_dropout = nn.Dropout(p=config['dropout_rate'])
        self.speech_classifier = nn.Linear(config['model_dim'], 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.conv_dropout(x)
        # LSTM expects (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x.permute(0, 2, 1))
        last_layer_h_n = h_n
        if self.lstm_layers > 1:
            # handle more than one layer
            last_layer_h_n = h_n[-1, :, :]
            last_layer_h_n = last_layer_h_n.unsqueeze(0)
        output = self.lstm_dropout(last_layer_h_n)
        output = output.flatten(start_dim=0, end_dim=1)
        x = self.speech_classifier(output)
        return x
