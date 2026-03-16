from typing import Optional, Dict, List
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.proj = None
        if self.use_residual and in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            out = out + identity

        out = self.relu(out)
        return out


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
        super().__init__()

        num_classes = config["num_classes"]
        in_channels = config["num_channels"]
        channels = config["channels"]
        blocks_per_stage = config["blocks_per_stage"]
        use_residual = config["use_residual"]
        dropout = config["dropout"]

        stages = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            blocks = []
            for _ in range(blocks_per_stage):
                blocks.append(BasicBlock(c_in, c_out, use_residual=use_residual))
                c_in = c_out
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            if i < len(channels) - 1:
                stages.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, pixel_values, labels=None):
        del labels
        x = self.features(pixel_values)
        x = self.pool(x)
        logits = self.classifier(x)
        return {"logits": logits}
