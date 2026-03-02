"""Default backend network: ResNet-18 with dynamic classifier head."""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18


class DynamicClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = 0
        self.classifier = nn.Identity()

    def ensure_num_classes(self, num_classes: int) -> None:
        if num_classes <= self.out_features:
            return
        new = nn.Linear(self.in_features, num_classes)
        if self.out_features > 0:
            with torch.no_grad():
                new.weight[: self.out_features] = self.classifier.weight
                new.bias[: self.out_features] = self.classifier.bias
        self.classifier = new
        self.out_features = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CLResNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        base = resnet18(weights=None)
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # exclude fc
        self.feature_dim = base.fc.in_features
        self.head = DynamicClassifier(self.feature_dim)

    def ensure_num_classes(self, num_classes: int) -> None:
        self.head.ensure_num_classes(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.flatten(1)
        return self.head(feats)


def build_model():
    return CLResNet(in_channels=3)
