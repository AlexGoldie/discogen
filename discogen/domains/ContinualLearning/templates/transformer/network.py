"""Transformer backend: ViT-tiny via timm with dynamic classifier head."""

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


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


class CLViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.head = DynamicClassifier(self.feature_dim)

    def ensure_num_classes(self, num_classes: int) -> None:
        self.head.ensure_num_classes(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self.backbone(x)
        return self.head(feats)


def build_model():
    return CLViT()
