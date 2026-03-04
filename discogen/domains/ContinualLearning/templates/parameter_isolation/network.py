"""Parameter isolation backend: multi-head classifier (per-task heads)."""

from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultiHeadResNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        base = resnet18(weights=None)
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = base.fc.in_features
        self.heads: Dict[int, nn.Linear] = {}
        self.active_task = 0
        self._max_classes = 0

    def set_active_head(self, task_id: int) -> None:
        self.active_task = int(task_id)
        if self.active_task not in self.heads:
            self.heads[self.active_task] = nn.Linear(self.feature_dim, max(1, self._max_classes))

    def ensure_num_classes(self, num_classes: int) -> None:
        # Grow all heads to new size
        if num_classes <= self._max_classes:
            return
        old = self._max_classes
        self._max_classes = num_classes
        for k, head in list(self.heads.items()):
            new = nn.Linear(self.feature_dim, self._max_classes)
            if old > 0:
                with torch.no_grad():
                    new.weight[:old] = head.weight
                    new.bias[:old] = head.bias
            self.heads[k] = new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.flatten(1)
        head = self.heads.get(self.active_task)
        if head is None:
            self.set_active_head(self.active_task)
            head = self.heads[self.active_task]
        return head(feats)


def build_model():
    return MultiHeadResNet(in_channels=3)
