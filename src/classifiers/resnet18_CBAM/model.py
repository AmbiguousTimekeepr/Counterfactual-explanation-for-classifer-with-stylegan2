"""ResNet18 + CBAM classifier model definition."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAMBlock(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = x * self.ca(x)
        return out * self.sa(out)


class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1
        self.cbam1 = CBAMBlock(64)
        self.layer2 = base_model.layer2
        self.cbam2 = CBAMBlock(128)
        self.layer3 = base_model.layer3
        self.cbam3 = CBAMBlock(256)
        self.layer4 = base_model.layer4
        self.cbam4 = CBAMBlock(512)
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.stem(x)
        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


__all__ = ["ResNet18_CBAM", "CBAMBlock", "ChannelAttention", "SpatialAttention"]
