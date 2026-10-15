from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SpectralResidualUnit(nn.Module):

    def __init__(self, channels: int, inter_channels: int, groups: int):
        super().__init__()
        self.group_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.mix_path = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.group_path(x) + self.mix_path(x), inplace=True)


class SPRNStyleRawHead(nn.Module):
    """Grouped spectral residual classifier inspired by SPRN."""

    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        groups: int = 11,
        group_width: int = 37,
        fc_planes: int = 128,
        inter_channels: int = 32,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.groups = int(groups)
        self.new_ch = int(math.ceil(in_ch / groups) * groups)
        self.group_width = int(group_width)
        stem_ch = self.groups * self.group_width
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.new_ch, stem_ch, kernel_size=1, groups=self.groups, bias=False
            ),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.res0 = SpectralResidualUnit(stem_ch, inter_channels, groups=self.groups)
        self.res1 = SpectralResidualUnit(stem_ch, inter_channels, groups=self.groups)
        self.project = nn.Sequential(
            nn.Conv2d(stem_ch, fc_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(fc_planes),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(fc_planes, num_classes)

    def _pad_bands(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.new_ch - x.shape[1]
        if pad <= 0:
            return x
        tail = x[:, -1:, :, :].expand(-1, pad, -1, -1)
        return torch.cat([x, tail], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_bands(x)
        x = self.stem(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.project(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
