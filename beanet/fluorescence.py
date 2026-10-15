from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out = channel // reduction
        if out == 0:
            out = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, out, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SSMF(nn.Module):
    """Fuse spatial and multiscale convolution features with a residual path."""

    def __init__(self, input_channels, reduction_N=16):
        super(SSMF, self).__init__()
        self.point_wise = nn.Conv2d(
            input_channels, reduction_N, kernel_size=1, padding=0, bias=False
        )
        self.depth_wise = nn.Conv2d(
            reduction_N,
            reduction_N,
            kernel_size=3,
            padding=1,
            groups=reduction_N,
            bias=False,
        )
        self.conv3D_3 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            bias=False,
        )
        self.conv3D_5 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 5),
            padding=(0, 0, 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(reduction_N)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * reduction_N, 2 * reduction_N // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2 * reduction_N // 4, 2 * reduction_N, 1, bias=False),
            nn.Sigmoid(),
        )
        self.shortcut = nn.Sequential()
        if input_channels != 2 * reduction_N:
            self.shortcut = nn.Conv2d(
                input_channels, 2 * reduction_N, 1, padding=0, bias=False
            )
        self.frelu = nn.Sequential(
            nn.Conv2d(
                2 * reduction_N, 2 * reduction_N, 3, padding=1, groups=2 * reduction_N
            ),
            nn.BatchNorm2d(2 * reduction_N),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x_1 = self.point_wise(x)
        x_2 = self.depth_wise(x_1)
        x_1_3d = x_1.unsqueeze(1)
        x_3_3 = self.conv3D_3(x_1_3d).squeeze(1)
        x_3_5 = self.conv3D_5(x_1_3d).squeeze(1)
        x_3 = (x_3_3 + x_3_5) / 2
        x_concat = torch.cat((x_2, x_3), dim=1)
        attn_weight = self.attn(x_concat)
        x = x_concat * attn_weight
        x = x + residual
        x_frelu = self.frelu(x)
        x = torch.max(x, x_frelu)
        return x


class LightCBAM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.chan_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spa_attn = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        chan_weight = self.chan_attn(x)
        x = x * chan_weight
        spa_feat = torch.cat(
            [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1
        )
        spa_weight = self.spa_attn(spa_feat)
        x = x * spa_weight
        return x


class NNAD(nn.Module):
    """Depthwise activation block with channel and spatial attention."""

    def __init__(self, dim, act_num=1, dilation=1):
        super(NNAD, self).__init__()
        self.act_num = act_num
        self.dim = dim
        self.dilation = dilation
        kernel_size = act_num * 2 + 1
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel_size, kernel_size))
        self.bn = nn.BatchNorm2d(dim, eps=1e-05)
        self.act = nn.GELU()
        self.attn = LightCBAM(dim)
        nn.init.orthogonal_(self.weight, gain=1.0)

    def forward(self, x):
        residual = x
        x = self.act(x)
        padding = self.act_num * self.dilation
        x = F.conv2d(
            x, self.weight, padding=padding, groups=self.dim, dilation=self.dilation
        )
        x = self.bn(x)
        x = self.attn(x)
        x = x + residual
        return x


class FluorescenceBranch(nn.Module):

    def __init__(self, in_ch):
        super(FluorescenceBranch, self).__init__()
        self.ssmf = SSMF(in_ch)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = NNAD(32)
        self.conv1 = nn.Conv2d(32, 32, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = NNAD(32)
        self.se = SELayer(32)

    def forward(self, x):
        x = self.ssmf(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
