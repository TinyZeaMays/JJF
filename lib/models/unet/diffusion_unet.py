from typing import Tuple

from torch import nn, Tensor

from lib.models.misc import WeightStandardizedConv2d


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: Tuple[float, float] = None) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
