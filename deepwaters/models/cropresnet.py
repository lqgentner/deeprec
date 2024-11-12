import abc

import torch
import torch.nn as nn
from torch import Tensor

from deepwaters.models.base import BaseModel, scalar2matrix


class SingleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
    ):
        super(SingleConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CropResBlock(nn.Module):
    """Like ResNetV1, but use padding=0 instead of stride=2 to downsample identity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
    ):
        super(CropResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if padding != 1:
            # Change window size by cropping to the center
            self.crop = nn.ZeroPad2d(2 * (-1 + padding))
        else:
            self.crop = None

        if out_channels != in_channels:
            self.downsample = nn.Sequential(
                # Downsample identity with a 1x1 convolution
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.crop is not None:
            identity = self.crop(identity)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class CropResBlockNoNorm(nn.Module):
    """Like ResNetV1, but use padding=0 instead of stride=2 to downsample identity.
    Without batch norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
    ):
        super(CropResBlockNoNorm, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
        )
        self.relu = nn.ReLU(inplace=True)

        if padding != 1:
            # Crop identity by removing outer pixels
            self.crop = nn.ZeroPad2d(2 * (-1 + padding))
        else:
            self.crop = None

        if out_channels != in_channels:
            # Change identity channels with a 1x1 convolution
            self.downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
            )

        else:
            self.downsample = None

    def forward(self, x: Tensor):
        identity = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)

        if self.crop is not None:
            identity = self.crop(identity)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class CropResBlockDropout(CropResBlock):
    """Like ResNetV1, but use padding=0 instead of stride=2 to downsample identity.
    Additionally, apply dropout at the end of the residual path"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
        dropout_p: float = 0.2,
    ):
        super(CropResBlockDropout, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            groups=groups,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor):
        x = super(CropResBlockDropout, self).forward(x)
        x = self.dropout(x)

        return x


class CropNullBlock(nn.Module):
    """ResNet without skip connections and without batch norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
    ):
        super(CropNullBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x


class ResNetMixin(nn.Module):
    @abc.abstractmethod
    def forward(self, x) -> None:
        pass

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CropResNet(BaseModel, ResNetMixin):
    def __init__(self, n_matrix_inps: int, n_scalar_inps: int, **kwargs) -> None:
        super(CropResNet, self).__init__(
            n_matrix_inps=n_matrix_inps, n_scalar_inps=n_scalar_inps, **kwargs
        )

        self.matrix_layers = nn.Sequential(
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
        )

        n_combin_inps = n_matrix_inps + n_scalar_inps

        self.common_layers = nn.Sequential(
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
            SingleConv2d(n_combin_inps, n_combin_inps),
        )

        self.fc = nn.Linear(n_combin_inps, 1)

        self._init_weights()

    def forward(self, *_, **inputs: Tensor) -> Tensor:
        # Extract inputs
        xm = inputs["matrix_inputs"]
        xs = inputs["scalar_inputs"]

        xm = self.matrix_layers(xm)
        xs = scalar2matrix(xs, out_height=xm.shape[2])
        x = torch.cat([xm, xs], dim=1)
        x = self.common_layers(x).squeeze()
        x = self.fc(x).squeeze()
        return x


class CropResNetPool(BaseModel, ResNetMixin):
    def __init__(self, n_matrix_inps: int, n_scalar_inps: int, **kwargs) -> None:
        super(CropResNetPool, self).__init__(
            n_matrix_inps=n_matrix_inps, n_scalar_inps=n_scalar_inps, **kwargs
        )

        self.matrix_layers = nn.Sequential(
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
            CropResBlock(n_matrix_inps, n_matrix_inps),
        )

        n_combin_inps = n_matrix_inps + n_scalar_inps

        self.common_layers = nn.Sequential(
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
            CropResBlock(n_combin_inps, n_combin_inps),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_combin_inps, 1)

        self._init_weights()

    def forward(self, *_, **inputs: Tensor) -> Tensor:
        # Extract inputs
        xm = inputs["matrix_inputs"]
        xs = inputs["scalar_inputs"]

        xm = self.matrix_layers(xm)
        xs = scalar2matrix(xs, out_height=xm.shape[2])
        x = torch.cat([xm, xs], dim=1)
        x = self.common_layers(x)
        x = self.pool(x).squeeze()
        x = self.fc(x).squeeze()
        return x
