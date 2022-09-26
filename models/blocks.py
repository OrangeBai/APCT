from torch.nn import functional as F

from core.utils import *


def set_activation(activation):
    if activation is None:
        return nn.Identity()
    elif activation.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif activation.lower() == 'prelu':
        return nn.PReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(0.1, inplace=True)


def set_bn(batch_norm, dim, channel):
    if not batch_norm:
        return nn.Identity()
    else:
        if dim == 1:
            return nn.BatchNorm1d(channel)
        else:
            return nn.BatchNorm2d(channel)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.FC = nn.Linear(in_channels, out_channels)
        self.BN = set_bn(kwargs['batch_norm'], dim=1, channel=out_channels)
        self.Act = set_activation(kwargs['activation'])

    def forward(self, x):
        x = self.FC(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, *args, **kwargs):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride,
                              bias=False
                              )
        self.BN = set_bn(kwargs['batch_norm'], 2, out_channels)
        self.Act = set_activation(kwargs['activation'])

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class FloatConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x, mask):
        x = self.conv(x)
        x[mask] = 0
        return x

