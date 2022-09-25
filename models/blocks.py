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
                              # bias=False
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


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, *args, **kwargs):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=stride, bias=False, **kwargs),
            ConvBlock(out_channels, out_channels * BasicBlock.expansion, stride=stride, bias=False,
                      batch_norm=kwargs['batch_norm'], activation=None),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                ConvBlock(in_channels, out_channels * BasicBlock.expansion, kernel_size=(1, 1),
                          bias=False, batch_norm=kwargs['batch_norm'], activation=None),
            )
        self.act = set_activation(kwargs['activation'])

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.residual_function = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=(1, 1), padding=0, **kwargs),
            ConvBlock(out_channels, out_channels, stride=stride, kernel_size=(3, 3), padding=1, **kwargs),
            ConvBlock(out_channels, out_channels * BottleNeck.expansion, kernel_size=(1, 1), padding=0,
                      batch_norm=kwargs['batch_norm'], activation=None),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                ConvBlock(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, padding=0,
                          batch_norm=kwargs['batch_norm'], activation=None)
            )
        self.act = set_activation(kwargs['activation'])

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))
