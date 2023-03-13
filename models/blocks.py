import torch
import torch.nn as nn
from core.utils import set_activation, set_gamma
from models.net.resnet import Bottleneck

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, act='relu'):
        super().__init__()

        self.LT = nn.Linear(in_channels, out_channels)
        self.BN = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.Act = set_activation(act)

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bn=True, act='relu',
                 **kwargs):
        super().__init__()
        self.LT = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, stride=stride, bias=False)
        self.BN = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.Act = set_activation(act)

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)

    def forward(self, x: torch.tensor):
        device = x.device
        (batch_size, num_channels, height, width) = x.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        std = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (x - means.to(device=device)) / std.to(device=device)


def check_block(model, block):
    return check_valid_block(block) and not check_last_block(model, block)


def check_last_block(model, block):
    return block == model.layers[-1]


def check_valid_block(block):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock) or isinstance(block, Bottleneck):
        return True
    return False