from typing import Type, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.blocks import conv1x1, BasicBlock, Bottleneck, ConvBlock
from models.base_model import BaseModel


class ResNet(BaseModel):

    def __init__(
            self,
            args,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__(args)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = ConvBlock(3, self.inplanes, kernel_size=7, stride=2, padding=3, bn=1, act='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, args.num_cls)
        self.layers = [self.conv1, self.maxpool, *list(self.layer1), *list(self.layer2), *list(self.layer3),
                       *list(self.layer4), self.avgpool, self.flatten, self.fc]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.norm_layer(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(args):
    return ResNet(args, BasicBlock, [2, 2, 2, 2])


def resnet34(args):
    return ResNet(args, BasicBlock, [3, 4, 6, 3])


def resnet50(args):
    return ResNet(args, Bottleneck, [3, 4, 6, 3])


def resnet101(args):
    return ResNet(args, Bottleneck, [3, 4, 23, 3])


def resnet152(args):
    return ResNet(args, Bottleneck, [3, 8, 36, 3], args)
