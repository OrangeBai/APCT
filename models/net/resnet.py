from models.base_model import BaseModel
from models.blocks import *


class Resnet(BaseModel):
    def __init__(self, block, num_block, args):
        super().__init__(args)
        self.num_cls = args.num_cls
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=(7, 7), padding=3, stride=2, **self.set_up_kwargs))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_cls)

    def _make_layer(self, block, out_channels, num_blocks, stride, **kwargs):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, **self.set_up_kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class Resnet18(Resnet):
    def __init__(self, args):
        super().__init__(BasicBlock, [2, 2, 2, 2], args)


class Resnet34(Resnet):
    def __init__(self, args):
        super().__init__(BasicBlock, [3, 4, 6, 3], args)


class Resnet50(Resnet):
    def __init__(self, args):
        super().__init__(BottleNeck, [3, 4, 6, 3], args)


class Resnet101(Resnet):
    def __init__(self, args):
        super().__init__(BottleNeck, [3, 4, 23, 3], args)


class Resnet152(Resnet):
    def __init__(self, args):
        super().__init__(BottleNeck, [3, 8, 36, 3], args)
