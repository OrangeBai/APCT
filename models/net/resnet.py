from models.blocks import *


class ResNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.in_channels = 64
        self.layers = []

        if args.net == 'resnet18':
            self.set_up(BasicBlock, [2, 2, 2, 2], args.num_cls)
        elif args.net == 'resnet34':
            self.set_up(BasicBlock, [3, 4, 6, 3], args.num_cls)
        elif args.net == 'resnet50':
            self.set_up(BasicBlock, [3, 4, 14, 3], args.num_cls)
        elif args.net == 'resnet101':
            self.set_up(BasicBlock, [3, 4, 23, 3], args.num_cls)
        elif args.net == 'resnet152':
            self.set_up(BasicBlock, [3, 4, 36, 3], args.num_cls)
        else:
            raise NameError()

    def set_up(self, block, num_block, num_classes):

        setattr(self, 'conv1', nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)))

        setattr(self, 'conv2_x', self._make_layer(block, 64, num_block[0], 1))
        setattr(self, 'conv3_x', self._make_layer(block, 128, num_block[1], 2))
        setattr(self, 'conv4_x', self._make_layer(block, 256, num_block[2], 2))
        setattr(self, 'conv5_x', self._make_layer(block, 512, num_block[3], 2))
        # we use a different input size than the original paper
        # so conv2_x's stride is 1

        setattr(self, 'avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        setattr(self, 'fc', nn.Linear(512 * block.expansion, num_classes))

        self.layers = [self.conv1, self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didn't mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottleneck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
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
