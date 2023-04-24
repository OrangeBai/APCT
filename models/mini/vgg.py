from ..base_model import BaseModel
from ..blocks import *

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 4096, 4096, None],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 4096, 4096, None],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 4096, 4096, None],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M',
              4096, 4096, None]
}


class VGG(BaseModel):

    def __init__(self, args):
        super().__init__(args)
        self.num_cls = args.num_cls
        self.args = args
        try:
            cfg = cfgs[args.net.lower()]
        except KeyError:
            raise NameError("No network named {}".format(args.net))
        self.set_up(cfg)

    def set_up(self, cfg):

        layers = []
        num_pooling = 0
        pre_filters = 3
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                num_pooling += 1
                if num_pooling == 5:
                    layers += [nn.AdaptiveAvgPool2d((1, 1))]
                    layers += [nn.Flatten()]
                    pre_filters = pre_filters * 1 * 1
            else:
                if num_pooling < 5:
                    layers += [ConvBlock(pre_filters, layer, kernel_size=(3, 3), padding=1,
                                         bn=self.args.batch_norm, act=self.args.activation)]
                    pre_filters = layer
                else:
                    if layer is not None:
                        layers += [LinearBlock(pre_filters, layer, bn=self.args.batch_norm,
                                               act=self.args.activation)]
                        pre_filters = layer
                    else:
                        layers += [LinearBlock(pre_filters, self.num_cls,
                                               bn=self.args.batch_norm, act=None)]
        setattr(self, 'layers', nn.Sequential(*layers))

    def forward(self, x):
        x = self.norm_layer(x)
        return self.layers(x)


def make_layers(cfg):
    layers = []

    input_channel = 3
    for layer in cfg:
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [ConvBlock(input_channel, layer, kernel_size=(3, 3), padding=1)]
        input_channel = layer

    return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, args):
        super().__init__(args)


class VGG13(VGG):
    def __init__(self, args):
        super().__init__(args)


def vgg16(args):
    return VGG(args)


def vgg13(args):
    return VGG(args)


def vgg11(args):
    return VGG(args)


def vgg19(args):
    return VGG(args)
