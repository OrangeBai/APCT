from models.blocks import *
from models.base_model import BaseModel

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

        if args.net.lower() == 'vgg11':
            cfg = cfgs['vgg11']
        elif args.net.lower() == 'vgg13':
            cfg = cfgs['vgg13']
        elif args.net.lower() == 'vgg16':
            cfg = cfgs['vgg16']
        elif args.net.lower() == 'vgg19':
            cfg = cfgs['vgg19']
        else:
            raise NameError("No network named {}".format(args.net))

        if args.config is not None:
            cfg = args.config

        self.set_up(cfg, args.model_type)

    def set_up(self, cfg, model_type):

        layers = []
        num_pooling = 0
        pre_filters = 3
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                num_pooling += 1
                if num_pooling == 5:
                    if model_type == 'mini':
                        layers += [nn.AdaptiveAvgPool2d((1, 1))]
                    else:
                        layers += [nn.AdaptiveAvgPool2d((7, 7))]
                    layers += [nn.Flatten()]
            else:
                if num_pooling < 5:
                    layers += [ConvBlock(pre_filters, layer, kernel_size=(3, 3), padding=1, **self.set_up_kwargs)]
                    pre_filters = layer
                else:
                    if layer is not None:
                        layers += [LinearBlock(pre_filters, layer, **self.set_up_kwargs)]
                        pre_filters = layer
                    else:
                        layers += [LinearBlock(pre_filters, self.num_cls, batch_norm=1, activation=None)]
        setattr(self, 'layers', nn.Sequential(*layers))

    def forward(self, x):

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


class VGG16(VGG):
    def __init__(self, args):
        super().__init__(args)


class VGG19(VGG):
    def __init__(self, args):
        super().__init__(args)
