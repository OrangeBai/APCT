import torch
import torch.nn as nn
from models.blocks import *
from models.base_model import BaseModel


class DNN(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        set_up_kwargs = {'batch_norm': 1 if args.batch_norm else 0,
                         'activation': args.activation}
        self.layers = self.set_up(**set_up_kwargs)

    def parse_layer_args(self):
        pass

    def set_up(self, **kwargs):
        layers = []
        layers += [nn.Flatten()]
        layers += [LinearBlock(self.args.input_size, self.args.width, **kwargs)]

        for i in range(self.args.depth - 1):
            layers += [LinearBlock(self.args.width, self.args.width, **kwargs)]

        layers += [LinearBlock(self.args.width, self.args.num_cls, batch_norm=kwargs['batch_norm'], activation=None)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
