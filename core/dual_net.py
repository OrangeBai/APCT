from models.blocks import *
from models import Bottleneck
import torch.nn.functional as F


class DualNet(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.gamma = set_gamma(args.activation)
        self.handles = []

    def compute_fixed_1batch(self, x):
        self.net.eval()
        fixed_neurons = []
        batch_x = self.net.norm_layer(x)
        for i, module in enumerate(list(self.net.layers)):
            batch_x = self.compute_pre_act(module, batch_x)
            if check_block(self.model, module):
                fixed = self._fixed_1batch(batch_x)
                fixed_neurons += [fixed]
                batch_x = module.Act(batch_x)
            else:
                fixed_neurons += [None]
        self.net.train()
        return fixed_neurons

    def compute_fixed_2batch(self, x1, x2):
        self.net.eval()
        fixed_neurons = []
        x1 = self.net.norm_layer(x1)
        x2 = self.net.norm_layer(x2)
        for i, module in enumerate(self.net.layers.children()):
            x1 = self.compute_pre_act(module, x1)
            x2 = self.compute_pre_act(module, x2)
            if check_block(self.net, module):
                fixed = self._fixed_2batch(x1, x2)
                fixed_neurons += [fixed]
                x1 = module.Act(x1)
                x2 = module.Act(x2)
            else:
                fixed_neurons += [None]
        self.net.train()
        return fixed_neurons

    def predict(self, x, fixed_neuron, eta_fixed, eta_float):
        self.net.eval()
        return self.forward(x, fixed_neuron, eta_fixed, eta_float, False)

    def forward(self, x, fixed_neurons, eta_fixed, eta_float, balance=True):
        batch_x = self.net.norm_layer(x)
        for i, module in enumerate(list(self.net.layers)):
            batch_x = self.compute_pre_act(module, batch_x)
            if check_block(self.net, module):
                h = self.set_hook(fixed_neurons[i], eta_fixed, eta_float, balance)
                self.handles += [module.Act.register_forward_pre_hook(h)]
                batch_x = module.Act(batch_x)
        self.remove_handles()
        return batch_x

    def remove_handles(self):
        """Remove all registered forward hook"""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return

    def set_hook(self, fixed, eta_fixed, eta_float, balance=False):
        def forward_pre_hook(m, inputs):
            x = inputs[0]
            return self.x_mask(x, eta_fixed, fixed, balance) + self.x_mask(x, eta_float, ~fixed, balance)

        return forward_pre_hook

    @staticmethod
    def x_mask(x, ratio, mask, balance=True):
        """masked forward for a given data x"""
        if ratio == 0:
            return x * mask
        else:
            if balance:
                return x * (1 + ratio) * mask - x.detach() * mask.detach() * ratio
            else:
                return x * (1 + ratio) * mask

    @staticmethod
    def compute_pre_act(module, x):
        """Compute the pre activation of a block"""
        if type(module) == ConvBlock or type(module) == LinearBlock:
            return module.BN(module.LT(x))
        elif type(module) == Bottleneck:
            out = module.bottle_net(x)
            identity = module.downsample(x)

            return out + identity
        else:
            return module(x)

    def _fixed_2batch(self, x_1, x_2):
        if len(self.gamma) == 1:
            return (x_1 - self.gamma[0]) * (x_2 - self.gamma[0]) > 0
        else:
            stacked = torch.stack([(x_1 - g) * (x_2 - g) > 0 for g in self.gamma])
            return torch.all(stacked, dim=0)

    def _fixed_1batch(self, batch_x):
        dims = len(batch_x.shape)
        if len(self.gamma) == 1:
            x_0_pattern = (batch_x - self.gamma[0])[0].repeat((len(batch_x),) + (1,) * (dims - 1))
            return x_0_pattern * (batch_x - self.gamma[0]) > 0


def check_block(model, module):
    return type(module) in [ConvBlock, LinearBlock, Bottleneck] and module != model.layers[-1]
