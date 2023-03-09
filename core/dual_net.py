from models.blocks import *
from models.net.resnet import Bottleneck
import torch.nn.functional as F


class DualNet(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.gamma = set_gamma(args.activation)

        self.counter = -1

        self.fixed_neurons = []
        self.handles = []

    @property
    def count_block_len(self):
        counter = 0
        for module in self.net.layers.children():
            if type(module) in [ConvBlock, LinearBlock, Bottleneck]:
                counter += 1
        return counter - 1

    def compute_fixed(self, x1, x2):
        self.net.eval()
        self.counter = -1
        fixed_neurons = []
        x1 = self.net.norm_layer(x1)
        x2 = self.net.norm_layer(x2)
        for i, module in enumerate(self.net.layers.children()):
            x1 = self.compute_pre_act(module, x1)
            x2 = self.compute_pre_act(module, x2)
            if self.check_block(module) and i != len(self.net.layers) - 1:
                fixed = self.compute_fix(x1, x2)
                fixed_neurons += [fixed]
                x1 = module.Act(x1)
                x2 = module.Act(x2)
            else:
                fixed_neurons += [None]

        self.fixed_neurons = fixed_neurons
        self.net.train()
        return fixed_neurons

    def predict(self, x, eta_fixed, eta_float):
        """
        for a batch of data, the first one should be raw data without perturbation
        """
        self.counter = -1
        fixed_neurons = []
        batch_x = self.net.norm_layer(x)
        for i, module in enumerate(list(self.net.layers)):
            batch_x = self.compute_pre_act(module, batch_x)
            if self.check_block(module) and i != len(list(self.net.layers)):
                fixed = self.compute_fix_single_batch(batch_x)
                fixed_neurons += [fixed]

                h = self.set_hook(fixed, eta_fixed, eta_float, False)
                self.handles += [module.Act.register_forward_pre_hook(h)]
                batch_x = module.Act(batch_x)
            else:
                fixed_neurons += [None]

        self.fixed_neurons = fixed_neurons
        self.remove_handles()
        return batch_x

    def forward(self, x, fixed_neurons, eta_fixed, eta_float):
        self.counter = -1
        batch_x = self.net.norm_layer(x)
        for i, module in enumerate(list(self.net.layers)):
            batch_x = self.compute_pre_act(module, batch_x)
            if self.check_block(module) and i != len(list(self.net.layers)) - 1:
                h = self.set_hook(fixed_neurons[i], eta_fixed, eta_float, True)
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

    def check_block(self, module):
        if type(module) in [ConvBlock, LinearBlock, Bottleneck]:
            self.counter += 1
            return 1
        else:
            return 0

    def dn_forward(self, x):
        for i, module in enumerate(self.net.layers.children()):
            x = self.compute_pre_act(module, x)
            if type(module) in [ConvBlock, LinearBlock]:
                p0 = (x < 0).sum(axis=0) > self.dn_rate * len(x)
                p1 = (x > 0).sum(axis=0) > self.dn_rate * len(x)
                p_same = torch.all(torch.stack([p0, p1]), dim=0).unsqueeze(dim=0)
                x = self.x_mask(x, self.eta_dn, p_same) + x * ~p_same
                x = module.Act(x)
        return x

    @property
    def mask_ratio(self):
        mask_mean = []
        if len(self.fixed_neurons) == 0:
            return 0
        for b_mask in self.fixed_neurons:
            if b_mask is not None:
                mask_mean += [to_numpy(b_mask).mean()]
            return np.array(mask_mean).mean()

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

    def compute_fix(self, x_1, x_2):
        if len(self.gamma) == 1:
            return (x_1 - self.gamma[0]) * (x_2 - self.gamma[0]) > 0
        else:
            stacked = torch.stack([(x_1 - g) * (x_2 - g) > 0 for g in self.gamma])
            return torch.all(stacked, dim=0)

    def compute_fix_single_batch(self, batch_x):
        dims = len(batch_x.shape)
        if len(self.gamma) == 1:
            x_0_pattern = (batch_x - self.gamma[0])[0].repeat((len(batch_x),) + (1,) * (dims - 1))
            return x_0_pattern * (batch_x - self.gamma[0]) > 0

    @staticmethod
    def _batch_norm(layer, x):
        if type(layer) in [nn.BatchNorm2d, nn.BatchNorm1d]:
            return F.batch_norm(x, layer.running_mean, layer.running_var, layer.weight, layer.bias)
        else:
            return x
