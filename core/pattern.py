from core.utils import *
from models.blocks import LinearBlock, ConvBlock
from random import random


class BaseHook:
    """
    Base method for adding hooks of a network.
    """
    def __init__(self, model):
        """
        Initialization method
        :param model: The model to be added
        """
        self.model = model
        self.handles = []       # handles for registered hooks
        self.features = {}      # recorder for computed attributes

    def set_up(self):
        """
        Remove all previous hooks and register hooks for each of t
        :return:
        """
        self.remove()
        for block_name, block in self.model.named_modules():
            if type(block) in [LinearBlock, ConvBlock]:
                self.features[block_name] = {}
                self.add_block_hook(block_name, block)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.features = {}

    def add_block_hook(self, block_name, block):
        for module_name, module in block.named_modules():
            if check_activation(module):
                self.features[block_name][module_name] = None
                handle = module.register_forward_hook(self.hook(block_name, module_name))
                self.handles.append(handle)

    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            pass
        return fn


class EntropyHook(BaseHook):
    """
    Entropy hook is a forward hood that computes the neuron entropy of the network.
    """
    def __init__(self, model, Gamma, ratio=1):
        """
        Initialization method.
        :param model: Pytorch model, which should be a sequential blocks
        :param Gamma: The breakpoint for a given activation function, i.e.
                        {0} separates ReLU and PReLU into two linear regions.
                        {-0.5, 0.5} separates Sigmoid and tanH into 2 semi-constant region and 1 semi-linear region.
        """
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1
        self.ratio = ratio

    def hook(self, block_name, layer_name):
        """

        :param block_name:
        :param layer_name:
        :return:
        """
        def fn(layer, input_var, output_var):
            """
            Count the frequency of each pattern
            """
            if random() < self.ratio:
                input_var = input_var[0]
                pattern = get_pattern(input_var, self.Gamma)
                if self.features[block_name][layer_name] is None:
                    self.features[block_name][layer_name] = np.zeros((self.num_pattern,) + pattern.shape[1:])
                for i in range(1 + len(self.Gamma)):
                    self.features[block_name][layer_name][i] += (pattern == i).sum(axis=0)
        return fn

    def retrieve(self, reshape=True):
        entropy = []
        for block in self.features.values():
            for layer in block.values():
                layer = layer.reshape(self.num_pattern, -1)
                layer /= layer.sum(axis=0)
                s = np.zeros(layer.shape[1:])
                for j in range(self.num_pattern):
                    s += -layer[j] * np.log(1e-8 + layer[j])

                entropy.append(s)
        return entropy


class PruneHook(EntropyHook):
    def __init__(self, model, Gamma, ratio=1):
        super().__init__(model, Gamma, ratio)

    def retrieve(self, reshape=True):
        entropy = {}
        for block_key, block in self.features.items():
            block_entropy = []
            for layer in block.values():
                if reshape:
                    layer = layer.reshape(self.num_pattern, -1)
                layer /= layer.sum(axis=0)
                s = np.zeros(layer.shape[1:])
                for j in range(self.num_pattern):
                    s += -layer[j] * np.log(1e-8 + layer[j])
                block_entropy.append(s)
            entropy[block_key] = block_entropy
        return entropy


class FloatHook(BaseHook):
    def __init__(self, model, Gamma):
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1

    def add_block_hook(self, block_name, block):
        for module_name, module in block.named_modules():
            if check_activation(module):
                self.features[block_name][module_name] = []
                handle = module.register_forward_hook(self.hook(block_name, module_name))
                self.handles.append(handle)

    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            input_var = input_var[0]
            pattern = get_pattern(input_var, self.Gamma)
            self.features[block_name][module_name].append(pattern)

        return fn

    def retrieve(self):
        ratios = []
        for block in self.features.values():
            for layer in block.values():
                con = np.concatenate(layer)
                diff = con - con[0]
                ft = np.any(diff, axis=0).sum()
                total = diff[0].size
                ratios.append(ft / total)
        self.features = {}
        self.set_up()
        return ratios


def get_pattern(input_var, Gamma):
    pattern = np.zeros(input_var.shape)
    num_of_pattern = len(Gamma)

    pattern[to_numpy(input_var <= Gamma[0])] = 0
    pattern[to_numpy(input_var > Gamma[-1])] = num_of_pattern
    for i in range(1, num_of_pattern):
        valid = pattern > Gamma[i] * pattern < Gamma[i + 1]
        pattern[to_numpy(valid)] = i
    return pattern


def min_max_pattern(pattern, mode='min'):
    if mode == 'min':
        return pattern.min(axis=0).astype(int)
    else:
        return pattern.max(axis=0).astype(int)


def unpack(stored_values):
    unpacked = [[np.concatenate(layer)] if type(layer[0]) == np.ndarray else [torch.concat(layer)]
                for block in stored_values.values() for layer in block.values()]
    return unpacked

