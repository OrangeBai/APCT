import time
from core.models.blocks import *


class BaseHook:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.features = {}

    def set_up(self):
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
    def __init__(self, model, Gamma):
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1

    def hook(self, block_name, layer_name):
        def fn(layer, input_var, output_var):
            """
            Count the frequency of each pattern
            """
            input_var = input_var[0]
            pattern = get_pattern(input_var, self.Gamma)
            if self.features[block_name][layer_name] is None:
                self.features[block_name][layer_name] = np.zeros((self.num_pattern,) + pattern.shape[1:])
            for i in range(1 + len(self.Gamma)):
                self.features[block_name][layer_name][i] += (pattern == i).sum(axis=0)
        return fn

    def retrieve(self):
        entropy = []
        for block in self.features.values():
            for layer in block.values():
                layer = layer.reshape(self.num_pattern, -1)
                layer /= layer.sum(axis=0)
                s = np.zeros(layer.shape[1])
                for j in range(self.num_pattern):
                    s += -layer[j] * np.log(1e-8 + layer[j])
                # layer_entropy = []
                entropy.append(s)
        return entropy


class FloatHook(BaseHook):
    def __init__(self, model, Gamma):
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1

    def save_outputs_hook(self, block_name, module_name):
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


def set_bound_hook(stored_values, bound=1e-1):
    """
    record input values of the module
    @param stored_values: recorder
    @return: activation hook
    """

    def hook(layer, input_var, output_var):
        input_var = input_var[0]
        bound_sum = input_var[input_var.abs() < bound].abs().sum()
        stored_values.append(bound_sum)

    return hook




def get_pattern(input_var, Gamma):
    pattern = np.zeros(input_var.shape)
    num_of_pattern = len(Gamma)

    pattern[to_numpy(input_var < Gamma[0])] = 0
    pattern[to_numpy(input_var > Gamma[-1])] = num_of_pattern
    for i in range(1, num_of_pattern):
        valid = pattern > Gamma[i] * pattern < Gamma[i + 1]
        pattern[to_numpy(valid)] = i
    return pattern


def get_similarity(pattern, Gamma):
    ps = []
    for i in range(len(Gamma) + 1):
        ps_i = (pattern == i).sum(axis=0) / len(pattern)
        ps.append(ps_i)
    return np.array(ps)


def min_max_pattern(pattern, mode='min'):
    if mode == 'min':
        return pattern.min(axis=0).astype(int)
    else:
        return pattern.max(axis=0).astype(int)


def unpack(stored_values):
    unpacked = [[np.concatenate(layer)] if type(layer[0]) == np.ndarray else [torch.concat(layer)]
                for block in stored_values.values() for layer in block.values()]
    return unpacked


def retrieve_float_neurons(stored_values):
    """
    calculate the float neurons of given pattern
    @param stored_values: stored value from ModelHook
    @return:
    """
    unpacked = unpack(stored_values)
    return [[np.all(layer - layer[0], axis=0) for layer in block]
            for block in unpacked]


def retrieve_lb_ub(stored_values, grad_bound):
    r"""
    Compute the upper and lower derivative bound for the pattern
    @param stored_values: recorder
    @param grad_bound: A set of gradient bounds for each activation region with length #\Gamma + 1. For instance,
                        if activation is ReLU with Gamma=[0]
                            the grad bound should be [(0,0), (1,1)]
    @return: the bound for each neuron, shape as:
                [
                    [[grad_lower_bound], [grad_upper_bound]], (instance_1),
                    [[grad_lower_bound], [grad_upper_bound]], (instance_2),
                    ...,
                    [[grad_lower_bound], [grad_upper_bound]], (instance_n)
                ]
    """

    unpacked = unpack(stored_values)

    # max_lambda = np.vectorize(lambda x: grad_bound[x][1])
    # min_lambda = np.vectorize(lambda x: grad_bound[x][0])
    #
    # lb = [[[min_max_pattern(layer[i: i + sample_size], 'min')
    #         for i in range(0, len(layer), sample_size)] for layer in block] for block in unpacked]
    #
    # ub = [[[min_max_pattern(layer[i: i + sample_size], 'max')
    #         for i in range(0, len(layer), sample_size)] for layer in block] for block in unpacked]

    net_lb, net_ub = [], []

    for block in unpacked:
        block_lb = []
        block_ub = []
        for layer in block:
            layer_lb, layer_ub = layer_lb_ub(layer, grad_bound)
            block_lb.append(layer_lb)
            block_ub.append(layer_ub)
        net_lb.append(block_lb)
        net_ub.append(block_ub)
    return net_lb, net_ub


def layer_lb_ub(layer, grad_bound):
    layer_lb = np.zeros(layer.shape[1:])
    layer_ub = np.zeros(layer.shape[1:])
    min_pattern = min_max_pattern(layer, 'min')
    max_pattern = min_max_pattern(layer, 'max')
    for j, (lb, ub) in enumerate(grad_bound):
        layer_lb[min_pattern == j] = lb
        layer_ub[max_pattern == j] = ub
    return layer_lb, layer_ub


def list_all(data, storage=None):
    if type(data) == list or type(data) == tuple:
        for d in data:
            list_all(d, storage)
    elif type(data) == dict:
        for k, v in data.items():
            list_all(v, storage)
    else:
        storage.append(data)
