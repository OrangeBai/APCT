import torch
from torch.nn.utils.prune import global_unstructured, L1Unstructured, random_structured, ln_structured, RandomStructured

from models.blocks import ConvBlock, LinearBlock


def check_valid_block(block):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        return True
    return False


def get_block_weight(block):
    if isinstance(block, ConvBlock):
        return getattr(block.Conv, 'weight')
    elif isinstance(block, LinearBlock):
        return getattr(block.FC, 'weight')
    else:
        raise NameError("Invalid Block for pruning")


def compute_importance(weight, channel_entropy, eta):
    if eta is None:
        importance_scores = None
    elif eta == 0:
        importance_scores = channel_entropy * torch.ones_like(weight)
    else:
        importance_scores = eta * channel_entropy * weight

    return importance_scores


def compute_neuron_entropy(block, neuron_entropy):
    if isinstance(block, ConvBlock):
        avg_axis = len(neuron_entropy[0].shape)
        filter_sim = neuron_entropy[0].mean(tuple(range(1, avg_axis)))
        channel_entropy = torch.tensor(filter_sim).view(-1, 1, 1, 1).cuda()
        return channel_entropy
    elif isinstance(block, LinearBlock):
        avg_axis = len(neuron_entropy[0].shape)
        filter_sim = neuron_entropy[0].mean(tuple(range(1, avg_axis)))
        channel_entropy = torch.tensor(filter_sim).view(-1, 1).cuda()
        return channel_entropy
    else:
        raise NameError("Invalid Block for pruning")


def get_pru_layer(block):
    if isinstance(block, ConvBlock):
        return block.Conv
    elif isinstance(block, LinearBlock):
        return block.FC


def compute_params(block, block_entropy, parameters_to_prune, importance_dict, eta):
    weight = get_block_weight(block)
    channel_entropy = compute_neuron_entropy(block, block_entropy)
    importance = compute_importance(weight, channel_entropy, eta=eta)
    layer = get_pru_layer(block)
    parameters_to_prune.append((layer, 'weight'))
    importance_dict[layer] = importance


def prune_model(parameters_to_prune, importance_dict, args):
    if args.method == 'L1Unstructured':
        global_unstructured(
            parameters_to_prune,
            pruning_method=L1Unstructured,
            amount=args.amount,
            importance_scores=importance_dict
        )
    elif args.method == 'RandomStructured':
        global_unstructured(
            parameters_to_prune,
            pruning_method=RandomStructured,
            amount=args.amount,
            importance_scores=importance_dict
        )
    elif args.method == 'LnStructured':
        for cur_param, cur_name in parameters_to_prune:
            ln_structured(cur_param, cur_name, args.amount, 2, dim=0,
                          importance_scores=importance_dict[cur_param])
    elif args.method == 'RandomUnstructured':
        for cur_param, cur_name in parameters_to_prune:
            random_structured(cur_param, cur_name, args.amount, dim=0)
    elif args.method == 'Hard':
        for cur_param, cur_name in parameters_to_prune:
            if isinstance(cur_param, torch.nn.Conv2d):
                num_filters = int(torch.sum(importance_dict[cur_param][:, 0, 0, 0] < args.conv_pru_bound))
            elif isinstance(cur_param, torch.nn.Linear):
                num_filters = int(torch.sum(importance_dict[cur_param][:, 0] < args.fc_pru_bound))
            else:
                raise NameError("Invalid Block for pruning")
            if num_filters > 0:
                ln_structured(cur_param, cur_name, num_filters, 2, dim=0,
                              importance_scores=importance_dict[cur_param])


def monitor(importance_dict):
    pruned = 0
    nele = 0

    for module in importance_dict.keys():
        pruned += torch.sum(module.weight == 0)
        nele += module.weight.nelement()
    print("Global sparsity: {:.2f}%".format(pruned / nele))
