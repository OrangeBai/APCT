import torch
from torch.nn.utils.prune import global_unstructured, L1Unstructured, random_structured, ln_structured, RandomStructured

from models.blocks import ConvBlock, LinearBlock


def check_valid_block(block):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        return True
    return False


def get_block_weight(block):
    """
    Get the weight of a given block
    :param block:
    :return:
    """
    if isinstance(block, ConvBlock):
        return getattr(block.Conv, 'weight')
    elif isinstance(block, LinearBlock):
        return getattr(block.FC, 'weight')
    else:
        raise NameError("Invalid Block for pruning")


def compute_importance(weight, channel_entropy, eta):
    """
    Compute the importance score based on weight and entropy of a channel
    :param weight:  Weight of the module, shape as:
                    ConvBlock: in_channels * out_channels * kernel_size_1 * kernel_size_2
                    LinearBlock: in_channels * out_channels
    :param channel_entropy: The averaged entropy of each channel, shape as in_channels * 1 * (1 * 1)
    :param eta: the importance of entropy in pruning,
                None:   prune without using weight
                0:      prune by weight
                else:   eta * channel_entropy * weight
    :return:    The importance_scores
    """
    if eta == 0:
        importance_scores = weight
    else:
        importance_scores = eta * channel_entropy * weight

    return importance_scores


def compute_neuron_entropy(block, neuron_entropy):
    """
    Compute the channel entropy of a network
    :param block: current block
    :param neuron_entropy:  the entropy of the pre-activation, shape as:
                            ConvBlock: out_channels * out_size * out * size
                            LinearBlock: out_channels
    :return: Averaged channel entropy
    """
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
    """
    Update the parameters for prune.
    :param block:
    :param block_entropy:
    :param parameters_to_prune:
    :param importance_dict:
    :param eta:
    :return:
    """
    weight = get_block_weight(block)
    channel_entropy = compute_neuron_entropy(block, block_entropy)
    importance = compute_importance(weight, channel_entropy, eta=eta)
    layer = get_pru_layer(block)
    parameters_to_prune.append((layer, 'weight'))
    importance_dict[layer] = importance
    return


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
            n_dims = len(cur_param.weight.shape)
            slc = [slice(None)] * n_dims
            if hasattr(cur_param, 'weight_mask'):
                keep_channel = cur_param.weight_mask.sum([d for d in range(n_dims) if d != 0]) != 0
                slc[0] = keep_channel

            tensor_to_pru = importance_dict[cur_param][slc]
            if isinstance(cur_param, torch.nn.Conv2d):
                num_filters = torch.sum(torch.as_tensor(tensor_to_pru[:, 0, 0, 0] < args.conv_pru_bound).to(torch.int))
            elif isinstance(cur_param, torch.nn.Linear):
                num_filters = torch.sum(torch.as_tensor(tensor_to_pru[:, 0] < args.conv_pru_bound).to(torch.int))
            else:
                raise NameError("Invalid Block for pruning")
            if 0 < num_filters < len(tensor_to_pru):
                ln_structured(cur_param, cur_name, int(num_filters), 2, dim=0,
                              importance_scores=importance_dict[cur_param])


def monitor(importance_dict, info):
    cur_pruned = []
    cur_element = []
    for i, module in enumerate(importance_dict.keys()):
        cur_pruned.append(torch.sum(module.weight == 0))
        cur_element.append(module.weight.nelement())
        info['sparsity/layer_{}'.format(str(i).zfill(2))] = torch.sum(module.weight == 0) / module.weight.nelement()
        print("Layer {0:d}: prune {1:d}, total {2:d}, "
              "sparsity: {3:.2f}%".format(i, cur_pruned[i], cur_element[i], cur_pruned[i] / cur_element[i]))

    print("Global sparsity: {:.2f}%".format(sum(cur_pruned) / sum(cur_element)))
    info['sparsity/global'] = sum(cur_pruned) / sum(cur_element)
    return
