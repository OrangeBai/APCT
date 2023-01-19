import torch
from torch.nn.utils.prune import global_unstructured, L1Unstructured, random_structured, ln_structured, RandomStructured
from models.blocks import ConvBlock, LinearBlock


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
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1
    e_new_shape = (-1, ) + (1, ) * (weight.dim() - 1)
    channel_entropy = torch.tensor(channel_entropy).view(e_new_shape)
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


def prune_block(block, block_entropy, eta):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        return prune_linear_transform_block(block, block_entropy, eta)
    # elif isinstance(block, LinearBlock):
    #     return prune_linear_block(block, block_entropy, eta)


def prune_linear_transform_block(block, block_entropy, eta):
    """
    :param block: Conv block to be pruned
    :param block_entropy: entropy of the block output (out_channels * H * W)
    :param eta: hyper parameter.
    :return:
    """
    weights = getattr(block.LT, 'weight').detach().cpu()
    num_dim = len(block_entropy[0].shape)                               # num of dimensions
    channel_entropy = block_entropy[0].mean(tuple(range(1, num_dim)))   # averaged entropy (out_channels, )
    lt_im_score = compute_importance(weights, channel_entropy, eta)
    bn_im_score = lt_im_score.mean(dim=tuple(range(1, weights.dim())))

    im_dict = {
        (block.LT, 'weight'): lt_im_score,
        (block.BN, 'weight'): bn_im_score,
        (block.BN, 'bias'): bn_im_score
    }
    return im_dict


def prune_bottle_neck_block(block):
    pass


def iteratively_prune(parameters_to_prune, im_dict, args):
    for cur_param, cur_name in parameters_to_prune:
        prune_module(cur_param, cur_name, im_dict[(cur_param, cur_name)], args)


def prune_module(cur_param, cur_name, im_score, args):
    if args.method == 'LnStructured':
        ln_structured(cur_param, cur_name, args.amount, 2, dim=0, importance_scores=im_score)
    elif args.method == 'RandomStructured':
        random_structured(cur_param, cur_name, args.amount, dim=0)
    elif args.method == 'Hard':
        n_dims = len(cur_param.weight.shape)
        slc = [slice(None)] * n_dims
        if hasattr(cur_param, 'weight_mask'):
            keep_channel = cur_param.weight_mask.sum([d for d in range(n_dims) if d != 0]) != 0
            slc[0] = keep_channel

            tensor_to_pru = im_score[slc]
            if isinstance(cur_param, torch.nn.Conv2d):
                num_filters = torch.sum(torch.as_tensor(tensor_to_pru[:, 0, 0, 0] < args.conv_pru_bound).to(torch.int))
            elif isinstance(cur_param, torch.nn.Linear):
                num_filters = torch.sum(torch.as_tensor(tensor_to_pru[:, 0] < args.conv_pru_bound).to(torch.int))
            else:
                raise NameError("Invalid Block for pruning")
            if 0 < num_filters < len(tensor_to_pru):
                ln_structured(cur_param, cur_name, int(num_filters), 2, dim=0, importance_scores=im_score)


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


def restructure(model):
    """remove the zero filters"""
    pass



# def compute_params(block, block_entropy, parameters_to_prune, importance_dict, eta):
#     """
#     Update the parameters for prune.
#     :param block:
#     :param block_entropy:
#     :param parameters_to_prune:
#     :param importance_dict:
#     :param eta:
#     :return:
#     """
#     prune_block(block, block_entropy, eta)
#     # weight = get_block_weight(block)
#     # channel_entropy = compute_neuron_entropy(block, block_entropy)
#     # importance = compute_importance(weight, channel_entropy, eta=eta)
#     # layer = get_pru_layer(block)
#     # parameters_to_prune.append((layer, 'weight'))
#     # importance_dict[layer] = importance
#     return prune_block(block, block_entropy, eta)