import torch
from torch.nn.utils.prune import l1_unstructured, random_structured, ln_structured, remove, identity, is_pruned
from models.blocks import ConvBlock, LinearBlock


def compute_importance(weight, channel_entropy, eta):
    """
    Compute the importance score based on weight and entropy of a channel
    :param weight:  Weight of the module, shape as:
                    ConvBlock: in_channels * out_channels * kernel_size_1 * kernel_size_2
                    LinearBlock: in_channels * out_channels
    :param channel_entropy: The averaged entropy of each channel, shape as in_channels * 1 * (1 * 1)
    :param eta: the importance of entropy in pruning,
                -1:     hard prune without using weight
                0:      prune by weight
                1:      prune by channel_entropy
                2: weight * entropy
                else:   eta * channel_entropy * weight
    :return:    The importance_scores
    """
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1
    weight = abs(weight)
    e_new_shape = (-1, ) + (1, ) * (weight.dim() - 1)
    channel_entropy = torch.tensor(channel_entropy).view(e_new_shape).cuda()
    if eta == -1:
        importance_scores = channel_entropy * torch.ones_like(weight)
    elif eta == 0:
        importance_scores = weight
    elif eta == 2:
        importance_scores = channel_entropy * weight
    elif eta == 3:
        importance_scores =1 / (1 / (channel_entropy +1e-8) + 1 / (weight+ 1e-8))
    elif eta == 4:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy * normed_weight
    elif eta == 5:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy + normed_weight
    else:
        raise ValueError()

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
    weights = getattr(block.LT, 'weight').detach()
    num_dim = len(block_entropy[0].shape)                               # num of dimensions
    channel_entropy = block_entropy[0].mean(tuple(range(1, num_dim)))   # averaged entropy (out_channels, )
    lt_im_score = compute_importance(weights, channel_entropy, eta)
    bn_im_score = lt_im_score.mean(dim=tuple(range(1, weights.dim())))

    block_type = 'ConvBlock' if isinstance(block, ConvBlock) else 'LinearBlock'
    im_dict = {
        (block.LT, 'weight', block_type): lt_im_score,
        (block.BN, 'weight', block_type): bn_im_score,
        (block.BN, 'bias', block_type): bn_im_score
    }
    return im_dict


def prune_bottle_neck_block(block):
    pass


def remove_block(block):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        remove(block.LT, 'weight')
        remove(block.BN, 'weight')
        remove(block.BN, 'bias')


def iteratively_prune(im_dict, args):
    for param_to_prune, im_score in im_dict.items():
        prune_module(param_to_prune, im_score, args)


def prune_module(param_to_prune, im_score, args):
    module, name, block = param_to_prune
    cur_param = getattr(module, name)
    num_dims = cur_param.dim()
    if args.method == 'LnStructured':
        if num_dims > 1:
            ln_structured(module, name, args.amount, 2, dim=0, importance_scores=im_score.cuda())
        else:
            l1_unstructured(module, name, args.amount, importance_scores=im_score.cuda())
    elif args.method == 'RandomStructured':
        random_structured(module, name, args.amount, dim=0)
    elif args.method == 'Hard':
        slc = [slice(None)] * num_dims
        if hasattr(module, name + '_mask'):
            # how many channels remained to be pruned
            keep_channel = getattr(module, name + '_mask')[(slice(None, ),) + (0,) * (num_dims - 1)] != 0
            slc[0] = keep_channel
        tensor_to_pru = im_score[slc]
        hard_ind = tensor_to_pru[(slice(None, ),) + (0,) * (num_dims - 1)]

        # set an upper bound for pruning
        maximum_to_prune = max(len(tensor_to_pru) - 0.15 * len(im_score), 0)

        if block == 'ConvBlock':
            num_filters = min(torch.sum(hard_ind < args.conv_pru_bound).to(torch.int), maximum_to_prune)
        elif block == 'LinearBlock':
            num_filters = min(torch.sum(hard_ind < args.fc_pru_bound).to(torch.int), maximum_to_prune)
        else:
            raise NameError("Invalid Block for pruning")

        if num_filters == 0:
            identity(module, name)
        elif 0 < num_filters < len(tensor_to_pru):
            if num_dims > 1:
                ln_structured(module, name, int(num_filters), 2, dim=0, importance_scores=im_score.cuda())
            else:
                l1_unstructured(module, name, int(num_filters), importance_scores=im_score.cuda())
        # else:
        #     Warning("Amount to prune should be less than number of params, "
        #                      "got {0} and {1}".format(num_filters, len(tensor_to_pru)))
        #     if not hasattr(module, name + '_mask'):
        #         identity(module, name)


def monitor(importance_dict, info):
    cur_pruned = []
    cur_element = []
    for i, (module, name, block) in enumerate(importance_dict.keys()):
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