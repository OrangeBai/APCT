import torch
from torch.nn.utils.prune import global_unstructured, L1Unstructured, random_structured, \
    ln_structured, remove, identity, is_pruned, l1_unstructured
from models.blocks import ConvBlock, LinearBlock
import torch.nn as nn


def prune_model(args, im_scores, channel_entropy):
    if args.method == 'L1Unstructured':
        l1_unstructured_prune(args, im_scores, channel_entropy)
    elif args.method == 'LnStructured':
        ln_structured_prune(args, im_scores, channel_entropy)
    elif args.method == 'Hard':
        hard_prune(args, im_scores, channel_entropy)
    else:
        raise NameError()


def compute_im_score(block, block_entropy, eta):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        return compute_linear_im_score(block, block_entropy, eta)
    # elif isinstance(block, LinearBlock):
    #     return prune_linear_block(block, block_entropy, eta)


def compute_linear_im_score(block, block_entropy, eta):
    """
    :param block: Conv block to be pruned
    :param block_entropy: entropy of the block output (out_channels * H * W)
    :param eta: hyper parameter.
    :return:
    """
    weights = getattr(block.LT, 'weight').detach()
    num_dim = len(block_entropy[0].shape)  # num of dimensions
    channel_entropy = block_entropy[0].mean(tuple(range(1, num_dim)))  # averaged entropy (out_channels, )
    lt_im_score = compute_importance(weights, channel_entropy, eta)
    bn_im_score = lt_im_score.mean(dim=tuple(range(1, weights.dim())))

    block_type = 'ConvBlock' if isinstance(block, ConvBlock) else 'LinearBlock'
    im_dict = {
        (block.LT, 'weight', block_type): lt_im_score,
        (block.BN, 'weight', block_type): bn_im_score,
        (block.BN, 'bias', block_type): bn_im_score
    }
    return im_dict


def avg_im_score(im_dict):
    return [v.mean() for k, v in im_dict.items()]


def prune_bottle_neck_block(block):
    pass


def remove_block(block):
    if isinstance(block, ConvBlock) or isinstance(block, LinearBlock):
        remove(block.LT, 'weight')
        remove(block.BN, 'weight')
        remove(block.BN, 'bias')


# def iteratively_prune(im_dict, args):
#     if args.method == 'L1Unstructured':
#         l1_unstructured_prune(im_dict, args)
#     elif args.method == 'LnStructured':
#         ln_structured_prune(im_dict, args)
#     elif args.method == 'Hard':
#         hard_prune(im_dict, args)
#     else:
#         raise NameError()


def l1_unstructured_prune(args, im_scores, channel_entropy):
    params = [(k[0], k[1]) for k in im_scores.keys()]
    pru_params, pru_im = [], {}
    for i, (param, (im_k, im_v)) in enumerate(zip(params, im_scores.items())):
        param_to_prune = getattr(param[0], param[1])
        sparsity = torch.sum(param_to_prune == 0) / param_to_prune.nelement()
        if sparsity < 0.80:
            pru_params.append(param)
            pru_im[im_k] = im_v

    global_unstructured(pru_params, L1Unstructured, pru_im, amount=args.conv_amount)


def ln_structured_prune(args, im_scores, channel_entropy):
    im_mean = sum([1 / torch.sqrt(v.mean()) for v in channel_entropy.values() if len(v) > 0]) / len(channel_entropy)
    cur_ratio = 0
    for k, im in im_scores.items():
        module, name, block = k
        num_dims = getattr(module, name).dim()
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            cur_ratio = (1 / torch.sqrt(channel_entropy[k].mean())) / im_mean
        if num_dims > 1:
            num_filters < len(tensor_to_pru)
            ln_structured(module, name, cur_ratio.item() * args.conv_amount, 2, dim=0, importance_scores=im.cuda())

        else:
            l1_unstructured(module, name, cur_ratio.item() * args.conv_amount, importance_scores=im.cuda())
    return


def compute_importance(weight, channel_entropy, eta):
    """
    Compute the importance score based on weight and entropy of a channel
    :param weight:  Weight of the module, shape as:
                    ConvBlock: in_channels * out_channels * kernel_size_1 * kernel_size_2
                    LinearBlock: in_channels * out_channels
    :param channel_entropy: The averaged entropy of each channel, shape as in_channels * 1 * (1 * 1)
    :param eta: the importance of entropy in pruning,
                0:      prune by weight
                1:      prune by channel_entropy
                2:      prune by weight * entropy
                3:
                else:   eta * channel_entropy * weight
    :return:    The importance_scores
    """
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1
    e_new_shape = (-1,) + (1,) * (weight.dim() - 1)
    channel_entropy = torch.tensor(channel_entropy).view(e_new_shape).cuda()
    if eta == 0:
        importance_scores = weight
    elif eta == 1:
        importance_scores = channel_entropy * torch.ones_like(weight)
    elif eta == 2:
        importance_scores = channel_entropy * weight
    elif eta == 3:
        importance_scores = 1 / (1 / (channel_entropy + 1e-8) + 1 / (weight + 1e-4))
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


# def prune_module(param_to_prune, im_score, args):
#     module, name, block = param_to_prune
#     cur_param = getattr(module, name)
#     num_dims = cur_param.dim()
#     elif args.method == 'RandomStructured':
#         random_structured(module, name, args.amount, dim=0)
#     elif args.method == 'Hard':
#         cur_param = getattr(module, name)
#         num_dims = cur_param.dim()
#         slc = [slice(None)] * num_dims
#         if hasattr(module, name + '_mask'):
#             keep_channel = getattr(module, name + '_mask').sum(tuple(range(1, cur_param.dim()))) != 0
#             slc[0] = keep_channel
#         tensor_to_pru = im_score[slc]
#
#         hard_ind = torch.Tensor(tensor_to_pru[(slice(None, ),) + (0,) * (num_dims - 1)])
#         if block == 'ConvBlock':
#             num_filters = torch.sum(hard_ind < args.conv_pru_bound).to(torch.int)
#         elif block == 'LinearBlock':
#             num_filters = torch.sum(hard_ind < args.fc_pru_bound).to(torch.int)
#         else:
#             raise NameError("Invalid Block for pruning")
#         if num_filters == 0:
#             identity(module, name)
#         elif 0 < num_filters < len(tensor_to_pru):
#             if num_dims > 1 :
#                 ln_structured(module, name, int(num_filters), 2, dim=0, importance_scores=im_score.cuda())
#             else:
#                 l1_unstructured(module, name, int(num_filters), importance_scores=im_score.cuda())
#         else:
#             raise ValueError("Amount to prune should be less than number of params, "
#                              "got {0} and {1}".format(num_filters, len(tensor_to_pru)))

def hard_prune(im_dict, args):
    pass
    #
    # cur_param = getattr(module, name)
    # num_dims = cur_param.dim()
    # slc = [slice(None)] * num_dims
    # if hasattr(module, name + '_mask'):
    #     keep_channel = getattr(module, name + '_mask').sum(tuple(range(1, cur_param.dim()))) != 0
    #     slc[0] = keep_channel
    # tensor_to_pru = im_score[slc]
    #
    # hard_ind = torch.Tensor(tensor_to_pru[(slice(None, ),) + (0,) * (num_dims - 1)])
    # if block == 'ConvBlock':
    #     num_filters = torch.sum(hard_ind < args.conv_pru_bound).to(torch.int)
    # elif block == 'LinearBlock':
    #     num_filters = torch.sum(hard_ind < args.fc_pru_bound).to(torch.int)
    # else:
    #     raise NameError("Invalid Block for pruning")
    # if num_filters == 0:
    #     identity(module, name)
    # elif 0 < num_filters < len(tensor_to_pru):
    #     if num_dims > 1 :
    #         ln_structured(module, name, int(num_filters), 2, dim=0, importance_scores=im_score.cuda())
    #     else:
    #         l1_unstructured(module, name, int(num_filters), importance_scores=im_score.cuda())
    # else:
    #     raise ValueError("Amount to prune should be less than number of params, "
    #                      "got {0} and {1}".format(num_filters, len(tensor_to_pru)))


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
    return info


def restructure(model):
    """remove the zero filters"""
    pass
