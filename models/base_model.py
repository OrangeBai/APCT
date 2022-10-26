import importlib
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from engine.dataloader import set_mean_sed


class BaseModel(nn.Module):
    # TODO Record epoch info
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.norm_layer = NormalizeLayer(*set_mean_sed(args))

    def save_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        torch.save(self.state_dict(), model_path)
        return

    def load_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        self.load_weights(torch.load(model_path))

        print('Loading model from {}'.format(model_path))
        return

    def load_weights(self, state_dict):
        new_dict = OrderedDict()
        for (k1, v1), (k2, v2) in zip(self.state_dict().items(), state_dict.items()):
            if v1.shape == v2.shape:
                new_dict[k1] = v2
            else:
                raise KeyError
        self.load_state_dict(new_dict)


def build_model(args):
    """Import the module "model/[model_name]_model.py"."""
    if args.model_type == 'dnn':
        model_file_name = "models." + args.model_type
    elif args.model_type == 'mini':
        model_file_name = "models." + "mini"
    elif args.model_type == 'net':
        model_file_name = "models." + "net"
    else:
        raise NameError('No model type named %s' % args.model_type)

    modules = importlib.import_module(model_file_name)
    try:
        net = getattr(modules, args.net)
        model = net(args)
    except NameError:
        raise NameError('No model named %s in %s' % args.net, model_file_name)
    return model


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, x: torch.tensor):
        (batch_size, num_channels, height, width) = x.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        std = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (x - means) / std
