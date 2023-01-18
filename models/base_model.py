import importlib
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from core.dataloader import set_mean_sed
from models.blocks import NormalizeLayer


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
        # torch.save(self.state_dict(), model_path)
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
    if args.dataset.lower() == 'minist':
        model_file_name = "models." + 'dnn'
    elif args.dataset.lower() in ['cifar10', 'cifar100']:
        model_file_name = "models." + "mini"
    elif args.dataset.lower() == 'imagenet':
        model_file_name = "models." + "net"
    else:
        raise NameError()

    modules = importlib.import_module(model_file_name)
    try:
        net = getattr(modules, args.net)
        model = net(args)
    except NameError:
        raise NameError('No model named %s in %s' % args.net, model_file_name)
    return model


