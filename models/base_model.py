import importlib
import os
from collections import OrderedDict
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    # TODO Record epoch info
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.set_up_kwargs = {'batch_norm': args.batch_norm, 'activation': args.activation}

    def forward(self, x):
        pass

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
    model = None
    if args.model_type == 'dnn':
        model_file_name = "models." + args.model_type
        modules = importlib.import_module(model_file_name)
        model = modules.__dict__['DNN'](args)
    elif args.model_type == 'mini':
        model_file_name = "models." + "mini"
        modules = importlib.import_module(model_file_name)
        model = modules.set_model(args)
    elif args.model_type == 'net':
        model_file_name = "models." + "net"
        modules = importlib.import_module(model_file_name)
        model = modules.set_model(args)
    else:
        raise NameError

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_file_name, args.net))
        exit(0)
    else:
        return model
