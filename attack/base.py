from core.utils import *
#from dataloader.base import set_mean_sed


class Attack(nn.Module):
    def __init__(self, model, args, **kwargs):
        super().__init__()
        self.model = model
        self.args = args
        self.mean, self.std = [torch.tensor(d).view(len(d), 1, 1) for d in set_mean_sed(args)]
        self.ord = kwargs['ord'] if 'ord' in kwargs.keys() else 'inf'
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def forward(self, *args):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

