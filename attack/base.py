from core.utils import *
#from dataloader.base import set_mean_sed


class Attack(nn.Module):
    def __init__(self, model, args, **kwargs):
        super().__init__()
        self.model = model
        self.args = args
        self.mean, self.std = [torch.tensor(d).view(len(d), 1, 1) for d in set_mean_sed(args)]
        self.ord = kwargs['ord'] if 'ord' in kwargs.keys() else 'inf'
        self.norm_layer = Normalize(mean=self.mean, std=self.std)

        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def _reverse_norm(self, x):
        device = x.device
        return x * self.std.to(device) + self.mean.to(device)

    def _norm(self, x):
        device = x.device
        return (x - self.mean.to(device)) / self.std.to(device)

    def forward(self, *args):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, images):
        # Broadcasting
        mean = self.mean.reshape(1, self.mean.shape[0], 1, 1)
        std = self.std.reshape(1, self.mean.shape[0], 1, 1)
        return (images - mean) / std
