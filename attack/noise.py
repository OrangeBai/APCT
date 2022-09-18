from attack.base import *


class Noise(Attack):
    def __init__(self, model, args, **kwargs):
        super(Noise, self).__init__(model, args, **kwargs)
        self.sigma = args.sigma

    def forward(self, images, labels):
        images = self._reverse_norm(images)
        n = images + torch.randn_like(images) * self.sigma
        return self._norm(n)
