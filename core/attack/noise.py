from core.attack.base import Attack
import torch


class Noise(Attack):
    def __init__(self, model, args, **kwargs):
        super(Noise, self).__init__(model, args, **kwargs)
        self.sigma = args.sigma

    def forward(self, images, labels):
        n = images + torch.randn_like(images) * self.sigma
        return n
