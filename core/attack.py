import torch

from core.utils import *
from core.dataloader import set_mean_sed


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


class Vanilla(Attack):
    def __init__(self, model, args, **kwargs):
        super(Vanilla, self).__init__(model, args)

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """

        return images


class Noise(Attack):
    def __init__(self, model, args, **kwargs):
        super(Noise, self).__init__(model, args, **kwargs)
        self.sigma = args.sigma

    def forward(self, images, labels):
        n = images + torch.randn_like(images) * self.sigma
        return n.detach()


class FGSM(Attack):
    def __init__(self, model, args, **kwargs):
        super(FGSM, self).__init__(model, args, **kwargs)
        self.eps = args.eps

    @torch.enable_grad()
    def forward(self, images, labels):

        images.requires_grad = True
        loss = nn.CrossEntropyLoss()
        outputs = self.model(images)  # from (0, 1) to normalized, and forward the emodel
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        if self.ord == 'inf':
            adv_images = images + self.eps * grad.sign()
        else:
            grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=-1, keepdim=True)
            grad_norm = grad_norm.view(grad_norm.shape[0], grad_norm.shape[1], 1, 1)
            adv_images = images + self.eps * grad / grad_norm

        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images


class PGD(Attack):
    def __init__(self, model, args, **kwargs):
        super(PGD, self).__init__(model, args, **kwargs)
        self.random_start = True
        self.eps = kwargs['eps'] if 'eps' in kwargs.keys() else 8 / 255
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 2 / 255
        self.steps = kwargs['steps'] if 'steps' in kwargs.keys() else 10
        self.restarts = kwargs['restarts'] if 'restarts' in kwargs.keys() else 1

    def forward(self, images, labels):
        loss_fn = nn.CrossEntropyLoss()
        images = images.detach()
        images.requires_grad = True

        if self.random_start:
            if self.ord == 'inf':
                delta = torch.randn_like(images) * self.eps
            else:
                delta = torch.randn_like(images)
                # batch_size * 1 * 1 * 1
                delta_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=-1).view(delta.shape[0], 1, 1, 1)
                delta = delta / delta_norm * self.eps
        else:
            delta = torch.zeros_like(images)

        delta.requires_grad = True
        for _ in range(self.steps):
            outputs = self.model(images + delta)
            cost = loss_fn(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, delta, retain_graph=False, create_graph=False)[0]

            if self.ord == 'inf':
                delta = delta + self.alpha * grad.sign()
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            else:
                # TODO review this part
                grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=-1, keepdim=True)
                grad = grad / grad_norm.view(grad_norm.shape[0], grad_norm.shape[1], 1, 1)
                delta += self.alpha * grad

                # maks for each sample, whether greater than eps or not
                mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= self.eps
                scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=-1) + 1e-8  # norm for each sample
                scaling_factor[mask] = self.eps
                delta = delta / (scaling_factor.view(-1, 1, 1, 1)) * self.eps
        return torch.clamp(images + delta, min=0, max=1).detach()


def set_attack(model, args, **kwargs):
    name = args.attack.lower()
    if name == 'vanilla':
        attack = Vanilla(model, args, **kwargs)
    elif name == 'fgsm':
        attack = FGSM(model, args, **kwargs)
    elif name == 'pgd':
        attack = PGD(model, args, **kwargs)
    elif name == 'noise':
        attack = Noise(model, args, **kwargs)
    else:
        raise NameError('Attack {0} not found'.format(name))
    return attack
