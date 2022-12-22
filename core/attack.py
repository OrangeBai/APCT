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
        return n


class FGSM(Attack):
    def __init__(self, model, args, **kwargs):
        super(FGSM, self).__init__(model, args, **kwargs)
        self.eps = args.eps

    def forward(self, images, labels):

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
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

        adv_images = images.clone().detach()

        if self.random_start:
            if self.ord == 'inf':
                adv_images = adv_images + torch.randn_like(adv_images) * self.eps
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            else:
                delta = torch.randn_like(adv_images)
                delta_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=-1).view(delta.shape[0], 1, 1, 1)
                delta = delta / delta_norm * torch.rand(1).cuda()
                adv_images = adv_images + delta

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = loss_fn(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.ord == 'inf':
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            else:
                grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=-1, keepdim=True)
                grad = grad / grad_norm.view(grad_norm.shape[0], grad_norm.shape[1], 1, 1)
                adv_images = adv_images.detach() + self.alpha * grad

                delta = adv_images - images
                mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= self.eps
                scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=-1) + 1e-8
                scaling_factor[mask] = self.eps
                delta = delta * self.eps / (scaling_factor.view(-1, 1, 1, 1))
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        adv_images = self._norm(adv_images)
        return adv_images


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
