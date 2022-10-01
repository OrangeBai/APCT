from attack.base import *


class PGD(Attack):
    def __init__(self, model, args, **kwargs):
        super(PGD, self).__init__(model, args, **kwargs)
        self.random_start = True
        self.eps = kwargs['eps'] if 'eps' in kwargs.keys() else 8 / 255
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 2 / 255
        self.steps = kwargs['steps'] if 'steps' in kwargs.keys() else 10
        self.restarts = kwargs['restarts'] if 'restarts' in kwargs.keys() else 1

    def attack(self, images, labels):
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
