from core.attack.base import Attack
import torch.nn as nn
import torch


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
