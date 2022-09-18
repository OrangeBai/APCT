from attack.base import Attack


class Vanilla(Attack):
    def __init__(self, model, args, **kwargs):
        super(Vanilla, self).__init__(model, args)

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """

        return images
