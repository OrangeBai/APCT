from .fgsm import FGSM
from .noise import Noise
from .pgd import PGD
from .vanilla import Vanilla


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
