from .cxfy import *
from .resnet import *
from .vgg import *


def set_model(args):
    if 'vgg' in args.net.lower():
        return VGG(args)
    elif 'res' in args.net.lower():
        return Resnet(args, )
    elif 'cxfy' in args.net.lower():
        return CXFY(args)
