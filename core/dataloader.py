from torchvision.transforms import *
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torch.utils.data import DataLoader
from config import *
import os

MNIST_MEAN_STD = (0.1307,), (0.3081,)
CIAFR10_MEAN_STD = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)]
CIAFR100_MEAN_STD = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]
IMAGENET_MEAN_STD = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]


def set_dataloader(args, datasets=None):
    if datasets is None:
        train_dataset, val_dataset = set_dataset(args)
    else:
        train_dataset, val_dataset = datasets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4)
    return train_loader, val_loader


def set_dataset(args):
    train_transform, test_transform = set_transforms(args)
    if args.dataset.lower() == 'mnist':
        train_dataset = MNIST(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset.lower() == 'cifar10':
        train_dataset = CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset.lower() == 'cifar100':
        train_dataset = CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(DATA_PATH, 'ImageNet-2012', 'train')
        test_dir = os.path.join(DATA_PATH, 'ImageNet-2012', 'val')
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        test_dataset = ImageFolder(test_dir, transform=test_transform)
    else:
        raise NameError('No dataset named %s' % args.dataset)
    return train_dataset, test_dataset


def set_transforms(args):
    if args.dataset.lower() == 'mnist':
        train_composed = [RandomCrop(32, padding=4), ToTensor()]
        test_composed = [ToTensor()]
    elif args.dataset.lower() in ['cifar10', 'cifar100']:
        train_composed = [RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor()]
        test_composed = [transforms.ToTensor()]
    elif args.dataset.lower() == 'imagenet':
        train_composed = [RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor()]
        test_composed = [transforms.Resize(256), transforms.CenterCrop(224), ToTensor()]
    else:
        raise NameError('No dataset named' % args.dataset)
    return Compose(train_composed), Compose(test_composed)


def set_mean_sed(args):
    if args.dataset.lower() == 'cifar10':
        mean, std = CIAFR10_MEAN_STD
    elif args.dataset.lower() == 'cifar100':
        mean, std = CIAFR100_MEAN_STD
    elif args.dataset.lower() == 'mnist':
        mean, std = MNIST_MEAN_STD
    elif args.dataset.lower() == 'imagenet':
        mean, std = IMAGENET_MEAN_STD
    else:
        raise NameError()
    return mean, std
