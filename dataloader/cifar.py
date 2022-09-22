import numpy as np
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler
from config import *

CIAFR10_MEAN_STD = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)]
CIAFR100_MEAN_STD = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]


def get_loaders(args):
    train_dataset, test_dataset = get_data_set(args)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
    )
    return train_loader, test_loader


def get_single_sets(args, *labels):
    mean, std = CIAFR10_MEAN_STD if args.dataset == 'cifar10' else CIAFR100_MEAN_STD
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    num_workers = 0
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(DATA_PATH, train=True, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(DATA_PATH, train=True, transform=test_transform, download=True)
    else:
        raise NameError('No module called {0}'.format(args.dataset))

    sub_sets = []
    for label in labels:
        if args.dataset == 'cifar10':
            sub_set = datasets.CIFAR10(DATA_PATH, train=True, transform=test_transform, download=True)
        elif args.dataset == 'cifar100':
            sub_set = datasets.CIFAR100(DATA_PATH, train=True, transform=test_transform, download=True)
        else:
            raise NameError('No module called {0}'.format(args.dataset))
        indicator = np.array(test_dataset.targets) == label
        sub_set.data = test_dataset.data[indicator]
        sub_set.targets = np.array(test_dataset.targets)[indicator]
        sub_sets.append(sub_set)

    return [
        data.DataLoader(dataset=sub_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        for sub_set in sub_sets]


def get_data_set(args):
    mean, std = CIAFR10_MEAN_STD if args.dataset == 'cifar10' else CIAFR100_MEAN_STD

    train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    test_transform = [transforms.ToTensor()]
    if args.data_bn:
        train_transform += [transforms.Normalize(mean, std)]
        test_transform += [transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    else:
        raise NameError('No module called {0}'.format(args.dataset))
    return train_dataset, test_dataset
