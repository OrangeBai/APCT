from torchvision import transforms, datasets
import torch.utils.data as data
from config import *
from torch.utils.data.distributed import DistributedSampler
MNIST_MEAN_STD = (0.1307,), (0.3081,)


def get_loaders(args):
    mean, std = MNIST_MEAN_STD if args.data_bn else ((0,), (1,))
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.MNIST(DATA_PATH, train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=data_transform, download=True)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=test_sampler
    )
    return train_loader, test_loader


def get_single_sets(args, *labels):
    """
    Load test dataset according to labels:
    'all': all data
    1 : data with label 1
    2 : data with label 2 ......
    :param args: labels:
                    'all': all data
                    1 : data with label 1
                    2 : data with label 2 ......
    :return: a collection of data loaders
    """
    mean, std = (0.5,), (1,)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=data_transform, download=True)

    sub_sets = []
    for label in labels:
        sub_set = datasets.MNIST(DATA_PATH, train=False, transform=data_transform, download=True)
        indicator = test_dataset.targets[test_dataset.targets == label]
        sub_set.data = test_dataset.data[indicator]
        sub_set.targets = test_dataset.targets[indicator]
        sub_sets.append(sub_set)

    return [data.DataLoader(
        dataset=sub_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    ) for sub_set in sub_sets]


def get_dataset(args):
    mean, std = MNIST_MEAN_STD if args.data_bn else (0,), (1,)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.MNIST(DATA_PATH, train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=data_transform, download=True)
    return train_dataset, test_dataset
