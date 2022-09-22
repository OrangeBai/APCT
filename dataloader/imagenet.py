import os
import random

import h5py
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import *

from config import *

IMAGENET_MEAN_STD = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]


class H5PYImageNet(Dataset):
    def __init__(self, file_dir, mode, transform):

        self.dataset = None
        self.transform = transform
        if mode == 'train':
            self.file_path = os.path.join(file_dir, 'train.hdf5')
        else:
            self.file_path = os.path.join(file_dir, 'val.hdf5')

        self.f = h5py.File(self.file_path, 'r')
        self.data = self.f['data']
        self.label = self.f['label']

    def shuffle(self):
        new_idx = list(range(len(self.data)))
        random.shuffle(new_idx)
        self.data = self.data[new_idx]
        self.label = self.label[new_idx]

    def __getitem__(self, index: int):
        sample = self.data[index]
        sample = self.transform(sample)
        target = self.label[index]
        return sample, target

    def __len__(self):
        return len(self.f['label'])


def get_dataset(args):
    mean, std = IMAGENET_MEAN_STD
    resize_path = os.path.join(DATA_PATH, 'ImageNet-sz', str(args.data_size))
    resize_exist = os.path.exists(resize_path)
    if resize_exist:
        data_dir = resize_path
    else:
        data_dir = os.path.join(DATA_PATH, 'ImageNet')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if resize_exist:
        train_composed = [ToTensor(), Resize((args.data_size, args.data_size)), transforms.Normalize(mean, std), 
                          CenterCrop((args.crop_size, args.crop_size)), RandomHorizontalFlip()]
        val_composed = [ToTensor(), Resize((args.data_size, args.data_size)),
                        CenterCrop((args.crop_size, args.crop_size))]
    else:
        train_composed = [ToTensor(), CenterCrop((args.crop_size, args.crop_size)), transforms.Normalize(mean, std), RandomHorizontalFlip()]
        val_composed = [ToTensor(), CenterCrop((args.crop_size, args.crop_size))]
    train_transform = Compose(train_composed)
    val_transform = Compose(val_composed)

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(val_dir, transform=val_transform)

    # train_dataset = H5PYImageNet(data_dir, 'train', transform=train_transform)
    # test_dataset = H5PYImageNet(data_dir, 'val', transform=val_transform)
    return train_dataset, test_dataset


def get_loaders(args):
    train_dataset, test_dataset = get_dataset(args)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        prefetch_factor=8
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader, test_loader

# class ValLoader(datasets.Dataset):
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
