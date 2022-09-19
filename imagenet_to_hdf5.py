import multiprocessing
import os
import random
import time
import warnings
import argparse
import h5py
import numpy as np
from PIL import Image
from torchvision.datasets.folder import find_classes
from torchvision.transforms import ToTensor

from config import *

warnings.filterwarnings("error")

image_size = 299
block_size = 1000


def retrieve_index_and_path(directory):
    _, class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances


def create_h5py(path, name):
    blocks = [path[i:i + block_size] for i in range(0, len(path), block_size)]

    f = h5py.File(os.path.join(DATA_PATH, 'ImageNet', name + '.hdf5'), 'w')
    f.create_dataset('data', shape=(0, image_size, image_size, 3), maxshape=(len(path), image_size, image_size, 3),
                     dtype=np.uint8, chunks=True)
    f.create_dataset('label', shape=(0,), maxshape=(len(path),), dtype=np.uint8, chunks=True)
    train_num = 0
    for block_idx, block in enumerate(blocks):
        t1 = time.time()
        instance_with_idx = zip(range(len(block)), block)
        pool = multiprocessing.Pool(num_cpus)
        img_batch, label_batch = zip(*[i for i in pool.map(process, instance_with_idx) if i is not None])
        pool.close()

        valid = len(img_batch)

        f['data'].resize(train_num + valid, axis=0)
        f['label'].resize(train_num + valid, axis=0)
        f['data'][train_num:train_num + valid] = img_batch
        f['label'][train_num:train_num + valid] = label_batch
        train_num += valid
        time_spent = time.time() - t1
        eta = time_spent * (len(blocks) - block_idx) / 60
        print('{0:4d}/{1:4d}: time spent: {2:2f}, eta:{3:2f}'.format(block_idx, len(blocks), time_spent, eta))
    f.close()


def process(input_tuple):
    idx, (path, label) = input_tuple
    try:
        im = Image.open(path).convert('RGB').resize((image_size, image_size))
        # img = (ToTensor()(im).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        img = ToTensor()(im).cuda()
        lb = label
    except Exception as e:
        print('Find Exception {0} at {1}'.format(e, path))
        os.remove(path)
        print('Deleting File {0}'.format(path))
        return
    return img, lb


def clean_dataset(path):
    blocks = [path[i:i + block_size] for i in range(0, len(path), block_size)]

    for block_idx, block in enumerate(blocks):
        t1 = time.time()
        instance_with_idx = zip(range(len(block)), block)
        pool = multiprocessing.Pool(num_cpus)
        zip(*[i for i in pool.map(process, instance_with_idx) if i is not None])
        pool.close()

        time_spent = time.time() - t1
        eta = time_spent * (len(blocks) - block_idx) / 60
        print('{0:4d}/{1:4d}: time spent: {2:2f}, eta:{3:2f}'.format(block_idx, len(blocks), time_spent, eta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='clean', choices=['clean', 'h5py'])
    args = parser.parse_args()

    num_cpus = multiprocessing.cpu_count()

    train_dir = os.path.join(DATA_PATH, 'ImageNet', 'train')
    val_dir = os.path.join(DATA_PATH, 'ImageNet', 'val')

    train_path = retrieve_index_and_path(train_dir)
    val_path = retrieve_index_and_path(val_dir)
    if args.mode == 'clean':
        print('Clean dataset:')
        clean_dataset(train_path)
        clean_dataset(val_path)
    else:
        random.shuffle(train_path)
        random.shuffle(val_path)

        create_h5py(train_path, 'train')
        create_h5py(val_path, 'val')

    print(1)

