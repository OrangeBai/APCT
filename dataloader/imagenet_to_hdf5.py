import numpy as np
import h5py
import os
from config import *
from torchvision.transforms import ToTensor
import multiprocessing
from PIL import Image
from torchvision.datasets.folder import find_classes
image_size = 299


def process(iii):
    idx, (path, label) = iii
    im = Image.open(path).resize((image_size, image_size))
    img = (ToTensor()(im).numpy() * 255).astype(np.ubyte)
    lb = label
    if idx == 100:
        return
    return img, lb


if __name__ == '__main__':
    batch_size = 100000
    num_cpus = multiprocessing.cpu_count()

    directory = os.path.join(DATA_PATH, 'ImageNet', 'train')
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

    # cats = os.listdir(prefix)
    # l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

    # for idx, (path, label) in enumerate(instances):
    #     valid.append(idx)
    instance_with_idx = zip(range(len(instances)), instances)
    pool = multiprocessing.Pool(num_cpus)
    c1, c2 = zip(*[i for i in pool.map(process, instance_with_idx) if i is not None])

    f = h5py.File(os.path.join(DATA_PATH,'ImageNet', 'h5py.hdf5'))

    dtst = f.create_dataset('train', maxshape=(None, 3, image_size, image_size),dtype=np.uint8)
    dtst

# f.close()
    print(1)
# for syn in l:
#     files = os.listdir(syn)
#     for file in files:

# def process(f):
#     global image_size
#     im = Image.open(f)
#     im = im.resize((image_size, image_size))
#     return im
# print(1)
#
#
# i = 0
# imagenet = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
# pool = multiprocessing.Pool(num_cpus)
# while i < len(l):
#     current_batch = l[i:i + batch_size]
#     current_res = np.array(pool.map(process, current_batch))
#     imagenet[i:i + batch_size] = current_res
#     i += batch_size
#     print(i, 'images')
