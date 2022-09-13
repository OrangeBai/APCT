import numpy as np
import h5py
import os
from config import *
from torchvision.transforms import ToTensor
import multiprocessing
from PIL import Image
batch_size = 100000
image_size = 256
num_cpus = multiprocessing.cpu_count()


def process(f):
    global image_size
    im = Image.open(f)
    im = im.resize((image_size, image_size))
    return im


prefix = os.path.join(DATA_PATH, 'ImageNet', 'train')
cats = os.listdir(prefix)
l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

f = h5py.File(os.path.join(DATA_PATH, 'ImageNet', 'h5py.hdf5'), "w")

for syn in l:
    files = os.listdir(syn)
    for file in files:
        im = Image.open(os.path.join(syn, files[0])).resize((256, 256))
        a = ToTensor()(im).numpy()

print(1)


i = 0
imagenet = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
pool = multiprocessing.Pool(num_cpus)
while i < len(l):
    current_batch = l[i:i + batch_size]
    current_res = np.array(pool.map(process, current_batch))
    imagenet[i:i + batch_size] = current_res
    i += batch_size
    print(i, 'images')