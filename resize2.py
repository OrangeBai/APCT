import datetime
import multiprocessing
import os
import time
from functools import partial
from config import *
from PIL import Image

SRC = os.path.join(DATA_PATH, 'ImageNet')
# DST = r'/home/orange/Main/Data/ImageNet-sz'
DST = os.path.join(DATA_PATH, 'ImageNet-sz')
# DST = r'/home/orange/Main/Data/ImageNet-sz'
SIZE = [160, 352]

num_cpus = multiprocessing.cpu_count()


def resize_folder(src_folder, dst_folders, sizes):
    [os.makedirs(dst_folder, exist_ok=True) for dst_folder in dst_folders]
    img_names = os.listdir(src_folder)
    pool = multiprocessing.Pool(num_cpus)
    # for img in img_names:
    #     resize_img(img, src_folder, dst_folders, sizes)
    pool.map(partial(resize_img, s_folder=src_folder, d_folders=dst_folders, sizes=sizes), img_names)
    return


def resize_img(n, s_folder, d_folders, sizes):
    img = Image.open(os.path.join(s_folder, n))
    img = img.convert('RGB')
    w, h = img.size
    for d_folder, size in zip(d_folders, sizes):
        ratio = min(h / size, w / size)
        im = img.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
        im.save(os.path.join(d_folder, n))


def resize_category(src, dst, cat_name):
    s_dir = os.path.join(src, 'train', cat_name)
    d_dirs = [os.path.join(dst, str(sz), 'train', cat_name) for sz in SIZE]
    resize_folder(s_dir, d_dirs, SIZE)

    s_dir = os.path.join(src, 'val', cat_name)
    d_dirs = [os.path.join(dst, str(sz), 'val', cat_name) for sz in SIZE]
    resize_folder(s_dir, d_dirs, SIZE)


if __name__ == '__main__':
    train_src = os.path.join(SRC, 'train')
    val_src = os.path.join(SRC, 'val')

    cats_train, cats_val = os.listdir(train_src), os.listdir(val_src)
    assert len(set(cats_val).difference(set(cats_train))) == 0
    for i, cat in enumerate(cats_train):
        t = time.time()
        resize_category(SRC, DST, cat)

        time_spent = time.time() - t
        eta = time_spent * (1000 - i) / 60
        print('[{0}]{1:4d} / 1000, time: {2}, eta: {3}'.format(datetime.datetime.now(), i, time.time() - t, eta))
