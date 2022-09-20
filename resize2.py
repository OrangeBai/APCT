import multiprocessing
import os
import time
from functools import partial

from PIL import Image

SRC = r'/home/orange/Main/Data/ImageNet'
# DST = r'/home/orange/Main/Data/ImageNet-sz'
DST = r'/home/orange/Main/Data/ImageNet-ss'
SIZE = [160, 352]

num_cpus = multiprocessing.cpu_count()


def resize_folder(src_folder, dst_folder, size):
    os.makedirs(dst_folder, exist_ok=True)
    img_names = os.listdir(src_folder)
    pool = multiprocessing.Pool(num_cpus)
    pool.map(partial(resize_img, s_folder=src_folder, d_folder=dst_folder, size=size), img_names)
    return


def resize_img(n, s_folder, d_folder, size):
    img = Image.open(os.path.join(s_folder, n))
    im = img.convert('RGB')
    w, h = img.size
    ratio = min(h / size, w / size)
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
    im.save(os.path.join(d_folder, n))


def resize_category(src, dst, cat_name):
    for sz in SIZE:
        s_dir = os.path.join(src, 'train', cat_name)
        d_dir = os.path.join(dst, str(sz), 'train', cat_name)
        resize_folder(s_dir, d_dir, 160)

        s_dir = os.path.join(src, 'val', cat_name)
        d_dir = os.path.join(dst, str(sz), 'val', cat_name)
        resize_folder(s_dir, d_dir, 160)


if __name__ == '__main__':
    train_src = os.path.join(SRC, 'train')
    val_src = os.path.join(SRC, 'val')

    cats_train, cats_val = os.listdir(train_src), os.listdir(val_src)
    assert len(set(cats_val).difference(set(cats_train))) == 0
    for i, cat in enumerate(cats_train):
        t = time.time()
        resize_category(SRC, DST, cat)

        print('{0:4d} / 1000, time: {1}'.format(i, time.time() - t))
