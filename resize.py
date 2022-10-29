import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image
num_cpus = multiprocessing.cpu_count()


PATH = Path('/home/orange/Main/Data/ImageNet')
# DEST = Path('/mnt/ram')
DEST = Path('/home/orange/Main/Data/ImageNet-sz')
# szs = (int(128*1.25), int(256*1.25))
# szs = (int(160*1.25),)
szs = (160, 352)


def resize_img(p, im, fn, sz):
    w, h = im.size
    ratio = min(h / sz, w / sz)
    im = im.convert('RBG')
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
    # import pdb; pdb.set_trace()
    new_fn = DEST / str(sz) / fn.relative_to(PATH)
    new_fn.parent.mkdir(exist_ok=True)
    im.save(new_fn)
    pool.map(process, instance_with_idx)


def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs: resize_img(p, im, fn, sz)


def resize_imgs(p):
    files = p.glob('*/*.JPEG')
    # list(map(partial(resizes, p), files))
    with ProcessPoolExecutor(cpus) as e: e.map(partial(resizes, p), files)


for sz in szs:
    ssz = str(sz)
    (DEST / ssz).mkdir(exist_ok=True)
    for ds in ('val', 'train'): (DEST / ssz / ds).mkdir(exist_ok=True)

for ds in ('val', 'train'): resize_imgs(PATH / ds)
