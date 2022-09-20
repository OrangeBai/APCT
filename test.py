from PIL import Image

path = '/home/orange/Main/Data/ImageNet/train/n04254680/n04254680_481.JPEG'
im = Image.open(path)

sz = 160
w, h = im.size
ratio = min(h / sz, w / sz)
im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)

new_fn = '/home/orange/Main/Data/test.JPEG'
im.convert('RGB').save(new_fn)
