<<<<<<< HEAD
from settings.test_setting import TestParser
from models.base_model import build_model
from exps.smoothed import *
from exps.text_acc import test_acc


if __name__ == '__main__':
    argsv = ['--test_name', 'smoothed_certify']
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args).cuda()
    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    # _, test_loader = set_loader(args)
    model.eval()
    smooth_test(model, args)
    test_acc(model, args)




=======
from PIL import Image

path = '/home/orange/Main/Data/ImageNet/train/n04254680/n04254680_481.JPEG'
im = Image.open(path)

sz = 160
w, h = im.size
ratio = min(h / sz, w / sz)
im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)

new_fn = '/home/orange/Main/Data/test.JPEG'
im.convert('RGB').save(new_fn)
>>>>>>> 8e9875c (.)
