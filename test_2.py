from settings.test_setting import TestParser
from models.base_model import build_model
from core.pattern import *
from exps.smoothed import *
from exps.text_acc import *

if __name__ == '__main__':
    argsv = ['--dataset', 'imagenet', '--exp_id', 'noise_000', '--model_type', 'net', '--test_name', 'smoothed_certify',
             '--net', 'resnet50', '--activation', 'ReLU', '--data_size', '256', '--crop_size', '224', '--method', 'STD', '--batch_size', '500', 
            '--N0', '50', '--N', '10000', '--skip', '25', '--sigma_2', '0.25'
             ]
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args).cuda()
    # model = resnet50().cuda()
    # torch.load(os.path.join(args.model_dir, ''))
    ckpt = torch.load(r'/home/user/Orange/ICLR/imagenet/pgd_1step/resnet50/noise_0.25/checkpoint.pth.tar')


    model = load_weight(model, ckpt['state_dict'])
    # model.load_state_dict(ckpt)
    # _, test_loader = set_loader(args)
    # model = nn.Sequential(NormalizeLayer(*set_mean_sed(args)), model)
    model = nn.Sequential(InputCenterLayer(set_mean_sed(args)[0]), model)
    model.eval()
    # test_acc(model, args)
    smooth_test(model, args)


    # metrics = MetricLogger()
    # for images, labels in test_loader:
    #     images, labels = images.to(2, non_blocking=True), labels.to(2, non_blocking=True)
    #     # with torch.no_grad():
    #     # print(images.shape)
    #     with torch.cuda.amp.autocast(dtype=torch.float16):
    #         pred = model(images)
    #
    #     top1, top5 = accuracy(pred, labels)
    #     metrics.update(top1=(top1, len(images)))



