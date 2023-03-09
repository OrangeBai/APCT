# import torch.nn
#
# from config import *
# from exps.smoothed import *
# from models.base_model import build_model
# from settings.test_setting import TestParser
# from exps.text_acc import *

# ############################# code for lip ####################################################
# def local_lip(argsv, sigma):
#     batch_size = 64
#     args = TestParser(argsv).get_args()
#     model = build_model(args).cuda()
#
#     _, test_dataset = set_data_set(args)
#
#     ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
#     model.load_weights(ckpt['model_state_dict'])
#     # float_hook = ModelHook(model, set_pattern_hook, [0])
#     mean, std = [torch.tensor(d).view(len(d), 1, 1) for d in set_mean_sed(args)]
#     model.eval()
#     all_flt = []
#     # test_acc(model, args)
#     m, v = [], []
#     for i in range(0, len(test_dataset), 100):
#         x, y, = test_dataset[i]
#         x, y = x.cuda(), torch.tensor(y).cuda()
#         batch = x.repeat((batch_size, 1, 1, 1))
#         label = y.repeat((batch_size, 1))
#         signal = torch.sign(torch.randn_like(batch).to(x.device))
#         n = signal / signal.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1) * sigma
#         n[0] = 0
#         batch_x = batch + n
#         batch_x = (batch_x - mean.to(batch_x.device)) / std.to(batch_x.device)
#         batch_x.requires_grad = True
#         pred = model(batch_x)
#         loss = torch.nn.CrossEntropyLoss()(pred, label.squeeze())
#         grad = torch.autograd.grad(loss, batch_x, retain_graph=False, create_graph=False)[0]
#         grad_view = grad.view(batch_size, -1)
#         pred_2 = model(batch_x + grad)
#
#         p = (pred_2 - pred).norm(p=2, dim=1) / grad_view.norm(p=2, dim=1)
#         m.append(p.mean().cpu().detach().numpy())
#         v.append(p.var().cpu().detach().numpy())
#         print(1)
#     return m, v
#
#     #     x, _ = test_dataset[i]
#     #     x = x.cuda()
#     #     x =
#     #
#     #     torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
#
#
# if __name__ == '__main__':
#     a = [
#         ['--dataset', 'cifar10', '--exp_id', 'std', '--model_type', 'mini', '--net', 'vgg16', '--data_size', '352'],
#         ['--dataset', 'cifar10', '--exp_id', 'noise_005', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
#          '352'],
#         ['--dataset', 'cifar10', '--exp_id', 'noise_010', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
#          '352'],
#         ['--dataset', 'cifar10', '--exp_id', 'noise_025', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
#          '352']
#     ]
#     xx = []
#
#     for b in a:
#         x = []
#         for s in range(1, 17):
#             x.append(local_lip(b, s))
#         xx.append(x)
#     print(1)
#     exp_dir = os.path.join(MODEL_PATH, 'exp')
#     os.makedirs(exp_dir, exist_ok=True)
#     n = np.array(xx)
#     np.save(os.path.join(exp_dir, 'lip'), n)
#
#
# ################################ smoothed ##################################
#
# import datetime
# import os
# import torch
# from core.scrfp import SCRFP, Smooth, ApproximateAccuracy
# from core.dataloader import *
#
#
# def smooth_test(model, args):
#     file_path = os.path.join(args.exp_dir, '_'.join([args.method, str(args.N0), str(args.N), str(args.sigma_2), str(args.eta_float)]))
#     smooth_pred(model, args)
#
#     certify_res = ApproximateAccuracy(file_path).at_radii(np.linspace(0, 1, 256))
#     output_path = os.path.join(args.exp_dir, file_path + '_cert.npy')
#     print(certify_res.mean())
#     np.save(output_path, certify_res)
#     return
#
#
# def smooth_pred(model, args):
#     if args.method == 'SMRAP':
#         smoothed_classifier = SCRFP(model, args)
#     else:
#         smoothed_classifier = Smooth(model, args)
#
#     # prepare output file
#     file_path = os.path.join(args.exp_dir, '_'.join([args.method, str(args.N0), str(args.N), str(args.sigma_2), str(args.eta_float)]))
#     f = open(file_path, 'w')
#     print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
#
#     # iterate through the dataset
#     if args.dataset.lower() == 'imagenet':
#         dataset = get_val(args)
#     else:
#         _, dataset = set_data_set(args)
#     for i in range(len(dataset)):
#
#         # only certify every args.skip examples, and stop after args.max examples
#         if i % args.skip != 0:
#             continue
#         if i == -1:
#             break
#
#         (x, label) = dataset[i]
#
#         before_time = time.time()
#         # certify the prediction of g around x
#         x = x.cuda()
#         with torch.cuda.amp.autocast(dtype=torch.float16):
#             prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.smooth_alpha, args.batch_size)
#         after_time = time.time()
#         correct = int(prediction == label)
#
#         time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
#         print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
#             i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
#     torch.argmax
#     f.close()
################################### flt nums #############################
#
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# sns.set()
# sns.set_theme(style="darkgrid")
# from config import *
# import pandas as pd
#
# large = 22
# med = 16
# small = 12
# params = {'axes.titlesize': large,
#           'legend.fontsize': large,
#           'figure.figsize': (12, 8),
#           'axes.labelsize': large,
#           'xtick.labelsize': large,
#           'ytick.labelsize': large,
#
#           'figure.titlesize': large}
# plt.rcParams.update(params)
# fig, ax = plt.subplots()
# np_file_0 = os.path.join(MODEL_PATH, 'exp', 'float_ratio_vgg_sigma_0.1.npy')
# np_file_1 = os.path.join(MODEL_PATH, 'exp', 'float_ratio_vgg_sigma_0.25.npy')
# flt0 = np.load(np_file_0)
# flt1 = np.load(np_file_1)
#
# d = flt0.mean(axis=2).T
# df = pd.DataFrame(data=d, columns=['STD', r'$\sigma$-0.05', r'$\sigma$-0.10', r'$\sigma$-0.25'])
# sns.violinplot(data=df, split=True, ax=ax)
#
# # fig, ax = plt.subplots(nrows=1, ncols=4)
# # for i in range(4):
# #     data = pd.DataFrame(
# #         {'0.1': flt0[i].mean(axis=1), '0.25': flt1[i].mean(axis=1)})
# #
# #     sns.violinplot(data=data, ax=ax[i])
# #     # ax[i].violinplot(data=data, split=True)
# #     ax[i].set_ylim(0.6, 0.97)
# # ax.set_xticks(['STD', '$\sigma$-0.10', '$\sigma$-0.05', '$\sigma$-0.05'])
# ax.set_ylabel('#fixed neuron / # neurons')
# ax.set_xlabel('Model')
# plt.show()
