import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataloader.base import *
from engine.logger import Log
from models import *


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model = build_model(args)

        self.mean, self.std = set_mean_sed(args)
        self.train_loader, self.test_loader = None, None
        self.optimizer, self.lr_scheduler = None, None
        self.loss_function = init_loss(self.args)

        self.metrics = MetricLogger()
        self.result = {'train': dict(), 'test': dict()}
        self.logger = None

        self.rank = 0
        self.world_size = 0

    def save_result(self, path, name=None):
        if not name:
            res_path = os.path.join(path, 'result')
        else:
            res_path = os.path.join(path, 'result_{}'.format(name))
        np.save(res_path, self.result)

    def record_result(self, epoch, mode='train'):

        epoch_result = {}
        for k, v in self.metrics.meters.items():
            epoch_result[k] = v.to_dict()
        self.result[mode][epoch] = epoch_result
        self.metrics.reset()
        return

    @property
    def trained_ratio(self):
        return self.lr_scheduler.last_epoch / self.args.total_step

    def warmup(self):
        if self.args.warmup_steps == 0:
            return
        loader = InfiniteLoader(self.train_loader)
        self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
        for cur_step in range(self.args.warmup_steps):
            images, labels = next(loader)
            images, labels = to_device(self.args.devices[0], images, labels)
            self.train_step(images, labels)
            if cur_step % self.args.print_every == 0 and cur_step != 0 and self.rank == 0:
                self.logger.step_logging(cur_step, self.args.warmup_steps, -1, -1, self.metrics, loader.metric)

            if cur_step >= self.args.warmup_steps:
                break
        self.logger.train_logging(-1, self.args.num_epoch, self.metrics, loader.metric)
        self.validate_epoch()
        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)

        return

    def validate_epoch(self):
        start = time.time()
        self.model.module.eval()
        for images, labels in self.test_loader:
            images, labels = images.to(self.rank), labels.to(self.rank)
            pred = self.model.module(images)
            top1, top5 = accuracy(pred, labels)
            self.metrics.update(top1=(top1, len(images)))
            # if self.args.record_lip:
            #     self.record_lip(images, labels, pred)
        self.logger.val_logging(self.metrics) + '\ttime:{0:.4f}'.format(time.time() - start)
        self.model.module.train()
        return self.metrics.meters['top1'].global_avg

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train_step(self, images, labels):
        images, labels = images.to(self.rank), labels.to(self.rank)
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        top1, top5 = accuracy(outputs, labels)
        self.metrics.update(top1=(top1, len(images)), top5=(top5, len(images)),
                            loss=(loss, len(images)), lr=(self.get_lr(), 1))

    def train_epoch(self, epoch):
        for step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.rank), labels.to(self.rank)
            self.train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0:
                self.logger.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch,
                                         self.metrics)
        self.logger.train_logging(epoch, self.args.num_epoch, self.metrics)

        return

    def train_model(self, rank, world_size):
        self.logger = Log(self.args, rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        self.rank = rank
        self._init_dataset()

        self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        if self.args.resume:
            start_epoch, best_acc = self.load_ckpt('best')
        else:
            start_epoch, best_acc = 0, 0

        # self.warmup()

        for epoch in range(start_epoch, self.args.num_epoch):
            self.train_epoch(epoch)
            self.record_result(epoch)

            self.record_result(epoch)
            acc = self.validate_epoch()
            if acc > best_acc:
                best_acc = acc
                if rank == 0:
                    self.save_ckpt(epoch, best_acc, 'best')

            self.model.train()
        self.save_result(self.args.model_dir)
        if rank == 0:
            self.save_ckpt(self.args.num_epoch, best_acc)

    def _init_dataset(self):
        train_dataset, test_dataset = set_data_set(self.args)
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler
        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            sampler=test_sampler
        )
        self.args.epoch_step = len(self.train_loader)
        self.args.total_step = self.args.num_epoch * self.args.epoch_step

        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)

        return

    def save_ckpt(self, cur_epoch, best_acc=0, name=None):
        ckpt = {
            'epoch': cur_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': best_acc
        }
        if not name:
            ckpt_path = os.path.join(self.args.model_dir, 'ckpt.pth')
        else:
            ckpt_path = os.path.join(self.args.model_dir, 'ckpt_{}.pth'.format(name))
        torch.save(ckpt, ckpt_path)
        return

    def load_ckpt(self, name):
        if not name:
            ckpt_path = os.path.join(self.args.model_dir, 'ckpt.pth')
        else:
            ckpt_path = os.path.join(self.args.model_dir, 'ckpt_{}.pth'.format(name))
        print('Trying to load CKPT from {0}'.format(ckpt_path))
        try:
            ckpt = torch.load(ckpt_path)
        except FileNotFoundError:
            print('CKPT not found, start from Epoch 0')
            return 0, 0
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        return ckpt['epoch'], ckpt['best_acc']

    # def record_lip(self, images, labels, outputs):
    #     perturbation = self.lip.attack(images, labels)
    #     local_lip = (self.model(images + perturbation) - outputs)
    #     lip_li = (local_lip.norm(p=float('inf'), dim=1) / perturbation.norm(p=float('inf'), dim=(1, 2, 3))).mean()
    #     lip_l2 = (local_lip.norm(p=2, dim=1) / perturbation.norm(p=2, dim=(1, 2, 3))).mean()
    #     self.update_metric(lip_li=(lip_li, len(images)), lip_l2=(lip_l2, len(images)))
    #     return
