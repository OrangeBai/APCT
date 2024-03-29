import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
from core.prune import *
from core.attack import set_attack
from core.pattern import *
from argparse import Namespace
from core.dataloader import set_dataloader, set_dataset
from models.base_model import build_model
from core.dual_net import *


class BaseTrainer(pl.LightningModule):
    def __init__(self, arg):
        """
        init base trainer
        :param arg: required name fields including:
                    (dataset, net,  project, name,
        """
        super().__init__()
        self.args = arg
        self.model = build_model(arg).cuda()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = None, None

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_loader, self.val_loader = set_dataloader(self.args)
            return
        else:
            return

    @staticmethod
    def check_valid_block(block):
        if isinstance(block, ConvBlock) or isinstance(block, LinearBlock) or isinstance(block, Bottleneck):
            return True
        return False

    def check_last_block(self, block):
        return block == self.model.layers[-1]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self, ):
        if self.args.num_step == -1:
            self.args.num_step = self.args.num_epoch * len(self.train_loader) / self.args.grad_accumulate
        optimizer = init_optimizer(self.args, self.model)

        lr_scheduler = init_scheduler(self.args, optimizer=optimizer)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)
        self.log('train/loss', loss, sync_dist=True)
        self.log('train/top1', top1, sync_dist=True)
        self.log('lr', self.lr, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        loss = self.loss_function(pred, labels)
        self.log('val/loss', loss, sync_dist=True, on_epoch=True)
        self.log('val/top1', top1, sync_dist=True, on_epoch=True)
        return

    def on_validation_epoch_end(self) -> None:
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.experiment.save('best.ckpt', policy='live')
        # exec('wandb --sync-all')
        return

    @property
    def lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        self.load_best(save_dir)
        pth_path = os.path.join(save_dir, 'model.pth')
        torch.save(self.model, pth_path)
        return

    def load_model(self, load_dir):
        pth_path = os.path.join(load_dir, 'model.pth')
        self.model = torch.load(pth_path)
        return

    def load_best(self, load_dir):
        ckpt_path = os.path.join(load_dir, 'best.ckpt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            self.model.load_weights(ckpt['state_dict'])
        return


class AttackTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.attack = set_attack(self.model, self.args)

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        images = self.attack(images, labels).detach()
        loss = super().training_step([images, labels], batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        images = self.attack(images, labels).detach()
        return super().validation_step([images, labels], batch_idx)


class EntropyTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model_hook = EntropyHook(self.model, set_gamma(self.args.activation))

    def setup(self, stage=None):
        if stage == 'fit':
            # for data split
            train_data, val_data = set_dataset(self.args)

            train_set_size = int(len(train_data) * self.args.split)
            valid_set_size = len(train_data) - train_set_size
            seed = torch.Generator().manual_seed(42)
            train_data, _ = data.random_split(train_data, [train_set_size, valid_set_size], generator=seed)
            self.train_loader, self.val_loader = set_dataloader(self.args, [train_data, val_data])

    def on_validation_epoch_start(self) -> None:
        self.model_hook.set_up()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, validation_step_outputs):
        info = {'step': self.global_step}
        res = self.model_hook.retrieve()
        avg_res, num_neuron = 0, 0
        for i, r in enumerate(res):
            avg_res += r.sum()
            num_neuron += r.size
            info['val_set/entropy_dist/layer_{}'.format(str(i).zfill(2))] = list(r)
            info['val_set/entropy_mean/layer_{}'.format(str(i).zfill(2))] = r.mean()
            info['val_set/entropy_var/layer_{}'.format(str(i).zfill(2))] = r.var()
        info['val_set/entropy_global'] = avg_res / num_neuron
        self.model_hook.remove()
        wandb.log(info)
        return

    def on_fit_end(self):
        self.test_on_train_set(-1)
        return

    def on_fit_start(self):
        self.test_on_train_set(-2)
        return

    def test_on_train_set(self, step):
        self.model_hook.set_up()
        dl = self.train_dataloader()
        self.model.eval()
        self.model = self.model.cuda()
        for x, _ in dl:
            self.model(x.cuda())
        res = self.model_hook.retrieve()
        self.model_hook.remove()
        info = {'step': step}
        avg_res, num_neuron = 0, 0
        for i, r in enumerate(res):
            avg_res += r.sum()
            num_neuron += len(r)
            info['train_set/entropy_dist/layer_{}'.format(str(i).zfill(2))] = list(r)
            info['train_set/entropy_mean/layer_{}'.format(str(i).zfill(2))] = r.mean()
            info['train_set/entropy_var/layer_{}'.format(str(i).zfill(2))] = r.var()
        info['train_set/entropy_global'] = avg_res / num_neuron
        self.model.train()
        wandb.log(info)
        return


class PruneTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model_hook = PruneHook(self.model, set_gamma(self.args.activation), 0.1)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch not in self.args.prune_milestones:
            return
        else:
            self.model_hook.set_up()

    # def training_step(self, batch, batch_idx):
    #     super().training_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.model_hook.remove()

    def on_validation_epoch_start(self) -> None:
        if self.current_epoch not in self.args.prune_milestones:
            return
        self.model_hook.set_up()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, validation_step_outputs):
        if self.current_epoch not in self.args.prune_milestones:
            return

        info = {'step': self.global_step, 'epoch': self.current_epoch}
        global_entropy = self.model_hook.retrieve(reshape=False)

        im_scores = {}
        channel_entropy = {}
        for name, block in self.model.named_modules():
            if not self.check_last_block(block) and self.check_valid_block(block):
                im_scores.update(compute_im_score(block, global_entropy[name], self.args.prune_eta))
                channel_entropy.update(compute_im_score(block, global_entropy[name], 1))

        prune_model(self.args, im_scores, global_entropy)
        monitor(im_scores, info)
        # self.model_hook.remove()

        wandb.log(info)
        return

    def save_model(self, save_dir):
        self.load_best(save_dir)
        self.iteratively_remove()
        pth_path = os.path.join(save_dir, 'model.pth')
        torch.save(self.model, pth_path)
        return

    def iteratively_remove(self):
        for name, module in self.named_modules():
            if not self.check_last_block(module) and self.check_valid_block(module):
                remove_block(module)


class DualNetTrainer(BaseTrainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.dual_net = DualNet(self.model, arg)
        self.fgsm = set_attack(self.model, Namespace(dataset=self.args.dataset, attack='fgsm', eps=4 / 255))

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        noised = images + torch.randn_like(images) * self.args.sigma
        fixed_neurons = self.dual_net.compute_fixed_2batch(images, noised)
        pred = self.dual_net(noised, fixed_neurons, self.args.eta_fixed, self.args.eta_float)
        loss = self.loss_function(pred, labels)
        top1, top5 = accuracy(pred, labels)
        self.log('train/loss', loss, sync_dist=True)
        self.log('train/top1', top1, sync_dist=True)
        self.log('lr', self.lr, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        images, labels = batch[0], batch[1]
        adv_images = self.fgsm(images, labels)
        pred = self.model(adv_images)
        top1, top5 = accuracy(pred, labels)
        self.log('val/adv_top1', top1, sync_dist=True, on_epoch=True)
        return


def set_pl_model(train_mode):
    if train_mode == 'std':
        return BaseTrainer
    if train_mode == 'adv':
        return AttackTrainer
    elif train_mode == 'exp':
        return EntropyTrainer
    elif train_mode == 'pru':
        return PruneTrainer
    elif train_mode == 'dual':
        return DualNetTrainer
    else:
        raise NameError
