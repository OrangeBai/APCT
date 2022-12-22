import pytorch_lightning as pl
from torch.nn.utils.prune import L1Unstructured, global_unstructured
import torch.utils.data as data
import wandb
from core.prune import *
from core.attack import set_attack
from core.pattern import *
from core.utils import *
from core.dataloader import set_dataloader, set_dataset
from models.base_model import build_model


class BaseTrainer(pl.LightningModule):
    def __init__(self, arg):
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

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self, ):
        if self.args.num_step == -1:
            self.args.num_step = self.args.num_epoch * len(self.train_loader)
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

    @property
    def lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)


class AttackTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.attack = set_attack(self.model, self.args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        images = self.attack(images, labels)
        loss = super().training_step([images, labels], batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        self.log('val/adv_top1', top1, sync_dist=True, on_epoch=True)
        return


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
        self.model_hook = PruneHook(self.model, set_gamma(self.args.activation))

    def on_validation_epoch_start(self) -> None:
        if not (self.current_epoch % self.args.prune_every == 0 and self.current_epoch > 0):
            return
        self.model_hook.set_up()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, validation_step_outputs):
        if not (self.current_epoch % self.args.prune_every == 0 and self.current_epoch > 0):
            return
        info = {'step': self.global_step}
        res = self.model_hook.retrieve(reshape=False)
        parameters_to_prune = []
        importance_dict = {}
        # torch.nn.utils.prune.global_unstructured()

        for name, block in self.model.named_modules():
            if not check_valid_block(block):
                continue
            compute_params(block, res[name], parameters_to_prune, importance_dict, self.args.prune_eta)

        prune_model(parameters_to_prune, importance_dict, self.args)

        pruned = 0
        nele = 0

        for module in importance_dict.keys():
            pruned += torch.sum(module.weight == 0)
            nele += module.weight.nelement()
        print("Global sparsity: {:.2f}%".format(pruned/nele))
        self.model_hook.remove()
        wandb.log(info)
        return


def set_pl_model(train_mode):
    if train_mode == 'std':
        return BaseTrainer
    elif train_mode == 'adv':
        return AttackTrainer
    elif train_mode == 'exp':
        return EntropyTrainer
    elif train_mode == 'pru':
        return PruneTrainer
    else:
        raise NameError
