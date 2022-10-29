<<<<<<< HEAD
from readline import set_pre_input_hook
=======
from tkinter.tix import Tree
>>>>>>> 3078b1d (.)
import torch.utils.data as data
from pytorch_lightning.loggers import CSVLogger
from attack import *
from core.pattern import *
from engine.dataloader import set_dataloader, set_dataset
from engine.logger import Log
import torch
from core.utils import *
from models.base_model import build_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import math

# DDP version
class PLModel(pl.LightningModule):
    def __init__(self, args):
        self.args = args
        #self._init_dataset()
        super().__init__()
        self.model = build_model(args)
        self.attack = set_attack(self.model, self.args)
        self.loss_function = torch.nn.CrossEntropyLoss()
        #self.start_epoch, self.best_acc = self.resume()
        self.model_hook = BaseHook(self.model, set_output_hook, set_gamma(self.args.activation))


        train_data, self.val_data = set_dataset(self.args)

        train_set_size = int(len(train_data) * self.args.split)
        valid_set_size = len(train_data) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_data, _ = data.random_split(train_data, [train_set_size, valid_set_size], generator=seed)

    def setup(self, stage):
        if stage == 'fit':
            self.train_loader, self.val_loader = set_dataloader(self.args, [self.train_data, self.val_data])
            return 
        else:
            return
        
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
        
    def configure_optimizers(self,):
        self.args.total_step = self.args.num_epoch * len(self.train_loader)
        optimizer = init_optimizer(self.args, self.model)

        lr_scheduler = init_scheduler(self.args, optimizer=optimizer)
        return [optimizer],[{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # images = self.attack(images, labels)
        
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)
<<<<<<< HEAD
        self.info = {'train/loss': loss, 'train/top1': top1, 'train/top5': top5[0], 'lr': self.optimizers().param_groups[0]['lr'], 'step': self.global_step}
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % self.args.pack_every == 0:
            res = self.model_hook.retrieve()
            for i, r in enumerate(res):
                
                self.info['train/entropy/layer/{}'.format(str(i).zfill(2))] = r.mean()
                self.info['train/entropy/layer_var/{}'.format(str(i).zfill(2))] = r.var()
        
        wandb.log(self.info)
=======
        self.metrics.update(
            top1=(top1, len(images)), top5=(top5, len(images)),
            loss=(loss, len(images)),
            lr=(self.get_lr(), 1)
        )
        self.metrics.all_reduce()

    # def warmup(self):
    #     if self.args.warmup_steps == 0:
    #         return
    #     loader = InfiniteLoader(self.train_loader)
    #     self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
    #     for cur_step in range(self.args.warmup_steps):
    #         images, labels = next(loader)
    #         images, labels = to_device(self.args.devices[0], images, labels)
    #         # self.train_step(images, labels)
    #         if cur_step % self.args.print_every == 0 and cur_step != 0 and self.rank == 0:
    #             self.logger.step_logging(cur_step, self.args.warmup_steps, -1, -1, self.metrics, loader.metric)
    #
    #         if cur_step >= self.args.warmup_steps:
    #             break
    #     self.logger.train_logging(-1, self.args.num_epoch, self.metrics, loader.metric)
    #     self.validate_epoch()
    #     self.optimizer = init_optimizer(self.args, self.model)
    #     self.lr_scheduler = init_scheduler(self.args, self.optimizer)
    #
    #     return

    def train_epoch(self, epoch):
        cur_time = time.time()
        for step, (images, labels) in enumerate(self.train_loader):
            data_time = time.time() - cur_time
            images, labels = images.to(self.rank, non_blocking=True), labels.to(self.rank, non_blocking=True)
            self.train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0 and self.rank == 0:
                self.logger.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch,
                                         self.metrics, self.time_metric)

            iter_time = time.time() - cur_time

            self.time_metric.update(iter_time=(iter_time, 1), data_time=(data_time, 1))
            self.time_metric.all_reduce()
            self.metrics.all_reduce()
            cur_time = time.time()
        if self.rank == 0:
            self.logger.train_logging(epoch, self.args.num_epoch, self.metrics, self.time_metric)
        self.time_metric.reset()
>>>>>>> 3078b1d (.)
        return



    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        batch_size = len(images)
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        loss = self.loss_function(pred, labels)
        
        return {'batch_size': batch_size, 'loss': loss * batch_size, 
                'top1':top1[0] * batch_size, 'top5': top5[0] * batch_size}
    
    def validation_step_end(self, batch_parts):
        # predictions from each GPU
        batch_output = {}
        for key, val in batch_parts.items():
            batch_output[key] = val
        return batch_output

<<<<<<< HEAD
    def validation_epoch_end(self, validation_step_outputs) -> None:
        epoch_output = {}
        for out in validation_step_outputs:
            for k, v in out.items():
                if k not in epoch_output.keys():
                    epoch_output[k] = v
                else:
                    epoch_output[k] += v
        info = {'step': self.global_step}
        for k, v in epoch_output.items():
            if k != 'batch_size':
                info['val/'+k] = v / epoch_output['batch_size']
        res = self.model_hook.retrieve()
        for i, r in enumerate(res):
            info['val/entropy_layer_{}'.format(str(i).zfill(2))] = list(r)
        wandb.log(info)
=======
        # self.warmup()

        for epoch in range(self.start_epoch, self.args.num_epoch):
            self.reset_lr_dt(epoch)
            self.train_epoch(epoch)
            self.record_result(epoch)

            # acc = self.validate_epoch()
            # if acc > self.best_acc:
            #     self.best_acc = acc
            #     if self.rank == 0:
            #         self.save_ckpt(epoch + 1, self.best_acc, 'best')

        if self.rank == 0:
            if self.args.save_name == '':
                self.args.save_name = 'epoch_{}'.format(str(self.args.num_epoch).zfill(3))
            self.save_result(self.args.model_dir, self.args.save_name)
            self.save_ckpt(self.args.num_epoch, self.best_acc, self.args.save_name)

    def _init_dataset(self):
        train_dataset, test_dataset = set_data_set(self.args)
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.test_sampler = DistributedSampler(test_dataset, shuffle=True)
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler

        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            sampler=self.test_sampler
        )

        self.args.epoch_step = len(self.train_loader)
        self.args.total_step = self.args.num_epoch * self.args.epoch_step
>>>>>>> 3078b1d (.)
        return


def run(args):
    callbacks=[
        ModelCheckpoint(save_top_k=1,mode="max",dirpath=args.model_dir, filename="ckpt-best"),
    ]
    logtool= WandbLogger(name="", save_dir=args.model_dir)
    trainer=pl.Trainer(devices="auto",
    precision=16,
    amp_backend="native",
    accelerator="cuda",
    strategy = DDPStrategy(find_unused_parameters=False),
    callbacks=callbacks,
    max_epochs=args.num_epoch,
    check_val_every_n_epoch=None,
    val_check_interval=200,
    logger=logtool,
    )

<<<<<<< HEAD
    # datamodule=DataModule(args)
    model=PLModel(args)
    trainer.fit(model)
=======
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

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    # def record_lip(self, images, labels, outputs):
    #     perturbation = self.lip.attack(images, labels)
    #     local_lip = (self.model(images + perturbation) - outputs)
    #     lip_li = (local_lip.norm(p=float('inf'), dim=1) / perturbation.norm(p=float('inf'), dim=(1, 2, 3))).mean()
    #     lip_l2 = (local_lip.norm(p=2, dim=1) / perturbation.norm(p=2, dim=(1, 2, 3))).mean()
    #     self.update_metric(lip_li=(lip_li, len(images)), lip_l2=(lip_l2, len(images)))
    #     return

    def _init_functions(self):
        self.optimizer = init_optimizer(self.args, self.model)
        self.lr_scheduler = init_scheduler(self.args, self.optimizer)

        self.loss_function = init_loss(self.args)

    def reset_lr_dt(self, epoch):
        if self.args.dataset != 'imagenet':
            if epoch == 0:
                self._init_dataset()
                self._init_functions()
            else:
                return

        if epoch == 0:

            with open(self.args.phase_path, 'r') as f:
                self.args.phase_file = yaml.load(f, Loader=yaml.FullLoader)
            cur_p, cur_file = check_phase(self.args.phase_file, epoch)

            self.args.data_size = cur_file['data_size']
            self.args.crop_size = cur_file['crop_size']
            self.args.batch_size = cur_file['batch_size']
            self._init_dataset()

            self.args.lr_scheduler = cur_file['lr_scheduler']
            self.args.lr = cur_file['lr']
            self.args.lr_e = cur_file['lr_e']
            self.args.total_step = (cur_file['end_epoch'] - cur_file['start_epoch']) * len(self.train_loader)

            self.logger.info('Switching to  {0}, with info {1}'.format(cur_p, cur_file))

            self._init_functions()
        else:
            cur_p, cur_file = check_phase(self.args.phase_file, epoch)
            pre_p, pre_file = check_phase(self.args.phase_file, epoch - 1)
            if pre_p == cur_p:
                return
            self.logger.info('Switching to  {0}, with info {1}'.format(cur_p, cur_file))
            if pre_file['data_size'] != cur_file['data_size']:
                self.args.data_size = cur_file['data_size']
                self.args.crop_size = cur_file['crop_size']
                self.args.batch_size = cur_file['batch_size']
                self._init_dataset()
                self.logger.info('Dataset Initialized')

            if cur_p != pre_p:
                self.args.lr_scheduler = cur_file['lr_scheduler']
                self.args.lr = cur_file['lr']
                self.args.lr_e = cur_file['lr_e']
                self.args.total_step = (cur_file['end_epoch'] - cur_file['start_epoch']) * len(self.train_loader)
                self._init_functions()
                self.logger.info('Optimizer Initialized')
        return
>>>>>>> 3078b1d (.)
