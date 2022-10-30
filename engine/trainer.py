from readline import set_pre_input_hook
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
        if self.args.num_step == None:
            self.args.num_step = self.args.num_epoch * len(self.train_loader)
        optimizer = init_optimizer(self.args, self.model)

        lr_scheduler = init_scheduler(self.args, optimizer=optimizer)
        return [optimizer],[{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # images = self.attack(images, labels)
        
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)
        self.info = {'train/loss': loss, 'train/top1': top1, 'train/top5': top5[0], 'lr': self.optimizers().param_groups[0]['lr'], 'step': self.global_step}
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % self.args.pack_every == 0:
            res = self.model_hook.retrieve()
            for i, r in enumerate(res):
                
                self.info['train/entropy/layer/{}'.format(str(i).zfill(2))] = r.mean()
                self.info['train/entropy/layer_var/{}'.format(str(i).zfill(2))] = r.var()
        
        wandb.log(self.info)
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
        return


def run(args):
    logtool= WandbLogger(name="", save_dir=args.model_dir, project=args.model_dir)
    wandb.config = args
    callbacks=[
        ModelCheckpoint(save_top_k=1,mode="max",dirpath=logtool.experiment.dir, filename="ckpt-best"),
    ]
    trainer=pl.Trainer(devices="auto",
    precision=16,
    amp_backend="native",
    accelerator="cuda",
    strategy = DDPStrategy(find_unused_parameters=False),
    callbacks=callbacks,
    max_epochs=args.num_epoch,
    max_steps=args.num_steps, 
    check_val_every_n_epoch=None,
    val_check_interval=args.val_every,
    logger=logtool,
    enable_progress_bar=args.npbar
    )

    # datamodule=DataModule(args)
    model=PLModel(args)
    trainer.fit(model)
