from readline import set_pre_input_hook
import torch.utils.data as data
from pytorch_lightning.loggers import CSVLogger
from attack import *
from core.pattern import *
from engine.dataloader import set_dataloader
from engine.logger import Log
import torch
from core.utils import *
from models.base_model import build_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
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
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_loader, self.val_loader = set_dataloader(self.args)
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
        return [optimizer],[lr_scheduler]

    # def on_train_start(self):
        
    #     self.float_entropy = []
    #     return super().on_train_start()

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # images = self.attack(images, labels)
        
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        self.lr_schedulers().step()
        top1, top5 = accuracy(outputs, labels)
        info = {'loss': loss, 'top1': top1, 'top5': top5[0], 'lr': self.optimizers().param_groups[0]['lr']}
        if batch_idx % 100 == 0:
            self.model_hook = BaseHook(self.model, set_output_hook, set_gamma(self.args.activation))
        if batch_idx % 100 == 10:
            res = self.model_hook.retrieve()
            self.model_hook.remove()
            for i, r in enumerate(res):
                info['entropy_layer_{}'.format(str(i).zfill(2))] = list(r)
        wandb.log(info)
        return loss
    # def on_train_epoch(self):
    #     #set new scheduler
    #     # self.attack.update_epoch()
    #     pass

    def validation_step(self, batch,batch_idx):
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        self.log('val_top1', top1, sync_dist=True, on_epoch=True)
        loss = self.loss_function(pred, labels)
        self.log('val_loss', loss, sync_dist=True)
        return loss


    def on_train_end(self) -> None:
        pass

def run(args):
    callbacks=[
        ModelCheckpoint(save_top_k=1,mode="max",dirpath=args.model_dir, filename="ckpt-best"),
        # LearningRateMonitor(logging_interval='step')
    ]
    logtool= CSVLogger(args.model_dir, name="")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.model_dir)
    trainer=pl.Trainer(devices="auto",
    precision=16,
    amp_backend="native",
    accelerator="cuda",
    strategy = DDPStrategy(find_unused_parameters=False),
    callbacks=callbacks,
    max_epochs=args.num_epoch,
    logger=tb_logger
    )

    # datamodule=DataModule(args)
    model=PLModel(args)
    trainer.fit(model)
