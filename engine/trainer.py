import torch.utils.data as data
from attack import *
from engine.dataloader import set_dataloader
from engine.logger import Log
import torch
from core.utils import *
from models.base_model import build_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

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
        """
        Initialize optimizer:
            SGD: Implements stochastic gradient descent (optionally with momentum).
                args.momentum: momentum factor (default: 0.9)
                args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
            Adam: Implements Adam algorithm.
                args.beta_1, beta_2:
                    coefficients used for computing running averages of gradient and its square, default (0.9, 0.99)
                args.eps: term added to the denominator to improve numerical stability (default: 1e-8)
                args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
        """
        args=self.args
        total_step = self.args.num_epoch * len(self.train_loader)
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr + 1e-8, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr + 1e-8, betas=(args.beta_1, args.beta_2),
                                        weight_decay=args.weight_decay)
        else:
            raise NameError('Optimizer {} not found'.format(args.lr_scheduler))

        if args.lr_scheduler == 'milestones':
            milestones = [milestone * total_step for milestone in args.milestones]
            lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
        elif args.lr_scheduler == 'linear':
            # diff = args.lr - args.lr_e
            # LinearLR(optimizer, start_factor=args.lr, end_factor=args.lr_e, total_iters=args.num_)
            # def lambda_rule(step):
            #     return (args.lr - (step / args.total_step) * diff) / args.lr

            lr_scheduler = LLR(optimizer, lr_st=args.lr, lr_ed=args.lr_e, steps=args.total_step)

        elif args.lr_scheduler == 'exp':
            gamma = math.pow(args.lr_e / args.lr, 1 / args.total_step)
            lr_scheduler = ExponentialLR(optimizer, gamma)
        elif args.lr_scheduler == 'cyclic':
            up = int(args.total_step * args.up_ratio)
            down = int(args.total_step * args.down_ratio)
            lr_scheduler = CyclicLR(optimizer, base_lr=args.lr_e, max_lr=args.lr,
                                    step_size_up=up, step_size_down=down, mode='triangular2', cycle_momentum=False)
        elif args.lr_scheduler == 'static':
            def lambda_rule(t):
                return 1.0

            lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
        # TODO ImageNet scheduler
        else:
            raise NameError('Scheduler {0} not found'.format(args.lr_scheduler))
        return [optimizer],[lr_scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        images = self.attack(images, labels)
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)
        self.log('loss', loss,prog_bar=True)
        self.log('top1', top1,prog_bar=True)
        self.log('top5', top5,prog_bar=True)
        return loss
    def on_train_epoch(self):
        #set new scheduler
        
        self.attack.update_epoch()
    def validation_step(self, batch,batch_idx):
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        self.log('val_top1', top1,prog_bar=True, sync_dist=True)
        loss = self.loss_function(pred, labels)
        # self.log('val_loss', loss,prog_bar=True, sync_dist=True)
        return loss


    def on_epoch_end(self) -> None:
        return super().on_epoch_end()

def run(args):
    callbacks=[
        ModelCheckpoint(save_top_k=1,mode="max",dirpath=args.model_dir, filename="ckpt-best"),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(monitor="val_top1",mode="max",patience=10),
    ]
    logtool=None
    trainer=pl.Trainer(devices="auto",
    precision=16,
    amp_backend="native",
    accelerator="cuda",
    strategy = DDPStrategy(find_unused_parameters=False),
    callbacks=callbacks,
    max_epochs=args.num_epoch,
    )

    # datamodule=DataModule(args)
    model=PLModel(args)
    trainer.fit(model)
