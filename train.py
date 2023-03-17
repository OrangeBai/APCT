from settings.train_settings import TrainParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import os
from core.trainer import set_pl_model
import pytorch_lightning as pl


if __name__ == '__main__':
    args = TrainParser().get_args()

    logtool = WandbLogger(name=args.name, save_dir=args.model_dir, project=args.project, config=args,
                          version=args.resume_id)

    # datamodule=DataModule(args)
    model = set_pl_model(args.train_mode)(args)

    callbacks = [
        ModelCheckpoint(monitor='val/top1', save_top_k=1, mode="max", save_on_train_epoch_end=False,
                        dirpath=logtool.experiment.dir, filename="best"),
    ]

    trainer = pl.Trainer(devices="auto",
                         precision=32,
                         amp_backend="native",
                         accelerator="cuda",
                         strategy='dp',
                         callbacks=callbacks,
                         max_epochs=args.num_epoch,
                         max_steps=args.num_step,
                         check_val_every_n_epoch=args.val_epoch,
                         val_check_interval=args.val_step,
                         logger=logtool,
                         enable_progress_bar=args.npbar,
                         inference_mode=False,
                         accumulate_grad_batches=args.grad_accumulate,
                         )
    if args.resume_id:
        logtool.experiment.restore('best.ckpt', replace=True)
        ckpt_path = os.path.join(logtool.experiment.dir, 'best.ckpt')
    else:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path)
    model.save_model(logtool.experiment.dir)
