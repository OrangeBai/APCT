from settings.train_settings import TrainParser
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from core.engine.trainer import set_pl_model
import pytorch_lightning as pl


if __name__ == '__main__':
    args = TrainParser().get_args()

    logtool = WandbLogger(name=args.name, save_dir=args.model_dir, project=args.project, config=args)

    # datamodule=DataModule(args)
    model = set_pl_model(args)

    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        # if epoch == 2:
        #     return 0.5
        #
        # elif epoch == 50:
        #     return 0.25
        #
        # elif 75 < epoch < 99:
        #     return 0.01
        return 0.5
    callbacks = [
        ModelCheckpoint(monitor='val/top1', save_top_k=1, mode="max", save_on_train_epoch_end=False,
                        dirpath=logtool.experiment.dir, filename="ckpt-best"),
    ]

    trainer = pl.Trainer(devices="auto",
                         precision=16,
                         amp_backend="native",
                         accelerator="cuda",
                         strategy=DDPStrategy(find_unused_parameters=False),
                         callbacks=callbacks,
                         max_epochs=args.num_epoch,
                         max_steps=args.num_step,
                         check_val_every_n_epoch=args.val_epoch,
                         val_check_interval=args.val_step,
                         logger=logtool,
                         enable_progress_bar=args.npbar
                         )
    trainer.fit(model)
