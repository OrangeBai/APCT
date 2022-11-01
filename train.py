from settings.train_settings import TrainParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from core.engine.trainer import set_pl_model
import pytorch_lightning as pl


if __name__ == '__main__':
    args = TrainParser().get_args()

    logtool = WandbLogger(name=args.name, save_dir=args.model_dir, project=args.project, config=args)
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", dirpath=logtool.experiment.dir, filename="ckpt-best"),
        # ModelPruning("l1_unstructured", amount=0.5)
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

    # datamodule=DataModule(args)
    model = set_pl_model(args)
    trainer.fit(model)



