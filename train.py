<<<<<<< HEAD
from settings.train_settings import TrainParser
from engine.trainer import run
import wandb
=======
from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'
>>>>>>> 3078b1d (.)

if __name__ == '__main__':
    args = TrainParser().get_args()
    wandb.init(dir=args.model_dir, project="express")
    wandb.config = args
    run(args)
