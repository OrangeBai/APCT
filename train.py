from settings.train_settings import *
from engine.trainer import BaseTrainer
from engine.adv_trainer import AdvTrainer
import torch.multiprocessing as mp
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':

    args = ArgParser(True).get_args()

    world_size = args.world_size

    trainer = AdvTrainer(args)
    # trainer.train_model(0, 1)
    mp.spawn(trainer.train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)
