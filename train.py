from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    arg_parser = ArgParser(True)

    world_size = 1

    trainer = BaseTrainer(arg_parser.get_args())
    # trainer.train_model(0, 1)
    mp.spawn(trainer.train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)
