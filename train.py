from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'

if __name__ == '__main__':

    args = ArgParser(True).get_args()

    mp.spawn(train,
             args=(args,),
             nprocs=args.world_size,
             join=True)
