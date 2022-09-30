import torch.multiprocessing as mp
import torch.distributed as dist
from engine.trainer import BaseTrainer
from settings.train_settings import *
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12350'


def train(rank, arg):
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    trainer = BaseTrainer(arg, rank)
    trainer.train_model()


if __name__ == '__main__':

    args = TrainParser().get_args()

    mp.spawn(train,
             args=(args,),
             nprocs=args.world_size,
             join=True)
