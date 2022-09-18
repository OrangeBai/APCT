from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':
    args = ArgParser(True).get_args()

    # dist.init_process_group("nvcc", init_method=store, rank=rank, world_size=args.nnodes * args.nproc_per_node)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:28765',
                            rank=args.local_rank, world_size=args.world_size)
    trainer = BaseTrainer(args, rank=args.local_rank)
    trainer.train_model()
