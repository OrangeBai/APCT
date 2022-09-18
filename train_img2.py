from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':
    args = ArgParser(True).get_args()

    dist.init_process_group('nccl')
    trainer = BaseTrainer(args, rank=args.local_rank)
    trainer.train_model()
