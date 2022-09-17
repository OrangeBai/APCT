from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    argv = ['--dataset', 'imagenet', '--lr_scheduler', 'linear', '--lr', '0', '--lr_e', '0.4',
            '--batch_size', '128', '--data_size', '160', '--crop_size', '128',
            '--num_epoch', '1']
    if rank == 0:
        args = ArgParser(True, argv).get_args()
    else:
        args = ArgParser(False, argv).get_args()
    trainer = BaseTrainer(args, rank)
    trainer.train_model()

    # argv = ['--dataset', 'imagenet', '--lr_scheduler', 'linear', '--lr', '0.4', '--lr_e', '0.04',
    #         '--batch_size', '128', '--data_size', '160', '--crop_size', '128',
    #         '--num_epoch', '5', '--resume', '1']
    # args = ArgParser(True, argv).get_args()
    #
    # mp.spawn(train,
    #          args=(args,),
    #          nprocs=args.world_size,
    #          join=True)
