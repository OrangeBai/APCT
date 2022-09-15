from settings.train_settings import *
from engine.trainer import BaseTrainer
import torch.multiprocessing as mp
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':
    argv = ['--dataset', 'imagenet', '--lr_scheduler', 'linear', '--lr', '0', '--base_lr', '0.4',
            '--batch_size', '512', '--data_size', '160', '--crop_size', '128']
    args = ArgParser(True, argv).get_args()

    world_size = args.world_size

    trainer = BaseTrainer(args)
    mp.spawn(trainer.train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)
