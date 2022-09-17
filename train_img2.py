import torch.distributed as dist

from engine.trainer import BaseTrainer
from settings.train_settings import *

if __name__ == '__main__':
    env_dict = {key: os.environ[key] for key in ('MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')}
    print(os.environ)
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    argv = ['--dataset', 'imagenet', '--lr_scheduler', 'linear', '--lr', '0', '--lr_e', '0.4',
            '--batch_size', '128', '--data_size', '160', '--crop_size', '128',
            '--num_epoch', '1']

    if os.environ['RANK'] == 0:
        arg_parser = ArgParser(True, argv)

    dist.barrier()
    args_parser = ArgParser(False, argv).load()

    args = args_parser.get_args()

    local_rank = os.environ['LOCAL_RANK']
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
