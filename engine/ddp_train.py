import torch.distributed as dist
from engine.trainer import BaseTrainer


def train(rank, args):
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    trainer = BaseTrainer(args, rank)
    trainer.train_model()
