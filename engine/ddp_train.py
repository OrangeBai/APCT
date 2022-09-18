import torch.distributed as dist
from engine.trainer import BaseTrainer


def train(args):
    dist.init_process_group("gloo")
    trainer = BaseTrainer(args, args.local_rank)
    trainer.train_model()
