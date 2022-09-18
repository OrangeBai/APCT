import torch.distributed as dist
from engine.trainer import BaseTrainer
from engine.adv_trainer import AdvTrainer

def train(rank, args):
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    trainer = AdvTrainer(args, rank)
    trainer.train_model()
