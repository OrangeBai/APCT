import os, argparse
import torch
import torch.distributed as dist

parse = argparse.ArgumentParser()
parse.add_argument('--init_method', type=str)
parse.add_argument('--rank', type=int)
parse.add_argument('--ws', type=int)
args = parse.parse_args()

if args.init_method == 'TCP':
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=args.rank, world_size=args.ws)
elif args.init_method == 'ENV':
    dist.init_process_group('nccl', init_method='env://')

rank = dist.get_rank()
print(f"rank = {rank} is initialized")
# 单机多卡情况下，localrank = rank. 严谨应该是local_rank来设置device
torch.cuda.set_device(rank)
tensor = torch.tensor([1, 2, 3, 4]).cuda()
print(tensor)