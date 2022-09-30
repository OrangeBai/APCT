from settings.train_settings import *
import torch.multiprocessing as mp
from engine.ddp_train import train
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12350'

if __name__ == '__main__':

    args = TrainParser().get_args()

    mp.spawn(train,
             args=(args,),
             nprocs=args.world_size,
             join=True)
