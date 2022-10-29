from settings.train_settings import TrainParser
from engine.trainer import run
import wandb

if __name__ == '__main__':
    args = TrainParser().get_args()
    wandb.init(dir=args.model_dir, project="express")
    wandb.config = args
    run(args)
