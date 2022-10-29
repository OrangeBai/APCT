from settings.train_settings import TrainParser
from engine.trainer import run
import wandb

if __name__ == '__main__':
    args = TrainParser().get_args()
    run(args)
