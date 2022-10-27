from settings.train_settings import TrainParser
from engine.trainer import run
import wandb

if __name__ == '__main__':
    wandb.init(project="my-test-project")
    args = TrainParser().get_args()
    run(args)
