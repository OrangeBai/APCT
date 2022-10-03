from settings.train_settings import TrainParser
from engine.trainer import run


if __name__ == '__main__':

    args = TrainParser().get_args()
    run(args)
