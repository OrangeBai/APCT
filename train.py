from settings.train_settings import ArgParser
from engine.trainer import run


if __name__ == '__main__':

    args = ArgParser(True).get_args()
    run(args)
