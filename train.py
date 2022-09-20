from settings.train_settings import ArgParser
from engine.trainer import run

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if __name__ == '__main__':

    args = ArgParser(True).get_args()
    run(args)
