import os
import logging
import torch
import datetime


class Log:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=os.path.join(args.model_dir, 'logger_' + str(args.exp_id)))
        self.logger.info(args)

    def step_logging(self, step, batch_num, epoch, epoch_num, metrics, time_metrics=None):
        space_fmt = ':' + str(len(str(batch_num))) + 'd'

        log_msg = '\t'.join(['Epoch: [{epoch}/{epoch_num}]',
                             '[{step' + space_fmt + '}/{batch_num}]',
                             '{time_str}',
                             '{meters}',
                             '{memory}'
                             ])

        if time_metrics is not None:
            eta_seconds = time_metrics.meters['iter_time'].global_avg * (batch_num - step)
            eta_string = 'eta: {}'.format(str(datetime.timedelta(seconds=int(eta_seconds))))

            time_str = '\t'.join([eta_string, str(time_metrics)])
        else:
            time_str = ''

        msg = log_msg.format(epoch=epoch, epoch_num=epoch_num,
                             step=step, batch_num=batch_num,
                             time_str=time_str, meters=str(metrics),
                             memory='max mem: {0:.2f}'.format(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                             )
        self.logger.info(msg)
        print(msg)
        return

    def train_logging(self, epoch, epoch_num, metrics, time_metrics=None):
        """
        Print loggings after training of each epoch
        @param epoch:
        @param epoch_num:
        @param metrics:
        @param time_metrics:
        @return:
        """
        self.logger.info('Epoch: [{epoch}/{epoch_num}] training finished'.format(epoch=epoch, epoch_num=epoch_num))
        log_msg = '\t'.join(['TRN INF:', '{meters}\t'])
        msg = log_msg.format(meters=str(metrics))
        if time_metrics is not None:
            msg += 'time: {time:.4f}'.format(time=time_metrics.meters['iter_time'].total)
        self.logger.info(msg)
        print(msg)
        return

    def val_logging(self, metrics):
        msg = '\t'.join(['VAL INF:', '{meters}']).format(meters=metrics)
        self.logger.info(msg)
        print(msg)
        return msg

    def info(self, msg):
        self.logger.info(msg)
