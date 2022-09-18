from attack import *
from engine.trainer import *


class AdvTrainer(BaseTrainer):
    def __init__(self, args, rank=-1):
        super(AdvTrainer, self).__init__(args, rank)
        self.attack = DDP(set_attack(self.model.module, self.args), device_ids=[rank], output_device=rank)

        dist.barrier()

    def train_step(self, images, labels):
        images, labels = images.to(self.rank), labels.to(self.rank)
        images = self.attack(images, labels)
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        top1, top5 = accuracy(outputs, labels)
        self.metrics.update(
            top1=(top1, len(images)), top5=(top5, len(images)),
            loss=(loss, len(images)),
            lr=(self.get_lr(), 1)
        )
        self.metrics.all_reduce()

    # def warmup(self):
    #     if self.args.warmup_steps == 0:
    #         return
    #     loader = InfiniteLoader(self.train_loader)
    #     self.lr_scheduler = warmup_scheduler(self.args, self.optimizer)
    #     for cur_step in range(self.args.warmup_steps):
    #         images, labels = next(loader)
    #         images, labels = to_device(self.args.devices[0], images, labels)
    #         # self.train_step(images, labels)
    #         if cur_step % self.args.print_every == 0 and cur_step != 0 and self.rank == 0:
    #             self.logger.step_logging(cur_step, self.args.warmup_steps, -1, -1, self.metrics, loader.metric)
    #
    #         if cur_step >= self.args.warmup_steps:
    #             break
    #     self.logger.train_logging(-1, self.args.num_epoch, self.metrics, loader.metric)
    #     self.validate_epoch()
    #     self.optimizer = init_optimizer(self.args, self.model)
    #     self.lr_scheduler = init_scheduler(self.args, self.optimizer)
    #
    #     return

    def train_epoch(self, epoch):
        cur_time = time.time()
        for step, (images, labels) in enumerate(self.train_loader):
            data_time = time.time() - cur_time

            images, labels = images.to(self.rank), labels.to(self.rank)
            self.train_step(images, labels)
            if step % self.args.print_every == 0 and step != 0 and self.rank == 0:
                self.logger.step_logging(step, self.args.epoch_step, epoch, self.args.num_epoch,
                                         self.metrics, self.time_metric)

            iter_time = time.time() - cur_time
            cur_time = time.time()
            self.time_metric.update(iter_time=(iter_time, 1), data_time=(data_time, 1))
            self.time_metric.all_reduce()
            self.metrics.all_reduce()
        if self.rank == 0:
            self.logger.train_logging(epoch, self.args.num_epoch, self.metrics, self.time_metric)
        self.time_metric.reset()
        return

    def validate_epoch(self):
        start = time.time()
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = images.to(self.rank), labels.to(self.rank)
            pred = self.model(images)
            top1, top5 = accuracy(pred, labels)
            self.metrics.update(top1=(top1, len(images)))
            self.metrics.all_reduce()
            # if self.args.record_lip:
            #     self.record_lip(images, labels, pred)
        self.logger.val_logging(self.metrics, time.time() - start)

        self.model.train()
        return self.metrics.meters['top1'].global_avg

    def train_model(self):

        # self.warmup()

        for epoch in range(self.start_epoch, self.args.num_epoch):
            self.train_epoch(epoch)
            self.record_result(epoch)

            acc = self.validate_epoch()
            if acc > self.best_acc:
                self.best_acc = acc
                if self.rank == 0:
                    self.save_ckpt(epoch + 1, self.best_acc, 'best')

        if self.rank == 0:
            if self.args.save_name == '':
                self.args.save_name = 'epoch_{}'.format(str(self.args.num_epoch).zfill(3))
            self.save_result(self.args.model_dir, self.args.save_name)
            self.save_ckpt(self.args.num_epoch, self.best_acc, self.args.save_name)
