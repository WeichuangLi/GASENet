#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/7 10:24
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : train.py
    @Desc    : 
"""

import os
import time
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions
from data import get_dataloader
from networks.trainer import Trainer
from validate import validate, accuracy, one_hot_accuracy
from earlystop import EarlyStopping
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.query_num = 8
    val_opt.sample_num = 8
    val_opt.num_instance = 1000
    return val_opt


if __name__ == '__main__':
    train_opt = TrainOptions().parse(print_options=True)
    val_opt = get_val_opt()

    sample_loader = get_dataloader(train_opt, train_opt.sample_num, "train", "support", False, mix=train_opt.mix_up)
    sample_size = len(sample_loader)
    query_loader = get_dataloader(train_opt, train_opt.query_num, "train", "query", True, mix=train_opt.mix_up)
    query_size = len(query_loader)

    print("Training dataset, {} sample batches, {} query batches".format(sample_size, query_size))

    model = Trainer(train_opt)
    if train_opt.early_stop:
        early_stopping = EarlyStopping(patience=train_opt.earlystop_epoch, delta=-0.0001, verbose=True)

    train_writer = SummaryWriter(os.path.join(train_opt.checkpoints_dir, train_opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(train_opt.checkpoints_dir, train_opt.name, "val"))

    if train_opt.last_epoch != -1:
        starting = train_opt.last_epoch
    else:
        starting = 0

    for epoch in range(starting, train_opt.epoch):
        start_time = time.time()
        epoch_iter = 0
        train_loss = 0
        train_reward = []

        for i, query_data in enumerate(query_loader):
            sample_data = sample_loader.__iter__().next()

            model.total_steps += 1
            epoch_iter += train_opt.query_class_num * (train_opt.query_num + train_opt.sample_num)
            model.set_input(sample_data, query_data)
            model.optimize_parameters()
            train_loss += model.get_loss()
            query_labels = query_data[1].long()

            # calculate accuracies
            relation_score = model.output.to(model.device)
            if train_opt.label_type == 'drag-label':
                predict_labels = torch.zeros(relation_score.data.shape[0])
                for index, item in enumerate(relation_score.data):
                    f_score = (item[2] + item[3]) / 2
                    r_score = (item[0] + item[1]) / 2
                    if f_score > r_score:
                        predict_labels[index] = 2
                    else:
                        predict_labels[index] = 0
                predict_labels.long().to(model.device)
                batch_reward = accuracy(query_labels.to(model.device), predict_labels.long().to(model.device))
            elif train_opt.label_type == 'one-hot':
                _, predict_labels = torch.max(relation_score.data, 1)
                batch_reward = one_hot_accuracy(query_labels.to(model.device), predict_labels.long().to(model.device))
            else:
                raise ValueError("Label type undefined")

            train_reward.append(batch_reward)

        train_acc = np.float_(np.mean(train_reward))
        train_loss = train_loss / model.total_steps
        lr = model.encoder_optimizer.param_groups[0]['lr']

        # train_writer.add_scalar('train_acc', train_cls_acc, model.total_steps)
        train_writer.add_scalar('train_acc', train_acc, model.total_steps)
        train_writer.add_scalar('lr', lr, model.total_steps)
        train_writer.add_scalar('train_loss', train_loss, model.total_steps)

        model.encoder_scheduler.step()
        model.scorer_scheduler.step()

        # Validation
        model.eval()
        val_acc, val_loss = validate(val_opt, model.encoder, model.scorer, mode='val')[:2]
        # val_writer.add_scalar('val_acc', val_cls_acc, model.total_steps)
        val_writer.add_scalar('val_acc', val_acc, model.total_steps)
        val_writer.add_scalar('val_loss', val_loss, model.total_steps)
        if train_opt.label_type == 'drag-label':
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f |'
                  ' Acc: %3.6f loss: %3.6f' %
                  (epoch + 1, train_opt.epoch, time.time() - start_time,
                   train_acc, train_loss, val_acc, val_loss))
        else:
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f |'
                  'Acc: %3.6f loss: %3.6f' %
                  (epoch + 1, train_opt.epoch, time.time() - start_time,
                   train_acc, train_loss, val_acc, val_loss))

        if (epoch + 1) % train_opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch + 1, model.total_steps))

            model.save_networks(epoch + 1, val_acc)

        if train_opt.early_stop:
            early_stopping(val_acc, val_loss, "loss", model, epoch + 1)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=train_opt.earlystop_epoch, delta=-0.0002, verbose=True)
                else:
                    print("Early stopping.")
                    break
        model.train()

    train_writer.flush()
    val_writer.flush()
