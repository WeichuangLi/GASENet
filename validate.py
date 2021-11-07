#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/7 10:24
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : validate.py
    @Desc    : 
"""

import torch
import numpy as np
from data import get_dataloader
from networks.trainer import Trainer


def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    labelConvert = lambda num: 0 if (num == 0 or num == 1) else 2
    # labelConvert = lambda num: 0 if (num == 0) else 1
    # Define the real images as positive sample, fake ones as negative sample
    tp, tn, fp, fn = [], [], [], []
    for j in range(len(y_true)):
        y_true_j = labelConvert(y_true[j])
        # print(y_true[j], y_true_j)
        y_pred_j = labelConvert(y_pred[j])
        # print('pred', y_pred[j], y_pred_j)
        if y_true_j == 0 and y_pred_j == 0:
            tp.append(1)
        elif y_true_j > 0 and y_pred_j > 0:
            tn.append(1)
        elif y_true_j == 0 and y_pred_j > 0:
            fn.append(1)
        else:
            fp.append(1)

    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    # print(tpr, tnr)
    # print("tpr:", tpr)
    # print("tnr:", tnr)
    acc = (tpr + tnr) / 2

    return acc, tpr, tnr


def one_hot_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        if y_true[i] < 2:
            y_true[i] = 0
        else:
            y_true[i] = 2

        if y_pred[i] < 2:
            y_pred[i] = 0
        else:
            y_pred[i] = 2
    rewards = [1 if y_pred[j] == y_true[j] else 0 for j in range(len(y_true))]
    total_rewards = np.sum(rewards)
    acc = total_rewards / len(y_true)
    return acc


def validate(opt, encoder=None, scorer=None, mode='val'):
    sample_loader = get_dataloader(opt, opt.sample_num, mode, "support", False, mix=opt.mix_up)
    query_loader = get_dataloader(opt, opt.query_num, mode, "query", True, mix=opt.mix_up)
    model = Trainer(opt, encoder, scorer)

    total_loss = 0.0

    with torch.no_grad():
        reward, y_true, y_pred, tpr, tnr = [], [], [], [], []
        relation_scores = torch.Tensor().to(model.device)
        sample_data = sample_loader.__iter__().next()
        for i, query_data in enumerate(query_loader):
            model.set_input(sample_data, query_data)
            relation_score = model.forward().to(model.device)
            relation_scores = torch.cat([relation_scores, relation_score])

            if opt.label_type == 'drag-label':
                predict_labels = torch.zeros(relation_score.data.shape[0])
                for index, item in enumerate(relation_score.data):
                    f_score = (item[2] + item[3]) / 2
                    r_score = (item[0] + item[1]) / 2
                    if f_score > r_score:
                        predict_labels[index] = 2
                    else:
                        predict_labels[index] = 0
                predict_labels.long().to(model.device)
            elif opt.label_type == 'one-hot':
                _, predict_labels = torch.max(relation_score.data, 1)
            else:
                raise ValueError("Label type undefined")

            query_labels = query_data[1].long()
            if opt.label_type == 'drag-label':
                batch_reward, batch_tpr, batch_tnr = accuracy(query_labels, predict_labels.long())
            else:
                batch_reward = one_hot_accuracy(query_labels, predict_labels.long())
            y_true.extend(query_labels)
            y_pred.extend(predict_labels.long())
            reward.append(batch_reward)
            tpr.append(batch_tpr)
            tnr.append(batch_tnr)

            if mode != 'test':
                batch_loss = model.get_loss()
                total_loss += batch_loss

    # val_cls_acc = np.float_(np.mean(val_cls))
    acc = np.float_(np.mean(reward))
    print("tpr: ", np.float_(np.mean(tpr)))
    print("tnr: ", np.float_(np.mean(tnr)))
    loss = total_loss / len(sample_loader)
    return acc, loss, y_true, y_pred, relation_scores
