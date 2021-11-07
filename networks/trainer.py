#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 16:13
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : trainer.py
    @Desc    : 
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from networks.base_model import BaseModel
from networks.resnet import resnet50
from networks.resnet_ibn import resnet50_ibn_b
from .RelationModules import RelationScorer, RelationEncoder


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt, encoder=None, scorer=None):
        super(Trainer, self).__init__(opt)
        self.support_class_num = opt.support_class_num
        self.query_class_num = opt.query_class_num
        self.sample_per_class = opt.sample_num
        self.query_num = opt.query_num
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.label_type = opt.label_type
        self.prototype_cal = opt.prototype_cal

        # Initialize basic model
        if encoder is not None:
            self.encoder = encoder
        else:
            if opt.encoder == 'res50':
                self.encoder = resnet50(pretrained=True)
            elif opt.encoder == 'relation':
                self.encoder = RelationEncoder()
            elif opt.encoder == 'res50_ibn_b':
                self.encoder = resnet50_ibn_b(pretrained=True)

        if scorer is not None:
            self.scorer = scorer
        else:
            if opt.scorer == 'relation':
                self.scorer = RelationScorer(opt)
            else:
                raise ValueError("Invalid input: {}".format(opt.scorer))

        # Initialize optimizers and loss function for training
        if self.isTrain:
            self.loss_fn = nn.MSELoss().to(self.device)

            if opt.optim == 'adam':
                self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                                          lr=opt.lr, betas=(opt.beta1, 0.999),
                                                          weight_decay=opt.weight_decay)
                self.scorer_optimizer = torch.optim.Adam(self.scorer.parameters(),
                                                         lr=opt.lr, betas=(opt.beta1, 0.999),
                                                         weight_decay=opt.weight_decay)
            elif opt.optim == 'sgd':
                self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(),
                                                         lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
                self.scorer_optimizer = torch.optim.SGD(self.scorer.parameters(),
                                                        lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
            else:
                raise ValueError("optim should be [adam, sgd]")

            self.encoder_scheduler = CosineAnnealingWarmRestarts(self.encoder_optimizer, 50, 1)
            self.scorer_scheduler = CosineAnnealingWarmRestarts(self.scorer_optimizer, 50, 1)

        # For continue training or test, load networks from disk
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt)
            if opt.continue_train:
                opt.continue_train = False

        self.encoder.to(self.device)
        self.scorer.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-7):
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False

        for param_group in self.scorer_optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, sample_input, query_input):
        self.sample_input = sample_input[0].to(self.device)
        self.sample_label = sample_input[1].to(self.device).float()
        self.query_input = query_input[0].to(self.device)
        self.query_label = query_input[1].to(self.device).float()

    def forward(self):
        # calculate features
        support_feature = self.encoder(self.sample_input.to(self.device)).to(self.device)
        support_feature = self.avg_pool(support_feature)
        height, width = support_feature.shape[-2], support_feature.shape[-1]
        support_feature = support_feature.view(self.support_class_num, self.sample_per_class,
                                               -1, height, width)  # cls * instance * c * h * w
        if self.prototype_cal == 'sum':
            support_feature = torch.sum(support_feature, 1).squeeze(1)
        elif self.prototype_cal == 'avg':
            support_feature = torch.mean(support_feature, 1).squeeze(1)
        else:
            raise ValueError("Prototype calculation method undefined, should be [sum, avg]")
        query_feature = self.encoder(self.query_input.to(self.device)).to(self.device)
        query_feature = self.avg_pool(query_feature)

        # calculate relations
        support_feature_ext = support_feature.unsqueeze(0) \
            .repeat(self.query_num * self.query_class_num, 1, 1, 1, 1).to(
            self.device)  # query_num * cls_num, cls_num, c, h, w

        query_feature_ext = query_feature.unsqueeze(0) \
            .repeat(self.support_class_num, 1, 1, 1, 1).to(self.device)  # cls_num, query_num * cls_num, c, h, w

        query_feature_ext = torch.transpose(query_feature_ext, 0, 1)
        relation_pairs = torch.cat((support_feature_ext, query_feature_ext), 2) \
            .view(self.query_num * self.query_class_num * self.support_class_num, -1, height, width).to(self.device)

        self.output = self.scorer(relation_pairs).view(-1, self.support_class_num)
        return self.output

    def get_loss(self):
        self.query_label = self.query_label.long()

        one_hot_labels = (torch.zeros(self.query_num * self.query_class_num, self.support_class_num).to(self.device)
                          .scatter_(1, self.query_label.view(-1, 1), 1))

        if self.label_type == 'one-hot':
            label = one_hot_labels
            out = self.output
            if self.loss_function == 'cosine':
                return self.loss_fn(out, label, torch.Tensor(out.size()[0]).cuda().fill_(1.0))
            elif self.loss_function == 'mse':
                return self.loss_fn(out, label)

        elif self.label_type == 'drag-label':
            query_label = self.query_label
            for index, i in enumerate(query_label.view(-1, 1)):
                if i.data == 2:
                    query_label[index] = 3
                elif i.data == 3:
                    query_label[index] = 2
                elif i.data == 0:
                    query_label[index] = 1
                elif i.data == 1:
                    query_label[index] = 0
            drag_labels = one_hot_labels.scatter_(1, query_label.view(-1, 1), 1)
            label = drag_labels
            r = torch.mean(self.output[:, 0:2], 1).view(-1, 1)
            f = torch.mean(self.output[:, 2:4], 1).view(-1, 1)

            # for ease of training, we only calculate loss by the prediction of [real_score, fake_score] and true value
            out = torch.cat((r, f), 1)
            return self.loss_fn(out, label[:, 1:3])
        else:
            raise ValueError("Label type should only be [one-hot, drag-label]")

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss().to(self.device)
        self.encoder_optimizer.zero_grad()
        self.scorer_optimizer.zero_grad()
        self.loss.backward()
        self.encoder_optimizer.step()
        self.scorer_optimizer.step()
