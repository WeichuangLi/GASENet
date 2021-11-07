#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 15:48
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : train_options.py
    @Desc    : 
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the '
                                                                           'end of epochs')

        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model '
                                                                       'by <epoch_count>, '
                                                                       '<epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler '
                                                                       'intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--epoch', type=int, default=200, help='epoch number')
        # hyper parameters for optimizers
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay factor for optimizers')

        # Early stop
        parser.add_argument('--early_stop', type=bool, default=False)
        parser.add_argument('--earlystop_epoch', type=int, default=7)

        return parser
