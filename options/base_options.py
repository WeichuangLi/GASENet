#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 15:46
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : base_options.py
    @Desc    : 
"""
import argparse
import os
import util
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--encoder', type=str, default='res50',
                            choices=['res50', 'relation', 'res50_ibn_b'],
                            help='encoder architecture')
        parser.add_argument('--scorer', type=str, default='relation', choices=['relation'],
                            help='scorer architecture')

        # data info
        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders train, '
                                                                     'val, test, etc)')
        parser.add_argument('--label_type', default='drag-label', choices=['drag-label', 'one-hot'],
                            help='drag-label means classify different gan as the same, '
                                 'while one-hot label will take the classification task into account')
        parser.add_argument('--mix_up', action='store_true', help='Whether to mix up different sub-domain')

        # Relation settings
        parser.add_argument('--support_class_num', default=4, type=int, help='n-way training')
        parser.add_argument('--query_class_num', default=4, type=int)
        parser.add_argument('--sample_num', default=8, type=int, help='n-shot training')
        parser.add_argument('--num_instance', default=7000, type=int, help='n images for each class')
        parser.add_argument('--query_num', default=8, type=int,
                            help='take n query images/class to calculate relations')
        parser.add_argument('--prototype_cal', default='sum', choices=['sum', 'avg'], type=str,
                            help='calculation method for the prototype')

        # Score module setting
        parser.add_argument('--relation_channel', default=4096, type=int, help='output channel number of relations')
        parser.add_argument('--hidden_size', default=8, type=int, help='hidden dimension in relation network')
        # parser.add_argument('--cnn_output_size', default=8, type=int, help='relation scorer last cnn output')

        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0, 1, 2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='Augmentation_',
                            help='name of the experiment. It decides the folder to store checkpoints')
        parser.add_argument('--loading_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use '
                                 'latest cached model')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: '
                                                                   'e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--encoder_pth', type=str, default='res50_ibn_b_epoch_200_val_0.998250.pth',
                            help='name of encoder model(xxx.pth)')
        parser.add_argument('--scorer_pth', type=str, default='relation_epoch_200_val_0.998250.pth',
                            help='name of scorer model(xxx.pth)')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Augmentation parameters
        parser.add_argument('--augmentation', action='store_true', help='whether to augment training data')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', type=tuple, default=[0, 0.5, 1.0, 1.5])
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_qual', type=tuple, default=[85, 100])

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')

        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        if opt.scorer == 'NIN':
            opt.scorer = opt.scorer + '_' + opt.norm_type

        opt.name += 'Mixup_{}_Prototype_{}_dlabel_{}-way__encoder_{}_scorer_{}'.format(opt.mix_up,
                                                                                       opt.prototype_cal,
                                                                                       opt.support_class_num,
                                                                                       opt.encoder, opt.scorer)

        if opt.encoder[:5] == 'res50':
            opt.relation_channel = 2048 * 2
        elif opt.encoder == 'relation':
            opt.relation_channel = 64 * 2

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # initialize dirs to save model
        encoder_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "encoder")
        scorer_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "scorer")
        util.mkdirs([encoder_save_dir, scorer_save_dir])

        self.opt = opt
        return self.opt
