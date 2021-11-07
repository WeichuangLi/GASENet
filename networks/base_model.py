#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 16:14
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : base_model.py
    @Desc    : 
"""
import os

import torch
import torch.nn as nn
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, epoch, val_acc):
        encoder_save_filename = '{}_epoch_{}_val_{:3.6f}.pth'.format(self.opt.encoder, epoch, val_acc)
        encoder_save_path = os.path.join(self.save_dir, "encoder", encoder_save_filename)

        # serialize model and optimizer to dict
        encdoer_state_dict = {
            'model': self.encoder.state_dict(),
            'optimizer': self.encoder_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(encdoer_state_dict, encoder_save_path)

        scorer_save_filename = '{}_epoch_{}_val_{:3.6f}.pth'.format(self.opt.scorer, epoch, val_acc)
        scorer_save_path = os.path.join(self.save_dir, "scorer", scorer_save_filename)

        # serialize model and optimizer to dict
        scorer_state_dict = {
            'model': self.scorer.state_dict(),
            'optimizer': self.scorer_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(scorer_state_dict, scorer_save_path)

    # load models from the disk
    def load_networks(self, opt):
        encoder_save_filename = opt.encoder_pth
        scorer_save_filename = opt.scorer_pth

        encoder_load_path = os.path.join(self.save_dir, "encoder", encoder_save_filename)
        scorer_load_path = os.path.join(self.save_dir, "scorer", scorer_save_filename)

        encoder_state_dict = torch.load(encoder_load_path, map_location=self.device)
        if hasattr(encoder_state_dict, '_metadata'):
            del encoder_state_dict._metadata

        self.encoder.load_state_dict(encoder_state_dict['model'])
        self.total_steps = encoder_state_dict['total_steps']

        # load optimizer if continue training
        if self.isTrain and not self.opt.new_optim:
            self.encoder_optimizer.load_state_dict(encoder_state_dict['optimizer'])
            # move optimizer state to GPU
            for state in self.encoder_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.encoder_optimizer.param_groups:
                g['lr'] = self.opt.lr

        scorer_state_dict = torch.load(scorer_load_path, map_location=self.device)
        if hasattr(scorer_state_dict, '_metadata'):
            del scorer_state_dict._metadata

        self.scorer.load_state_dict(scorer_state_dict['model'])
        self.total_steps = scorer_state_dict['total_steps']

        if self.isTrain and not self.opt.new_optim:
            self.scorer_optimizer.load_state_dict(scorer_state_dict['optimizer'])
            # move optimizer state to GPU
            for state in self.scorer_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.scorer_optimizer.param_groups:
                g['lr'] = self.opt.lr

    def test(self):
        with torch.no_grad():
            self.forward()

    def eval(self):
        self.encoder.eval()
        self.scorer.eval()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
