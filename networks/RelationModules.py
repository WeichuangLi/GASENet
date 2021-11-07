#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 16:43
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : RelationModules.py
    @Desc    : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationEncoder(nn.Module):
    """Shallow CNN encoder, using this encoder will induce a quite bad performance"""

    def __init__(self):
        super(RelationEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationScorer(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, opt):
        super(RelationScorer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(opt.relation_channel, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, opt.hidden_size)
        self.fc2 = nn.Linear(opt.hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.avgpool(self.layer2(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

