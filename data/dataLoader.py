#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/6 15:04
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : dataLoader.py
    @Desc    : 
"""
import random
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, class_num, num_instance, data_source, shuffle=True, mix=False):
        super().__init__(data_source)
        self.num_per_class = num_per_class
        self.class_num = class_num
        self.num_inst = num_instance
        self.shuffle = shuffle
        self.data_source = data_source
        if mix:
            self.num_per_class *= 2
            self.num_inst *= 2
            self.class_num = self.class_num // 2

    def __iter__(self):
        batch_num = len(self.data_source) // (self.num_per_class * self.class_num)
        batch = []
        for i in range(batch_num):
            mini_batch = []
            for j in range(self.class_num):
                if self.shuffle:
                    for k in torch.randperm(self.num_inst)[i * self.num_per_class:(i + 1) * self.num_per_class]:
                        mini_batch.append(k + j * self.num_inst)
                else:
                    for k in range(self.num_inst)[i * self.num_per_class:(i + 1) * self.num_per_class]:
                        mini_batch.append(k + j * self.num_inst)
            if self.shuffle:
                random.shuffle(mini_batch)
            batch.extend(mini_batch)

        return iter(batch)

    def __len__(self):
        return (len(self.data_source) // (self.num_per_class * self.class_num)) * (self.num_per_class * self.class_num)
