#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/11 13:33
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : dataset.py
    @Desc    : 
"""
import os
from torch.utils.data import Dataset
from PIL import Image


class DragDataset(Dataset):
    def __init__(self, opt, dir, label, transform=None, target_transform=None):
        self.dir = dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = label
        self.instance_num = len(os.listdir(self.dir))

    def __len__(self):
        return self.instance_num

    def __getitem__(self, index):
        filename = os.listdir(self.dir)[index]
        filepath = os.path.join(self.dir, filename)

        image = Image.open(filepath)

        label = self.img_labels
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        sample = [image, label]
        return sample

