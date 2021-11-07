# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_augmentation.py
# Time       ：20/8/2021 8:38 AM
# Author     ：Weichuang Li
# Email      : WeichuangLi1999@outlook.com
# Description：
"""
import numpy as np
from random import random, choice
from .postProcessing import gaussian_blur, cv2_jpg
from PIL import Image


def augmentation(img, opt):
    img = np.array(img)
    if opt.augmentation:
        if random() < opt.blur_prob:
            sig = sample_discrete(opt.blur_sig)
            gaussian_blur(img, sig)

        if random() < opt.jpg_prob:
            qual = sample_continuous(opt.jpg_qual)
            img = cv2_jpg(img, qual)
    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)
