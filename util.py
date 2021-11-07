#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 15:47
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : util.py
    @Desc    : 
"""
import os


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)