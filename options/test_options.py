#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/4 15:49
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : test_options.py
    @Desc    : 
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--test_split', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False
        return parser
