#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/4/14 20:07
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : postProcessing.py
    @Desc    : 
"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter


def processing(img, opt):

    if opt.post_process:
        if opt.process_type == 'jpg':
            img = np.array(img)
            img = cv2_jpg(img, opt.process_parameter)

        elif opt.process_type == 'gaussian-blur':
            blur = transforms.GaussianBlur(kernel_size=opt.process_parameter[0], sigma=opt.process_parameter[1])
            img = blur(img)
            img = np.array(img)

        elif opt.process_type == 'resize':
            img = np.array(img)
            img = cv2_resize(img, opt.process_parameter)

        else:
            raise ValueError("Unknown process type: {}".format(opt.process_parameter))

    else:
        img = np.array(img)
    return Image.fromarray(img)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def cv2_resize(img, percent):
    height = img.shape[0]
    width = img.shape[1]
    dim = (int(width*percent/100), int(height*percent/100))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
