#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/10 9:32
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : __init__.py
    @Desc    : 
"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from .dataLoader import ClassBalancedSampler
from .dataset import DragDataset
from .postProcessing import processing
from .data_augmentation import augmentation


def _get_dataset(folder):
    """
        label: 0
        root/train/0-Real/xxx.png
        ...
        label: 1
        root/train/1-GAN1/xxx.png
        ...
        label:2
        root/train/2-GAN2/xxx.png
    """
    return datasets.ImageFolder(
        folder,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )


def get_dataset(opt, mode, dataset_type, transform=None):
    folder = os.path.join(opt.dataroot, mode)
    dataset_list = []
    # When training, all images were chosen from the following folders
    # When testing, the reference images are also from the following folders
    if mode != "test" or (mode == "test" and dataset_type == "support"):
        data_sets = ['0-FFHQ',  '1-CelebaHQ', '2-PGGAN', '3-StyleGAN']
        dataset_number = len(data_sets)

        # During testing, we choose the support image from training set, so we need to reassign the folder path
        if mode == "test" and dataset_type == "support":
            folder = os.path.join(opt.dataroot, "train")

        for i, data_set in enumerate(data_sets):
            # during training, we only augment query images to mimic the real-world scenario
            if mode == "train" and dataset_type == "query":
                transform = transforms.Compose([
                    transforms.Lambda(lambda img: augmentation(img, opt)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

            tmp_folder = os.path.join(folder, data_set.strip())
            if dataset_type == "support" and mode == "train":
                print("Reference folder: ", tmp_folder)
            elif mode == "train":
                print("Query folder:", tmp_folder)
            single_dataset = DragDataset(opt, tmp_folder, int(data_set.strip('-')[0].strip()), transform)

            dataset_list.append(single_dataset)
        dataset = torch.utils.data.ConcatDataset(dataset_list)

    # During testing, we might need to choose specific folders.
    elif mode == "test" and dataset_type == "query":
        data_sets = opt.testsets.split(',')
        dataset_number = len(data_sets)

        for i, data_set in enumerate(data_sets):
            transform = transforms.Compose([
                transforms.Lambda(lambda img: processing(img, opt)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            data_set = data_set.strip()
            tmp_folder = os.path.join(folder, data_set)
            print("Query folder:", tmp_folder)
            single_dataset = DragDataset(opt, tmp_folder, label=int(data_set.strip('-')[0].strip()), transform=transform)

            dataset_list.append(single_dataset)
        dataset = torch.utils.data.ConcatDataset(dataset_list)

    else:
        raise ValueError('Unexpected input, mode:{}, dataset_type:{}'.format(mode, dataset_type))

    return dataset, dataset_number


def get_dataloader(opt, num_per_class, mode='train', loader_type='', shuffle=False, mix=False):
    image_datasets, class_num = get_dataset(opt, mode, loader_type)
    if mode == 'test' and loader_type == 'support':
        sampler = ClassBalancedSampler(num_per_class, class_num, 7000, image_datasets, shuffle, mix)
        loader = DataLoader(image_datasets, batch_size=num_per_class * opt.support_class_num,
                            sampler=sampler, pin_memory=True)
    else:
        sampler = ClassBalancedSampler(num_per_class, class_num, opt.num_instance, image_datasets, shuffle, mix)
        loader = DataLoader(image_datasets, batch_size=num_per_class * opt.query_class_num,
                            sampler=sampler, pin_memory=True)

    return loader
