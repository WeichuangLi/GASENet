#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/11 11:31
    @Author  : Weichuang Li
    @Email   : WeichuangLi1999@outlook.com
    @Project : RN-GANdetector
    @File    : eval.py
    @Desc    : 
"""

import csv
import datetime
from validate import validate
from options.test_options import TestOptions
from util import mkdirs
from networks.trainer import Trainer

import warnings

warnings.filterwarnings("ignore")

test_opt = TestOptions().parse(print_options=False)
encoder_name = test_opt.encoder
scorer_name = test_opt.scorer

print("Encoder:{}, Scorer:{}".format(encoder_name, scorer_name))
print("Encoder_pth:{}, Scorer_pth:{}".format(test_opt.encoder_pth, test_opt.scorer_pth))

rows = [
    ['Encoder: {}, Scorer: {}'.format(encoder_name, scorer_name)],
    ['test']
]

exp_dict = {'PGGAN': ['1-CelebaHQ, 2-PGGAN'],
            'StyleGAN': ['0-FFHQ, 3-StyleGAN'],
            'StyleGAN2': ['0-FFHQ, 4-StyleGAN2'],
            'StarGAN-short': ['1-CelebaHQ, 6-StarGAN-black'],
            'StarGAN': ['1-CelebaHQ, 6-StarGAN-black',
                        '1-CelebaHQ, 6-StarGAN-blond',
                        '1-CelebaHQ, 6-StarGAN-brown',
                        '1-CelebaHQ, 6-StarGAN-male',
                        '1-CelebaHQ, 6-StarGAN-smiling', ],
            'BigGAN': ['0-biggan-real, 8-biggan-generated'],
            'GauGAN': ['0-gaugan-real, 9-gaugan-generated'],
            'trans-celeb': ['1-CelebaHQ, 2-trans-celeb-generated'],
            'trans-FFHQ': ['0-FFHQ, 4-trans-ffhq-generated'],
            'RelGAN': ['1-CelebaHQ, 4-relgan-generated'], }

exp_list = []
exp_name = ['PGGAN',
            'StyleGAN',
            'StyleGAN2',
            'StarGAN',
            'trans-FFHQ',
            'RelGAN',
            ]

for i in exp_name:
    exp_list.append(exp_dict[i])

shot_num = 1
for shot in range(shot_num):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    total_summary = [['dataset', 'process_type', 'process_parameter', 'acc']]
    for sub_exp in exp_list:
        for exp in sub_exp:
            test_opt.testsets = exp
            test_opt.support_class_num = 4
            test_opt.query_class_num = len(test_opt.testsets.split(','))
            test_opt.num_instance = 2000
            test_opt.sample_num = 8
            test_opt.query_num = 15

            # Post-process
            test_opt.post_process = True
            # test_opt.verbose =

            name = [key for key, value in exp_dict.items() if value == sub_exp][0]
            test_results = './results/{}/{}_{}_shot_{}_ref'.format(time, name, shot_num, test_opt.sample_num)
            if test_opt.post_process:
                test_results += '_post-process'

            test_results += name
            test_results += '/'
            mkdirs(test_results)

            type_list = [
                'jpg',
                'gaussian-blur',
                'resize'
            ]

            parameter_list = [
                [95, 90, 85, 80, 75, 70],
                [[1, 0.5], [1, 1], [1, 1.5],
                 [3, 0.5], [3, 1.0], [3, 1.5]],
                [40, 60, 80, 120, 140, 160, ],
            ]

            filename = ''
            if test_opt.post_process:
                for type_i, pro_type in enumerate(type_list):
                    for para in parameter_list[type_i]:
                        print("[{}/{}] Testing on {}. Reference Number: {}".format(shot + 1, shot_num, exp,
                                                                                   test_opt.sample_num))
                        filename = ''
                        label_pred = [['y_true', 'y_pred', 'relation_score']]
                        test_opt.process_type = pro_type
                        test_opt.process_parameter = para

                        print("Operation:{}, factor:{}".format(test_opt.process_type, str(test_opt.process_parameter)))

                        val_acc, _, y_true, y_pred, relation_score = \
                            validate(test_opt, mode="test")
                        val_acc *= 100
                        relation_score = relation_score.data
                        for i in range(len(y_true)):
                            line = [y_true[i].item(), y_pred[i].item()]
                            line.extend(relation_score[i].tolist())
                            label_pred.append(line)
                        dataset = test_opt.testsets.split(',')
                        # print("Test sets:", dataset)
                        print(val_acc)
                        summary = [exp, pro_type, para, val_acc]
                        total_summary.append(summary)

                        filename = '/{}_{}_{}.csv'.format(name[0], pro_type, str(para))
                        label_pre_name = test_results + filename
                        print(label_pre_name)
                        with open(label_pre_name, 'w', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',')
                            csv_writer.writerows(label_pred)

            else:
                print("Testing on {}".format(exp))
                filename = ''
                label_pred = [['y_true', 'y_pred', 'relation_score']]
                val_acc, _, y_true, y_pred, relation_score = \
                    validate(test_opt, mode="test")
                val_acc *= 100
                relation_score = relation_score.data
                for i in range(len(y_true)):
                    line = [y_true[i].item(), y_pred[i].item()]
                    line.extend(relation_score[i].tolist())
                    label_pred.append(line)
                dataset = test_opt.testsets.split(',')
                print("Validation Accuracy：", val_acc)
                summary = [exp, val_acc]
                total_summary.append(summary)

                filename = '/{}_vs_{}.csv'.format(dataset[0].strip(), dataset[1].strip())

                label_pre_name = test_results + filename
                # print(label_pre_name)
                with open(label_pre_name, 'w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerows(label_pred)

            summary_name = "./results/{}/summary.csv".format(time)
            print("Summary csv: {}".format(summary_name))
            with open(summary_name, 'w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerows(total_summary)
