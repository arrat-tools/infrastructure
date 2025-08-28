#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:07:24 2022

@author: trcadmin
"""

import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS



LIST_FILE = {
    'val': 'list/val.txt',
}


@DATASETS.register_module
class custom(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading data...')
        

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)
                

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
            

        return infos

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        

        return 
