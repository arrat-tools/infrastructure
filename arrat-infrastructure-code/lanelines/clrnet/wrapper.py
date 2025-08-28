#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:54:56 2022

@author: Punit Tulpule
"""
#Wrapper script for evaluating clrnet with different backbones and training datasets



config_options = ['culane,dla34',
                  'culane,resnet34',
                  'culane,resnet101',
                  'tusimple,resnet18']

config_select = 1
dataset_path = './Data/i70-set1'
if config_select ==1:
    runfile('main.py', 'configs/clrnet/custom_cla_dlr34_culane.py --validate --load_from pretrained/culane_dla34.pth --gpus 0 --view --data_path '+dataset_path)
elif config_select ==2:
    runfile('main.py', 'configs/clrnet/custom_clr_resnet34_culane.py --validate --load_from pretrained/culane_r34.pth --gpus 0 --view --data_path '+dataset_path)
elif config_select ==3:
    runfile('main.py', 'configs/clrnet/custom_clr_resnet101_culane.py --validate --load_from pretrained/culane_r101.pth --gpus 0 --view --data_path '+dataset_path)
else:
    runfile('main.py', 'configs/clrnet/custom_clr_resnet18_tusimple.py --validate --load_from pretrained/tusimple_r18.pth --gpus 0 --view --data_path '+dataset_path)
