# -*- coding: utf8 -*-
# pylint: skip-file
# author: ronniecao
# time: 2021/07/26
# description: start script of template matching
from __future__ import print_function
import argparse
import os
import random
import copy
import json
import logging
import yaml
import numpy
import wx
import wx.lib.scrolledpanel as scrolled
from template.data import Processor
from template.model import Model


class TemplateMatcher:
    def __init__(self, option=None, data_dir=None, output_dir=None):

        self.data_dir = data_dir

        # 读取配置
        self.option = option
        print(self.option)

        # 设置seed
        random.seed(self.option['option']['seed'])
        numpy.random.seed(self.option['option']['seed'])

        # 实例化数据准备模块
        self.processor = Processor(
            option=self.option,
            backup_dir=output_dir,
            logs_dir=output_dir,
        )
        logging.info('Create Processor instance processor')

        # 实例化模型模块
        self.model = Model(
            option=self.option,
            backup_dir=output_dir,
            logs_dir=output_dir,
        )
        logging.info('Create Model instance model')

        # 读取数据，开始运行
        self.model.init_window()
        self.processor.init_datasets(main_dir=self.data_dir)
        self.model.init_model(processor=self.processor)
        """
        for match_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
            for n_neighbour in [5, 10]:
                self.option['option']['n_neighbour'] = n_neighbour
                self.option['option']['match_ratio'] = match_ratio
                self.model.run()
        """


if __name__ == '__main__':
    print('current process id: %d' % (os.getpid()))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-name')
    parser.add_argument('-config')
    arg = parser.parse_args()
    config_path = arg.config
    mtd = arg.method

    if True:
        config_path = 'scripts/logs/extract-v34/option.yaml'
        mtd = 'matching'

    if mtd in ['matching']:
        option = yaml.load(open(config_path, 'r'))
        tm = TemplateMatcher(
            option=option,
            data_dir=option['option']['data_dir'],
            output_dir=os.path.join(option['option']['logs_dir'], option['option']['seq']),
        )
        tm.model.frame.Show()
        tm.model.app.MainLoop()