# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/07/26
# description: data processor of template matching
from __future__ import print_function
import os
import time
import random
import copy
import json
import logging
import multiprocessing as mp
import cv2
import numpy
import template.utils as utils


"""
数据准备类：对数据进行预处理、数据增强等过程
"""
class Processor:
    def __init__(self,
        option,
        backup_dir,
        logs_dir):

        # 读取配置
        self.option = option
        self.backup_dir = backup_dir
        self.logs_dir = logs_dir

    def init_datasets(self, main_dir):
        """
        读取数据
        """
        # 读取item
        self.datasets = []
        for tplid in os.listdir(main_dir):
            for docid in os.listdir(os.path.join(main_dir, tplid)):
                data_path = os.path.join(main_dir, tplid, docid, '%s.json' % (docid))
                pic_path = os.path.join(main_dir, tplid, docid, '%s.jpg' % (docid))
                if os.path.exists(data_path):
                    item = {'index': len(self.datasets), 'docid': docid,
                        'data_path': data_path, 'pic_path': pic_path, 'tplid': tplid}
                    self.datasets.append(item)

        # 随机打乱
        random.shuffle(self.datasets)

        print('init datasets: %d' % (len(self.datasets)))

    def get_data_from_disk(self, item):
        """
        从硬盘读取数据
        """
        item['content'] = json.load(open(item['data_path'], 'r'))
        self.get_words_from_item(item)
        if item['pic_path'] != '' and os.path.exists(item['pic_path']):
            item['pic'] = numpy.array(cv2.imread(item['pic_path']))[:, :, 0:3]
        else:
            item['pic'] = None
        item['content']['neighbour_dict'], item['content']['edge_dict'] = \
            self.calculate_edge(item)

    def get_words_from_item(self, item):
        """
        读取所有words
        """
        # 获取页面范围
        page_left, page_top, page_right, page_bottom = 10000, 10000, 0, 0
        for word in item['content']['words']:
            page_left = min(page_left, word['box'][0])
            page_top = min(page_top, word['box'][1])
            page_right = max(page_right, word['box'][2])
            page_bottom = max(page_bottom, word['box'][3])
        item['content']['page_width'] = page_right - page_left
        item['content']['page_height'] = page_bottom - page_top

    def calculate_edge(self, item):
        """
        计算每条边
        """
        # KNN方法确定每个word的neighbour
        neighbour_dict = {}
        for wida, worda in enumerate(item['content']['words']):
            neighbours = []
            for widb, wordb in enumerate(item['content']['words']):
                if wida == widb:
                    continue
                euc_dist = utils.calculate_euclidean_distance(worda['box'], wordb['box'])
                neighbours.append([widb, euc_dist])
            neighbours = sorted(neighbours, key=lambda x: x[1])
            neighbour_dict[wida] = neighbours[0: self.option['option']['n_neighbour']]

        # 创建边的字典，边的两个节点按升序排列
        edge_dict = {}
        for wida in neighbour_dict:
            for widb, euc_dist in neighbour_dict[wida]:
                edge = sorted([wida, widb])
                edge_string = '%d&%d' % (edge[0], edge[1])
                edge_dict[edge_string] = euc_dist

        return neighbour_dict, edge_dict