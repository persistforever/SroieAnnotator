# _*_ coding: utf-8 _*_
# author: ronniecao
# time: 2021/07/26
# description: util tools of template matching
import os
import sys
import cv2
import logging
import json
import psutil
import re
import math
import json
import operator
import numpy
import multiprocessing as mp
from multiprocessing.sharedctypes import Array, Value
from ctypes import c_double, cast, POINTER


def calculate_euclidean_distance(boxa, boxb):
    """
    计算两个box的欧氏距离
    """
    xcenter_a = 1.0 * (boxa[0] + boxa[2]) / 2.0
    ycenter_a = 1.0 * (boxa[1] + boxa[3]) / 2.0
    xcenter_b = 1.0 * (boxb[0] + boxb[2]) / 2.0
    ycenter_b = 1.0 * (boxb[1] + boxb[3]) / 2.0

    euc_dist = 0.5 * math.sqrt(
        (xcenter_a - xcenter_b) ** 2 + (ycenter_a - ycenter_b) ** 2)

    return euc_dist

def calculate_angle(boxa, boxb):
    """
    计算两个box的夹角
    """
    xcenter_a = 1.0 * (boxa[0] + boxa[2]) / 2.0
    ycenter_a = 1.0 * (boxa[1] + boxa[3]) / 2.0
    xcenter_b = 1.0 * (boxb[0] + boxb[2]) / 2.0
    ycenter_b = 1.0 * (boxb[1] + boxb[3]) / 2.0

    x_dist = xcenter_b - xcenter_a
    y_dist = ycenter_b - ycenter_a

    angle = math.degrees(math.atan2(y_dist, x_dist)) + 180

    return angle