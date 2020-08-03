# -*- coding: utf-8 -*-
from __future__ import print_function
import wx
import json
import os
import numpy
import zipfile
import time
import cv2
import pickle

def read_datas():
	mode = 'train'
	
	name_dict = {}
	# with open(os.path.join(mode, 'ids.txt'), 'r') as fo:
		# for line in fo:
			# name_dict[line.strip()] = None

	datas = {}
	for file_name in os.listdir(os.path.join(mode, 'pic')):
		name = file_name.split('.')[0]
		# if name not in name_dict:
			# continue
		pic_path = os.path.join(mode, 'pic', '%s.jpg' % (name))
		ocr_path = os.path.join(mode, 'ocr', '%s.txt' % (name))
		pkl_path = os.path.join(mode, 'data', '%s.pkl' % (name))
		if os.path.exists(pic_path) and \
			os.path.exists(ocr_path):
			datas[name] = {'pic_path': pic_path, 
				'ocr_path': ocr_path, 'pkl_path': pkl_path}

	for name in datas:
		pic_path = datas[name]['pic_path']
		image = numpy.array(cv2.imread(pic_path))[:, :, 0:3]
		ocr_path = datas[name]['ocr_path']
		words, chars = [], []
		with open(ocr_path, 'r') as fo:
			for line in fo:
				res = line.strip().split(',')
				if len(res) >= 9:
					[left, top, _, _, right, bottom, _, _] = [int(t) for t in res[0:8]]
					text = ','.join(res[8:])
					width = 1.0 * (right - left) / len(text)
					contain_chars = []
					for i in range(len(text)):
						start_x = left + i * width
						char = {'text': text[i], 'index': len(chars),
							'box': [int(start_x), int(top), int(start_x + width), int(bottom)]}
						contain_chars.append(char['index'])
						chars.append(char)
					word = {}
					for i, cidx in enumerate(contain_chars):
						if len(chars[cidx]['text'].strip()) == 0:
							if word != {}:
								words.append(word)
								word = {}
						elif chars[cidx]['text'] in [':']:
							if word != {}:
								words.append(word)
								word = {'contain_chars': [cidx]}
								words.append(word)
								word = {}
						else:
							if 'contain_chars' not in word:
								word['contain_chars'] = []
							word['contain_chars'].append(cidx)
					if word != {}:
						words.append(word)

		for i, word in enumerate(words):
			left = min([chars[cidx]['box'][0] for cidx in word['contain_chars']])
			top = min([chars[cidx]['box'][1] for cidx in word['contain_chars']])
			right = max([chars[cidx]['box'][2] for cidx in word['contain_chars']])
			bottom = max([chars[cidx]['box'][3] for cidx in word['contain_chars']])
			middle = 1.0 * (bottom + top) / 2.0
			height = (bottom - top) * 0.5
			new_top = middle - height / 2.0
			new_bottom = middle + height / 2.0
			word['index'] = i
			word['box'] = [int(left), int(top), int(right), int(bottom)]
			word['adjust_box'] = [int(left), int(new_top), int(right), int(new_bottom)]
			word['text'] = ''.join([chars[cidx]['text'] for cidx in word['contain_chars']])
			word['ss_label'] = 'back'

		data = {'name': name, 'pages': {0: 
			{'words': words, 'chars': chars, 'size': [image.shape[1], image.shape[0]]}}}
		pickle.dump(data, open(datas[name]['pkl_path'], 'wb'))

read_datas()