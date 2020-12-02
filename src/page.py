# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import json
import math
import collections
import time
import pickle
import random
import shutil
from multiprocessing import Process, Queue, Value, Array
import cv2
import numpy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics


def grater_data():
    source_dir = '/home/caory/data/dpp_data/pfb_grater_answer/'
    target_train_dir = '/home/caory/data/datasets/extract_information/20200722/train_data/'
    target_valid_dir = '/home/caory/data/datasets/extract_information/20200722/valid_data/'
    file_list = Queue(maxsize=10000)
    for doc_name in os.listdir(os.path.join(source_dir, 'answers')):
        docid = doc_name.split('.')[0]
        file_list.put(docid)
    print('file_list:', file_list.qsize())

    def convert(file_list, length):
        while not file_list.empty():
            print(colored(file_list.qsize(), 'red'), colored(length, 'blue'))
            docid = file_list.get()
            pdf_path = os.path.join(source_dir, 'pdfs', '%s.pdf' % (docid))
            data_path = os.path.join(source_dir, 'answers', '%s.zip' % (docid))
            if os.path.exists(pdf_path):
                data = json.loads(read_zip_first_file(data_path))['pages']['1']
                doc = PDFDoc(pdf_path)
                page_data = {'docid': docid, 'words': [], 'information': {}, 'size': data['size']}
                page_image = numpy.array(doc.images[1])
                # 移除pairs外的文本
                if 'information' in data:
                    if 'pairs' in data:
                        new_texts = []
                        for text in data['texts']:
                            [tleft, ttop, tright, tbottom] = text['box']
                            for pair in data['pairs']:
                                [pleft, ptop, pright, pbottom] = pair['outline']
                                if pleft <= tleft <= tright <= pright and \
                                    ptop <= ttop <= tbottom <= pbottom:
                                    new_texts.append(text)
                                    break
                        texts = new_texts
                    else:
                        texts = data['texts']
                    words = []
                    for text in texts:
                        for char in WordUtil.iter_words_from_texts([text]):
                            if len(char['text']) > 0:
                                words.append({'box': char['box'], 'text': char['text']})
                    page_data['words'] = words
                    information = data['information']
                    for key in information:
                        if key.startswith('ext') or 'key' not in information[key] or \
                            'value' not in information[key]:
                            continue
                        [kleft, ktop, kright, kbottom] = information[key]['key'][0]['box']
                        [vleft, vtop, vright, vbottom] = information[key]['value'][0]['box']
                        contain_key_words, contain_value_words = [], []
                        for j, word_dict in enumerate(words):
                            [l, t, r, b] = word_dict['box']
                            kcl = max(kleft, l)
                            kct = max(ktop, t)
                            kcr = min(kright, r)
                            kcb = min(kbottom, b)
                            if kcr >= kcl and kcb >= kct:
                                carea = 1.0 * (kcr - kcl) * (kcb - kct)
                                area = 1.0 * (r - l) * (b - t)
                                ratio = 1.0 * carea / area if area != 0.0 else 0.0
                                if ratio >= 0.8:
                                    contain_key_words.append(j)
                            vcl = max(vleft, l)
                            vct = max(vtop, t)
                            vcr = min(vright, r)
                            vcb = min(vbottom, b)
                            if vcr >= vcl and vcb >= vct:
                                carea = 1.0 * (vcr - vcl) * (vcb - vct)
                                area = 1.0 * (r - l) * (b - t)
                                ratio = 1.0 * carea / area if area != 0.0 else 0.0
                                if ratio >= 0.8:
                                    contain_value_words.append(j)
                        if contain_key_words and contain_value_words:
                            page_data['information'][key] = {
                                'key': [{
                                    'box': information[key]['key'][0]['box'],
                                    'text': information[key]['key'][0]['text'],
                                    'contain_words': contain_key_words,
                                }],
                                'value': [{
                                    'box': information[key]['value'][0]['box'],
                                    'text': information[key]['value'][0]['text'],
                                    'contain_words': contain_value_words,
                                }]
                            }
                    pageid = 0
                    if random.random() <= 0.9:
                        target_sub_dir = os.path.join(target_train_dir, docid)
                    else:
                        target_sub_dir = os.path.join(target_valid_dir, docid)
                    if not os.path.exists(target_sub_dir):
                        os.mkdir(target_sub_dir)
                    target_data_path = os.path.join(target_sub_dir, '%s_0.json' % (docid))
                    target_image_path = os.path.join(target_sub_dir, '%s_0.jpg' % (docid))
                    json.dump(page_data, open(target_data_path, 'w'), indent=4)
                    cv2.imwrite(target_image_path, page_image)
        
    process_list = []
    for i in range(40):
        process = Process(target=convert, args=(file_list, file_list.qsize()))
        process_list.append(process)
    for process in process_list:
        process.start()

def rvl_data():
    pic_dir = '/home/caory/data/datasets/extract_information/20200622/pic/'
    pkl_dir = '/home/caory/data/datasets/extract_information/20200622/new_data/'
    target_dir = '/home/caory/data/datasets/extract_information/20200622/test_data/'
    name_dict = {}
    for name in os.listdir(pkl_dir):
        [fname, ftype] = name.split('.')
        if fname not in name_dict:
            name_dict[fname] = {}
        name_dict[fname][ftype] = None
    for name in os.listdir(pic_dir):
        [fname, ftype] = name.split('.')
        if fname not in name_dict:
            name_dict[fname] = {}
        name_dict[fname][ftype] = None

    for fname in name_dict:
        print(fname)
        if 'pkl' in name_dict[fname] and 'jpg' in name_dict[fname]:
            os.mkdir(os.path.join(target_dir, fname))
            source_path = os.path.join(pkl_dir, '%s.pkl' % (fname))
            data = pickle.load(open(source_path, 'rb'))
            target_path = os.path.join(target_dir, fname, '%s_0.pkl' % (fname))
            pickle.dump(data['pages'][0], open(target_path, 'wb'))
            source_path = os.path.join(pic_dir, '%s.jpg' % (fname))
            target_path = os.path.join(target_dir, fname, '%s_0.jpg' % (fname))
            shutil.move(source_path, target_path)

def wiki_data():
    source_dir = '/home/caory/data/datasets/wiki_kv_dataset/zh_table/'
    target_train_dir = '/home/caory/data/datasets/extract_information/20200917/train_data/'
    target_valid_dir = '/home/caory/data/datasets/extract_information/20200917/valid_data/'
    digit_pattern = re.compile(r'[0-9]')
    file_list = Queue(maxsize=1000000)
    n_fold = 0
    for fold in os.listdir(source_dir):
        n_fold += 1
        if n_fold >= 1000:
            break
        for hash_name in os.listdir(os.path.join(source_dir, fold)):
            data_path = os.path.join(source_dir, fold, hash_name, 'dockv.json')
            pdf_path = os.path.join(source_dir, fold, hash_name, 'debug.pdf')
            tuple_path = os.path.join(source_dir, fold, hash_name, 'tablekv.json')
            if os.path.exists(data_path) and os.path.exists(pdf_path) and os.path.exists(tuple_path):
                file_list.put([fold, hash_name, data_path, pdf_path, tuple_path])
    print(file_list.qsize())

    def convert(file_list, length):
        while not file_list.empty():
            print(colored(file_list.qsize(), 'red'), colored(length, 'blue'))
            [fold, hash_name, data_path, pdf_path, tuple_path] = file_list.get()
            if random.random() <= 0.01:
                target_sub_dir = os.path.join(target_valid_dir, fold)
                target_sub_sub_dir = os.path.join(target_valid_dir, fold, hash_name)
            else:
                target_sub_dir = os.path.join(target_train_dir, fold)
                target_sub_sub_dir = os.path.join(target_train_dir, fold, hash_name)
            if not os.path.exists(target_sub_dir):
                os.mkdir(target_sub_dir)
            if not os.path.exists(target_sub_sub_dir):
                os.mkdir(target_sub_sub_dir)
            pageid = 0
            page_data = json.load(open(data_path, 'r'))
            tuple_data = json.load(open(tuple_path, 'r'))
            information = {}
            for tupleid, tuple_info in tuple_data['tuples'].items():
                if tuple_info['key_cell'] != [-1, -1] and \
                    tuple_info['value_cell'] != [-1, -1] and \
                    'cls_conv' in tuple_info and \
                    'key_conv' in tuple_info and \
                    'value_conv' in tuple_info:
                    cls = tuple_info['cls_conv']
                    key = tuple_info['key_conv']
                    value = tuple_info['value_conv']
                    [x, y] = tuple_info['key_cell']
                    key_contain_words, value_contain_words = [], []
                    if not digit_pattern.search(key):
                        for widx, word in enumerate(page_data['words']):
                            if word['ss_label'] == cls + '|k':
                                key_contain_words.append(widx)
                            elif word['ss_label'] == cls + '|v':
                                value_contain_words.append(widx)
                        information[cls] = {
                            'key': [{'text': key, 'contain_words': key_contain_words}],
                            'value': [{'text': value, 'contain_words': value_contain_words}]}
                    else:
                        for word in page_data['words']:
                            if word['ss_label'].split('|')[0] == cls:
                                word['ss_label'] = 'back'
            page_data['information'] = information
            doc = PDFDoc(pdf_path)
            page_image = numpy.array(doc.images[int(pageid)])
            target_data_path = os.path.join(target_sub_sub_dir, '%s_%d.json' % (hash_name, int(pageid)))
            target_image_path = os.path.join(target_sub_sub_dir, '%s_%d.jpg' % (hash_name, int(pageid)))
            json.dump(page_data, open(target_data_path, 'w'))
            cv2.imwrite(target_image_path, page_image)
        
    process_list = []
    for i in range(40):
        process = Process(target=convert, args=(file_list, file_list.qsize()))
        process_list.append(process)
    for process in process_list:
        process.start()

def wiki_table_data():
    source_dir = '/home/caory/data/datasets/wiki_kv_dataset/zh_table/'
    target_train_dir = '/home/caory/data/datasets/extract_information/20200917/train_data/'
    target_valid_dir = '/home/caory/data/datasets/extract_information/20200917/valid_data/'
    digit_pattern = re.compile(r'[0-9]')
    file_list = Queue(maxsize=1000000)
    n_fold = 0
    for fold in os.listdir(source_dir):
        n_fold += 1
        if n_fold >= 1000:
            break
        for hash_name in os.listdir(os.path.join(source_dir, fold)):
            data_path = os.path.join(source_dir, fold, hash_name, 'dockv.json')
            pdf_path = os.path.join(source_dir, fold, hash_name, 'debug.pdf')
            tuple_path = os.path.join(source_dir, fold, hash_name, 'tablekv.json')
            if os.path.exists(data_path) and os.path.exists(pdf_path) and os.path.exists(tuple_path):
                file_list.put([fold, hash_name, data_path, pdf_path, tuple_path])
    print(file_list.qsize())

    def convert(file_list, length):
        while not file_list.empty():
            print(colored(file_list.qsize(), 'red'), colored(length, 'blue'))
            [fold, hash_name, data_path, pdf_path, tuple_path] = file_list.get()
            if random.random() <= 0.01:
                target_sub_dir = os.path.join(target_valid_dir, fold)
                target_sub_sub_dir = os.path.join(target_valid_dir, fold, hash_name)
            else:
                target_sub_dir = os.path.join(target_train_dir, fold)
                target_sub_sub_dir = os.path.join(target_train_dir, fold, hash_name)
            if not os.path.exists(target_sub_dir):
                os.mkdir(target_sub_dir)
            if not os.path.exists(target_sub_sub_dir):
                os.mkdir(target_sub_sub_dir)
            pageid = 0
            page_data = json.load(open(data_path, 'r'))
            tuple_data = json.load(open(tuple_path, 'r'))
            information = {}
            for widx, word in enumerate(page_data['words']):
                for ss_label in word['ss_label']:
                    if len(ss_label.split('|')) > 1:
                        information[ss_label.split('|')[0]] = {}
            for cls in information:
                key_contain_words, value_contain_words = [], []
                for widx, word in enumerate(page_data['words']):
                    if cls + '|k' in word['ss_label']:
                        key_contain_words.append(widx)
                    elif cls + '|v'in word['ss_label']:
                        value_contain_words.append(widx)
                key = ''.join([page_data['words'][widx]['text'] \
                    for widx in key_contain_words])
                value = ''.join([page_data['words'][widx]['text'] \
                    for widx in value_contain_words])
                information[cls] = {
                    'key': [{'text': key, 'contain_words': key_contain_words}],
                    'value': [{'text': value, 'contain_words': value_contain_words}]}
            page_data['information'] = information
            doc = PDFDoc(pdf_path)
            page_image = numpy.array(doc.images[int(pageid)])
            target_data_path = os.path.join(target_sub_sub_dir, '%s_%d.json' % (hash_name, int(pageid)))
            target_image_path = os.path.join(target_sub_sub_dir, '%s_%d.jpg' % (hash_name, int(pageid)))
            json.dump(page_data, open(target_data_path, 'w'))
            cv2.imwrite(target_image_path, page_image)
        
    process_list = []
    for i in range(40):
        process = Process(target=convert, args=(file_list, file_list.qsize()))
        process_list.append(process)
    for process in process_list:
        process.start()

def wiki_data_to_hyp():
    source_dir = '/home/caory/data/datasets/wiki_kv_dataset/zh_infobox/'
    target_dir = '/home/caory/data/datasets/wiki_kv_dataset/zh_infobox_to_hyp/'
    file_list = Queue(maxsize=1000000)
    for fold in os.listdir(source_dir):
        for hash_name in os.listdir(os.path.join(source_dir, fold)):
            tuple_path = os.path.join(source_dir, fold, hash_name, 'tablekv.json')
            if os.path.exists(tuple_path):
                file_list.put([fold, hash_name, tuple_path])
    print(file_list.qsize())

    def convert(file_list, length):
        while not file_list.empty():
            print(colored(file_list.qsize(), 'red'), colored(length, 'blue'))
            [fold, hash_name, tuple_path] = file_list.get()
            target_sub_dir = os.path.join(target_dir, fold)
            if not os.path.exists(target_sub_dir):
                os.mkdir(target_sub_dir)
            target_path = os.path.join(target_dir, fold, '%s.json' % (hash_name))
            shutil.copy(tuple_path, target_path)
    
    process_list = []
    for i in range(20):
        process = Process(target=convert, args=(file_list, file_list.qsize()))
        process_list.append(process)
    for process in process_list:
        process.start()

def sroie_data():
    source_dir = '/home/caory/github/SroieAnnotator/data/train/new_data_20200731/'
    target_dir = '/home/caory/data/datasets/extract_information/20201120/train_data_random_split/'
    data_list = []
    n = 0
    for docid in os.listdir(source_dir):
        n += 1
        print(n, docid)
        data_path = os.path.join(source_dir, docid, '%s_0.json' % (docid))
        image_path = os.path.join(source_dir, docid, '%s_0.jpg' % (docid))
        if os.path.exists(data_path) and os.path.exists(image_path):
            data = json.load(open(data_path))
            if 'image' in data:
                del data['image']
            data['information'] = {}
            for widx, word in enumerate(data['words']):
                if word['ss_label'] != 'back':
                    [cls, ktype] = word['ss_label'].split('|')
                    if cls not in data['information']:
                        data['information'][cls] = {
                            'key': [{'contain_words': [], 'text': ''}],
                            'value': [{'contain_words': [], 'text': ''}]}
                    if ktype == 'k':
                        data['information'][cls]['key'][0]['contain_words'].append(widx)
                    elif ktype == 'v':
                        data['information'][cls]['value'][0]['contain_words'].append(widx)
            for cls in data['information']:
                for sub_key in ['key', 'value']:
                    info = data['information'][cls][sub_key][0]
                    info['text'] = '<#>'.join([
                        data['words'][widx]['text'] for widx in info['contain_words']])
            data_list.append([docid, data, image_path])
    random.shuffle(data_list)
    for docid, data, image_path in data_list[0:20]:
        if not os.path.exists(os.path.join(target_dir, docid)):
            os.mkdir(os.path.join(target_dir, docid))
        target_path = os.path.join(target_dir, docid, '%s_0.json' % (docid))
        target_image_path = os.path.join(target_dir, docid, '%s_0.jpg' % (docid))
        json.dump(data, open(target_path, 'w'), indent=4)
        shutil.copy(image_path, target_image_path)

def get_new_data(data):
    data['information'] = {}
    for widx, word in enumerate(data['words']):
        word['box'] = word['orig_box']
        for ss_label in word['ss_label']:
            if ss_label != 'back':
                [cls, ktype] = ss_label.split('|')
                if cls not in data['information']:
                    data['information'][cls] = {
                        'key': [{'contain_words': [], 'text': ''}],
                        'value': [{'contain_words': [], 'text': ''}]}
                if ktype == 'k':
                    data['information'][cls]['key'][0]['contain_words'].append(widx)
                elif ktype == 'v':
                    data['information'][cls]['value'][0]['contain_words'].append(widx)
    for cls in data['information']:
        for sub_key in ['key', 'value']:
            info = data['information'][cls][sub_key][0]
            info['text'] = '<#>'.join([
                data['words'][widx]['text'] for widx in info['contain_words']])

    return data

def fcc_data():
    source_dir = '/home/caory/github/SroieAnnotator/fcc_data/data/'
    target_train_dir = '/home/caory/data/datasets/extract_information/20201117/train_data/'
    target_valid_dir = '/home/caory/data/datasets/extract_information/20201117/valid_data/'
    file_list = []
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    file_list.append([fname, docid, data_path, image_path])

    random.shuffle(file_list)
    n_train = 10
    train_list = file_list[0:n_train]
    valid_list = file_list[n_train:]
    n = 0
    for flist, target_dir in [[train_list, target_train_dir], [valid_list, target_valid_dir]]:
        for fname, docid, data_path, image_path in flist:
            n += 1
            print(n, fname)
            data = json.load(open(data_path))
            if data.get('is_wrong'):
                continue
            if 'image' in data:
                del data['image']
            data = get_new_data(data)
            if not os.path.exists(os.path.join(target_dir, docid)):
                os.mkdir(os.path.join(target_dir, docid))
            target_path = os.path.join(target_dir, docid, fname)
            target_image_path = os.path.join(target_dir, docid, fname.replace('json', 'jpg'))
            json.dump(data, open(target_path, 'w'), indent=4)
            shutil.copy(image_path, target_image_path)

def grater_statistic():
    source_train_dir = '/home/caory/data/datasets/extract_information/20200722/train_data/'
    source_valid_dir = '/home/caory/data/datasets/extract_information/20200722/valid_data/'
    target_dir = '/home/caory/data/datasets/extract_information/20200722/cls_dict/'
    file_list = Queue(maxsize=1000000)
    n_doc = Value('i', 0)
    n_tuple = Value('i', 0)
    if False:
        for source_dir in [source_train_dir, source_valid_dir]:
            for docid in os.listdir(source_dir):
                data_path = os.path.join(source_dir, docid, '%s_0.json' % (docid))
                image_path = os.path.join(source_dir, docid, '%s_0.jpg' % (docid))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    file_list.put([docid, data_path, image_path])
    else:
        for source_dir in [source_train_dir, source_valid_dir]:
            for fold in os.listdir(source_dir):
                for docid in os.listdir(os.path.join(source_dir, fold)):
                    data_path = os.path.join(source_dir, fold, docid, '%s_0.json' % (docid))
                    image_path = os.path.join(source_dir, fold, docid, '%s_0.jpg' % (docid))
                    if os.path.exists(data_path) and os.path.exists(image_path):
                        file_list.put([docid, data_path, image_path])
    print(file_list.qsize())

    def count(file_list, length):
        while not file_list.empty():
            print(colored(file_list.qsize(), 'red'), colored(length, 'blue'))
            [docid, data_path, image_path] = file_list.get()
            data = json.load(open(data_path, 'r'))
            cls_dict = {}
            if 'information' in data:
                for key in data['information']:
                    if key.startswith('ext'):
                        continue
                    if key not in cls_dict:
                        cls_dict[key] = {}
                    words = data['words']
                    text = '<#>'.join([words[widx]['text'] for widx in \
                        data['information'][key]['key'][0]['contain_words']])
                    if text not in cls_dict[key]:
                        cls_dict[key][text] = 0
                    cls_dict[key][text] += 1
                    n_tuple.value += 1
                target_path = os.path.join(target_dir, '%s.json' % (docid))
                json.dump(cls_dict, open(target_path, 'w'), indent=4)
                n_doc.value += 1
            print('n_doc: %d, n_tuple: %d' % (n_doc.value, n_tuple.value))
    
    process_list = []
    for i in range(40):
        process = Process(target=count, args=(file_list, file_list.qsize()))
        process_list.append(process)
    for process in process_list:
        process.start()

def wiki_statistic():
    source_train_dir = '/home/caory/data/datasets/extract_information/20201110/train_data/'
    source_valid_dir = '/home/caory/data/datasets/extract_information/20201110/valid_data/'
    cls_path = '/home/caory/data/datasets/extract_information/20201110/classes.json'
    file_list = []
    if True:
        for source_dir in [source_train_dir, source_valid_dir]:
            for docid in os.listdir(source_dir):
                for fname in os.listdir(os.path.join(source_dir, docid)):
                    if fname.split('.')[1] == 'json':
                        data_path = os.path.join(source_dir, docid, fname)
                        image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                        if os.path.exists(data_path) and os.path.exists(image_path):
                            file_list.append([docid, data_path, image_path])
    else:
        for source_dir in [source_train_dir, source_valid_dir]:
            for fold in os.listdir(source_dir):
                for docid in os.listdir(os.path.join(source_dir, fold)):
                    for fname in os.listdir(os.path.join(source_dir, fold, docid)):
                        if fname.split('.')[1] == 'json':
                            data_path = os.path.join(source_dir, fold, docid, fname)
                            image_path = os.path.join(source_dir, fold, docid, fname.replace('json', 'jpg'))
                            if os.path.exists(data_path) and os.path.exists(image_path):
                                file_list.append([docid, data_path, image_path])

    cls_dict = {}
    for i, [docid, data_path, image_path] in enumerate(file_list):
        if i % (len(file_list) // 10) == 0:
            print('{read data} [%d/%d]' % (i, len(file_list)))
        data = json.load(open(data_path, 'r'))
        if 'information' in data:
            for key in data['information']:
                if key.startswith('ext'):
                    continue
                if key not in cls_dict:
                    cls_dict[key] = {'query': {}, 'doc': []}
                words = data['words']
                text = '<#>'.join([words[widx]['text'] for widx in \
                    data['information'][key]['key'][0]['contain_words']])
                if text not in cls_dict[key]['query']:
                    cls_dict[key]['query'][text] = 0
                cls_dict[key]['query'][text] += 1
                cls_dict[key]['doc'].append(docid)
    n_query, n_sample, n_pair = 0, 0, 0
    for cls in cls_dict:
        nq = len(cls_dict[cls]['query'])
        nd = len(cls_dict[cls]['doc'])
        n_query += nq
        n_sample += nq * nd
        n_pair += nd
    
    print('n_doc:', len(file_list), 'n_cls:', len(cls_dict), 
        'n_query:', n_query, 'n_pair:', n_pair, 'n_sample:', n_sample)

    query_dict = {}
    for cls in cls_dict:
        [query, num] = max(cls_dict[cls]['query'].items(), key=lambda x: x[1])
        query_dict[cls] = query
    json.dump(query_dict, open(cls_path, 'w'), indent=4)

def get_keywords():
    source_dir = '/home/caory/github/SroieAnnotator/fcc_data/data/'
    pic_dir = '/home/caory/github/SroieAnnotator/fcc_data/pic/'
    cluster_dir = '/home/caory/github/SroieAnnotator/fcc_data/cluster/'
    n = 0
    data_dict = {}
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    data_dict[fname] = {'docid': docid, 'data': data, 'data_path': data_path, 'image_path': image_path}
                    keyword_dict = {}
                    for kkey in ['num|k', 'ffrom|k', 'fto|k', 'adv|k', 'total|k']:
                        keyword_dict[kkey] = {'contain_widx': [], 'box': []}
                    for kkey in keyword_dict:
                        for widx, word in enumerate(data['words']):
                            if kkey in word['ss_label']:
                                keyword_dict[kkey]['contain_widx'].append(widx)
                        if keyword_dict[kkey]['contain_widx']:
                            left = min([data['words'][widx]['box'][0] for widx in keyword_dict[kkey]['contain_widx']])
                            top = min([data['words'][widx]['box'][1] for widx in keyword_dict[kkey]['contain_widx']])
                            right = max([data['words'][widx]['box'][2] for widx in keyword_dict[kkey]['contain_widx']])
                            bottom = max([data['words'][widx]['box'][3] for widx in keyword_dict[kkey]['contain_widx']])
                            keyword_dict[kkey]['box'] = [left, top, right, bottom]
                        else:
                            keyword_dict[kkey]['box'] = [0, 0, 0, 0]
                    features = []
                    for i in range(5):
                        for j in range(i+1, 5):
                            keya = ['num|k', 'ffrom|k', 'fto|k', 'adv|k', 'total|k'][i]
                            keyb = ['num|k', 'ffrom|k', 'fto|k', 'adv|k', 'total|k'][j]
                            [la, ta, ra, ba] = keyword_dict[keya]['box']
                            [lb, tb, rb, bb] = keyword_dict[keyb]['box']
                            ldist = 1.0 * (la - lb) / data['size'][1]
                            tdist = 1.0 * (ta - tb) / data['size'][0]
                            rdist = 1.0 * (ra - rb) / data['size'][1]
                            bdist = 1.0 * (ba - bb) / data['size'][0]
                            features.extend([ldist, tdist, rdist, bdist])
                    data['features'] = features

    # 获取index
    fname2index, index2fname = {}, {}
    feature_list = []
    n = 0
    for fname in data_dict:
        fname2index[fname] = n
        index2fname[n] = fname
        feature_list.append(data_dict[fname]['data']['features'])
        n += 1
    feature_list = numpy.array(feature_list)
    print(feature_list.shape, len(index2fname))

    # 聚类
    clustering = KMeans(n_clusters=10, max_iter=1000).fit(feature_list)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(index2fname[i])
    for label in clusters:
        print(label, len(clusters[label]))

    if False:
        # 根据聚类结果拷贝图片
        for label in clusters:
            print('output %d: %d' % (label, len(clusters[label])))
            target_dir = os.path.join(cluster_dir, str(label))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            for fname in clusters[label]:
                target_path = os.path.join(target_dir, fname).replace('json', 'jpg')
                source_path = data_dict[fname]['image_path']
                print(target_path, source_path)
                shutil.copy(source_path, target_path)

    # 根据聚类结果分配训练/测试集
    target_train_dir = '/home/caory/data/datasets/extract_information/20201118/train_data/'
    target_valid_dir = '/home/caory/data/datasets/extract_information/20201118/valid_data/'
    n = 0
    for label in clusters:
        random.shuffle(clusters[label])
        n_train = int(len(clusters[label]) * 0.1)
        train_fnames = clusters[label][0:n_train]
        valid_fnames = clusters[label][n_train:]
        for fnames, target_dir in [[train_fnames, target_train_dir], [valid_fnames, target_valid_dir]]:
            for fname in fnames:
                data_path = data_dict[fname]['data_path']
                image_path = data_dict[fname]['image_path']
                docid = data_dict[fname]['docid']
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    data = get_new_data(data)
                    if not os.path.exists(os.path.join(target_dir, docid)):
                        os.mkdir(os.path.join(target_dir, docid))
                    target_path = os.path.join(target_dir, docid, fname)
                    target_image_path = os.path.join(target_dir, docid, fname.replace('json', 'jpg'))
                    json.dump(data, open(target_path, 'w'), indent=4)
                    shutil.copy(image_path, target_image_path)

def get_keywords_fcc_data():
    source_dir = '../fcc_data/data/'
    pic_dir = '../fcc_data/pic/'
    keyword_dir = '../fcc_data/keyword/'
    cluster_dir = '../fcc_data/cluster/'
    cluster_label_path = '../fcc_data/cluster_label/doc_cls.json'
    docid2label = json.load(open(cluster_label_path, 'r'))

    n = 0
    data_dict = {}
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    data_dict[fname] = {'docid': docid, 'data': data,
                        'data_path': data_path, 'image_path': image_path}
                    word_tfs = {}
                    for word_dict in data['words'][0:500]:
                        if not re.match(r'[A-Z]', word_dict['text'][0]):
                            continue
                        if word_dict['text'] not in word_tfs:
                            word_tfs[word_dict['text']] = 0
                        word_tfs[word_dict['text']] += 1
                    for word in word_tfs:
                        word_tfs[word] = 1.0 * word_tfs[word] / len(data['words'])
                    data_dict[fname]['word_tfs'] = word_tfs

    # 获取word的idf
    word_dict = {}
    for fname in data_dict:
        for word in data_dict[fname]['word_tfs']:
            if word not in word_dict:
                word_dict[word] = {'tfs': [], 'idf': 0}
            word_dict[word]['tfs'].append(data_dict[fname]['word_tfs'][word])
            word_dict[word]['idf'] += 1

    # 计算word的tf-idf并排序
    n_doc = len(data_dict)
    new_word_dict = {}
    for word in word_dict:
        if len(word) < 3 or len(word) > 30:
            continue
        tf_value = numpy.mean(word_dict[word]['tfs'])
        idf_value = math.log(1.0 * n_doc / word_dict[word]['idf'])
        tf_idf_value = tf_value * idf_value
        new_word_dict[word] = {'tf_value': tf_value, 'idf_value': idf_value,
            'tf_idf_value': tf_idf_value}
    word_dict = new_word_dict
    keyword_dict = collections.OrderedDict(
        sorted(word_dict.items(), key=lambda x: x[1]['tf_idf_value']))
    for word in keyword_dict:
        word_info = keyword_dict[word]
        # print(word, word_info['tf_value'], word_info['idf_value'], word_info['tf_idf_value'])
    keywords = list(keyword_dict.keys())
    # print(keywords)

    # 每篇文档找出前K个最高的词
    new_keywords = []
    if True:
        for i, fname in enumerate(data_dict):
            doc_keywords = {}
            for word in data_dict[fname]['data']['words']:
                if word['text'] in keyword_dict:
                    doc_keywords[word['text']] = keyword_dict[word['text']]
            doc_keywords = collections.OrderedDict(
                sorted(doc_keywords.items(), key=lambda x: x[1]['tf_idf_value'])[0:10])
            new_keywords.extend(list(doc_keywords.keys()))
        keywords = list(set(new_keywords))
    print('keywords number', len(keywords))

    # 画出keyword结果
    if False:
        for i, fname in enumerate(data_dict):
            print(i, 'print keyword:', fname)
            image = cv2.imread(data_dict[fname]['image_path'])
            for word in data_dict[fname]['data']['words']:
                if word['text'] in keywords:
                    [left, top, right, bottom] = [int(t) for t in word['box']]
                    cv2.rectangle(image, (left-2, top-2), (right+2, bottom+2), [255, 50, 50], 2)
            keyword_path = os.path.join(keyword_dir, '%s.jpg' % (fname))
            cv2.imwrite(keyword_path, image)

    # 获取index，计算features
    fname2index, index2fname = {}, {}
    feature_list = []
    n = 0
    for fname in data_dict:
        print('calculate feature', n)
        fname2index[fname] = n
        index2fname[n] = fname
        word_boxes = {}
        for word in keywords:
            word_boxes[word] = [0, 0, 0, 0]
        for word in data_dict[fname]['data']['words']:
            if word['text'] in word_boxes:
                word_boxes[word['text']] = [
                    1.0 * word['box'][0] / data_dict[fname]['data']['size'][1],
                    1.0 * word['box'][1] / data_dict[fname]['data']['size'][0],
                    1.0 * word['box'][2] / data_dict[fname]['data']['size'][1],
                    1.0 * word['box'][3] / data_dict[fname]['data']['size'][0]]
        feature = []
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                boxi = word_boxes[keywords[i]]
                boxj = word_boxes[keywords[j]]
                feature.extend([boxi[t] - boxj[t] for t in range(4)])
        feature_list.append(feature)
        n += 1
    feature_list = numpy.array(feature_list)
    print(feature_list.shape, len(index2fname))

    # 聚类
    clustering = KMeans(n_clusters=10, max_iter=5000).fit(feature_list)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(index2fname[i])
    for label in clusters:
        print(label, len(clusters[label]))

    # 根据聚类结果计算评估指标
    true_label = []
    for i, label in enumerate(clustering.labels_):
        name = index2fname[i].split('.')[0]
        true_label.append(int(docid2label['-'.join(name.split('_'))]))
    true_label = numpy.array(true_label)
    score = metrics.normalized_mutual_info_score(true_label, clustering.labels_)
    print(score)

    if False:
        # 根据聚类结果拷贝图片
        for label in clusters:
            print('output %d: %d' % (label, len(clusters[label])))
            target_dir = os.path.join(cluster_dir, str(label))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            for fname in clusters[label]:
                target_path = os.path.join(target_dir, fname).replace('json', 'jpg')
                source_path = data_dict[fname]['image_path']
                print(target_path, source_path)
                shutil.copy(source_path, target_path)
        # 保存feature
        feature_path = os.path.join(cluster_dir, 'features.json')
        feature_dict = {}
        for i, label in enumerate(clustering.labels_):
            fname = index2fname[i]
            feature = [round(t, 8) for t in feature_list[i,:].tolist()]
            feature_dict[fname] = {'label': int(label), 'feature': feature}
        json.dump(feature_dict, open(feature_path, 'w'), indent=4)

    if False:
        # 根据聚类结果分配训练/测试集
        target_train_dir = '/home/caory/data/datasets/extract_information/20201119/train_data/'
        target_valid_dir = '/home/caory/data/datasets/extract_information/20201119/valid_data/'
        n = 0
        for label in clusters:
            random.shuffle(clusters[label])
            n_train = 1
            train_fnames = clusters[label][0:n_train]
            valid_fnames = clusters[label][n_train:]
            for fnames, target_dir in [[train_fnames, target_train_dir], [valid_fnames, target_valid_dir]]:
                for fname in fnames:
                    data_path = data_dict[fname]['data_path']
                    image_path = data_dict[fname]['image_path']
                    docid = data_dict[fname]['docid']
                    if os.path.exists(data_path) and os.path.exists(image_path):
                        n += 1
                        print(n, fname)
                        data = json.load(open(data_path))
                        if data.get('is_wrong'):
                            continue
                        if 'image' in data:
                            del data['image']
                        data = get_new_data(data)
                        if not os.path.exists(os.path.join(target_dir, docid)):
                            os.mkdir(os.path.join(target_dir, docid))
                        target_path = os.path.join(target_dir, docid, fname)
                        target_image_path = os.path.join(
                            target_dir, docid, fname.replace('json', 'jpg'))
                        json.dump(data, open(target_path, 'w'), indent=4)
                        shutil.copy(image_path, target_image_path)

def get_keywords_grater_data():
    source_dir = '/home/caory/github/SroieAnnotator/grater_data/data/'
    target_dir_random = '/home/caory/data/datasets/extract_information/20201123/train_data_random_split/'
    target_dir_cluster = '/home/caory/data/datasets/extract_information/20201123/train_data_cluster_split/'
    target_dir_valid = '/home/caory/data/datasets/extract_information/20201123/valid_data/'
    cluster_dir = '/home/caory/github/SroieAnnotator/grater_data/cluster/'
    n = 0
    docid2cls = json.load(open('/home/caory/github/SroieAnnotator/grater_data/bank_name.json', 'r'))
    print(docid2cls)
    cls_dict = {}
    data_dict = {}
    # random split
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    cls = docid2cls[docid]
                    data_dict[fname] = {'docid': docid, 'data': data, 'cls': cls,
                        'data_path': data_path, 'image_path': image_path}
                    if cls not in cls_dict:
                        cls_dict[cls] = []
                    cls_dict[cls].append(fname)
    fname_list_random, fname_list_cluster = [], []
    for cls in cls_dict:
        random.shuffle(cls_dict[cls])
        cls_dict[cls] = cls_dict[cls][0: random.randint(20, 100)]
        fname_list_cluster.append(cls_dict[cls][0])
        fname_list_random.extend(cls_dict[cls])
        print(cls, len(cls_dict[cls]))
    print('fname_list length:', len(fname_list_random))
    random.shuffle(fname_list_random)
    n = len(cls_dict) # 个数为cls_dict的个数
    # random split
    for fname in fname_list_random[0:n]:
        docid = data_dict[fname]['docid']
        data = data_dict[fname]['data']
        image_path = data_dict[fname]['image_path']
        if not os.path.exists(os.path.join(target_dir_random, docid)):
            os.mkdir(os.path.join(target_dir_random, docid))
        target_path = os.path.join(target_dir_random, docid, '%s_0.json' % (docid))
        target_image_path = os.path.join(target_dir_random, docid, '%s_0.jpg' % (docid))
        json.dump(data, open(target_path, 'w'), indent=4)
        shutil.copy(image_path, target_image_path)
    # cluster split
    for fname in fname_list_cluster:
        docid = data_dict[fname]['docid']
        data = data_dict[fname]['data']
        image_path = data_dict[fname]['image_path']
        if not os.path.exists(os.path.join(target_dir_cluster, docid)):
            os.mkdir(os.path.join(target_dir_cluster, docid))
        target_path = os.path.join(target_dir_cluster, docid, '%s_0.json' % (docid))
        target_image_path = os.path.join(target_dir_cluster, docid, '%s_0.jpg' % (docid))
        json.dump(data, open(target_path, 'w'), indent=4)
        shutil.copy(image_path, target_image_path)
    # all data
    for fname in fname_list_random:
        docid = data_dict[fname]['docid']
        data = data_dict[fname]['data']
        image_path = data_dict[fname]['image_path']
        if not os.path.exists(os.path.join(target_dir_valid, docid)):
            os.mkdir(os.path.join(target_dir_valid, docid))
        target_path = os.path.join(target_dir_valid, docid, '%s_0.json' % (docid))
        target_image_path = os.path.join(target_dir_valid, docid, '%s_0.jpg' % (docid))
        json.dump(data, open(target_path, 'w'), indent=4)
        shutil.copy(image_path, target_image_path)

def get_keywords_sroie_data():
    source_dir = '/home/caory/github/SroieAnnotator/data/train/new_data_20200731/'
    cluster_dir = '/home/caory/github/SroieAnnotator/data/cluster/'
    n = 0
    data_dict = {}
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    data_dict[fname] = {'docid': docid, 'data': data,
                        'data_path': data_path, 'image_path': image_path}
                    word_tfs = {}
                    for word_dict in data['words']:
                        if word_dict['text'] not in word_tfs:
                            word_tfs[word_dict['text']] = 0
                        word_tfs[word_dict['text']] += 1
                    for word in word_tfs:
                        word_tfs[word] = 1.0 * word_tfs[word] / len(data['words'])
                    data_dict[fname]['word_tfs'] = word_tfs

    # 获取word的idf
    word_dict = {}
    for fname in data_dict:
        for word in data_dict[fname]['word_tfs']:
            if word not in word_dict:
                word_dict[word] = {'tfs': [], 'idf': 0}
            word_dict[word]['tfs'].append(data_dict[fname]['word_tfs'][word])
            word_dict[word]['idf'] += 1

    # 计算word的tf-idf并排序
    n_doc = len(data_dict)
    new_word_dict = {}
    for word in word_dict:
        if len(word) < 3 or len(word) > 30:
            continue
        tf_value = numpy.mean(word_dict[word]['tfs'])
        idf_value = math.log(1.0 * n_doc / word_dict[word]['idf'])
        tf_idf_value = tf_value * idf_value
        new_word_dict[word] = {'tf_value': tf_value, 'idf_value': idf_value,
            'tf_idf_value': tf_idf_value}
    word_dict = new_word_dict
    keyword_dict = collections.OrderedDict(
        sorted(word_dict.items(), key=lambda x: x[1]['tf_idf_value'])[0:50])
    for word in keyword_dict:
        word_info = keyword_dict[word]
        print(word, word_info['tf_value'], word_info['idf_value'], word_info['tf_idf_value'])
    keywords = list(keyword_dict.keys())
    print(keywords)

    # 获取index，计算features
    fname2index, index2fname = {}, {}
    feature_list = []
    n = 0
    for fname in data_dict:
        fname2index[fname] = n
        index2fname[n] = fname
        word_boxes = {}
        for word in keywords:
            word_boxes[word] = [0, 0, 0, 0]
        for word in data_dict[fname]['data']['words']:
            if word['text'] in word_boxes:
                word_boxes[word['text']] = [
                    1.0 * word['box'][0] / data_dict[fname]['data']['size'][1],
                    1.0 * word['box'][1] / data_dict[fname]['data']['size'][0],
                    1.0 * word['box'][2] / data_dict[fname]['data']['size'][1],
                    1.0 * word['box'][3] / data_dict[fname]['data']['size'][0]]
        feature = []
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                boxi = word_boxes[keywords[i]]
                boxj = word_boxes[keywords[j]]
                feature.extend([boxi[t] - boxj[t] for t in range(4)])
        feature_list.append(feature)
        n += 1
    feature_list = numpy.array(feature_list)
    print(feature_list.shape, len(index2fname))

    # 聚类
    clustering = KMeans(n_clusters=20, max_iter=1000).fit(feature_list)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(index2fname[i])
    for label in clusters:
        print(label, len(clusters[label]))

    if False:
        # 根据聚类结果拷贝图片
        for label in clusters:
            print('output %d: %d' % (label, len(clusters[label])))
            target_dir = os.path.join(cluster_dir, str(label))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            for fname in clusters[label]:
                target_path = os.path.join(target_dir, fname).replace('json', 'jpg')
                source_path = data_dict[fname]['image_path']
                print(target_path, source_path)
                shutil.copy(source_path, target_path)
        # 保存feature
        feature_path = os.path.join(cluster_dir, 'features.json')
        feature_dict = {}
        for i, label in enumerate(clustering.labels_):
            fname = index2fname[i]
            feature = [round(t, 8) for t in feature_list[i,:].tolist()]
            feature_dict[fname] = {'label': int(label), 'feature': feature}
        json.dump(feature_dict, open(feature_path, 'w'), indent=4)

    if False:
        # 根据聚类结果分配训练/测试集
        target_train_dir = '/home/caory/data/datasets/extract_information/20201120/train_data_cluster_split/'
        n = 0
        for label in clusters:
            random.shuffle(clusters[label])
            n_train = 1
            train_fnames = clusters[label][0:n_train]
            for fnames, target_dir in [[train_fnames, target_train_dir]]:
                for fname in fnames:
                    data_path = data_dict[fname]['data_path']
                    image_path = data_dict[fname]['image_path']
                    docid = data_dict[fname]['docid']
                    if os.path.exists(data_path) and os.path.exists(image_path):
                        n += 1
                        print(n, fname)
                        data = json.load(open(data_path))
                        if data.get('is_wrong'):
                            continue
                        if 'image' in data:
                            del data['image']
                        data['information'] = {}
                        for widx, word in enumerate(data['words']):
                            if word['ss_label'] != 'back':
                                [cls, ktype] = word['ss_label'].split('|')
                                if cls not in data['information']:
                                    data['information'][cls] = {
                                        'key': [{'contain_words': [], 'text': ''}],
                                        'value': [{'contain_words': [], 'text': ''}]}
                                if ktype == 'k':
                                    data['information'][cls]['key'][0][
                                        'contain_words'].append(widx)
                                elif ktype == 'v':
                                    data['information'][cls]['value'][0][
                                        'contain_words'].append(widx)
                        for cls in data['information']:
                            for sub_key in ['key', 'value']:
                                info = data['information'][cls][sub_key][0]
                                info['text'] = '<#>'.join([data['words'][widx]['text'] \
                                    for widx in info['contain_words']])
                        if not os.path.exists(os.path.join(target_dir, docid)):
                            os.mkdir(os.path.join(target_dir, docid))
                        target_path = os.path.join(target_dir, docid, fname)
                        target_image_path = os.path.join(
                            target_dir, docid, fname.replace('json', 'jpg'))
                        json.dump(data, open(target_path, 'w'), indent=4)
                        shutil.copy(image_path, target_image_path)

def get_cluster_fcc_data():
    source_dir = '/home/caory/github/SroieAnnotator/fcc_data/data/'
    pic_dir = '/home/caory/github/SroieAnnotator/fcc_data/pic/'
    cluster_dir = '/home/caory/github/SroieAnnotator/fcc_data/cluster/'
    n = 0
    data_dict = {}
    for docid in os.listdir(source_dir):
        for fname in os.listdir(os.path.join(source_dir, docid)):
            if fname.split('.')[1] == 'json':
                data_path = os.path.join(source_dir, docid, fname)
                image_path = os.path.join(source_dir, docid, fname.replace('json', 'jpg'))
                if os.path.exists(data_path) and os.path.exists(image_path):
                    n += 1
                    print(n, fname)
                    data = json.load(open(data_path))
                    if data.get('is_wrong'):
                        continue
                    if 'image' in data:
                        del data['image']
                    data_dict[fname] = {'docid': docid, 'data': data,
                        'data_path': data_path, 'image_path': image_path}

    distance_path = os.path.join(cluster_dir, 'distance.json.npy')
    fnames = list(data_dict.keys())
    if os.path.exists(distance_path):
        index2fname = {}
        for i, fname in enumerate(fnames):
            index2fname[i] = fname
        dist_matrix = numpy.load(distance_path)
    else:
        # 遍历两两doc，计算他们之间的距离
        st = time.time()
        file_list = Queue(maxsize=10000)
        dist_matrix = Array('d', [0.0] * (len(fnames) * len(fnames)))
        index2fname = {}
        for i, fname in enumerate(fnames):
            index2fname[i] = fname
            file_list.put([i, fname])

        def _producer(file_list, length):
            while not file_list.empty():
                [i, fnamea] = file_list.get()
                print('calculating %d/%d' % (i, len(fnames)))
                dda = data_dict[fnamea]
                for j, fnameb in enumerate(fnames[i+1:]):
                    ddb = data_dict[fnameb]
                    overlap_words, wordsa, wordsb = {}, {}, {}
                    for word in dda['data']['words']:
                        if word['text'] not in wordsa:
                            if 3 <= len(word['text']) <= 15 and \
                                not re.findall('\d', word['text']):
                                wordsa[word['text']] = word
                    for word in ddb['data']['words']:
                        if word['text'] not in wordsb:
                            if 3 <= len(word['text']) <= 15 and \
                                not re.findall('\d', word['text']):
                                wordsb[word['text']] = word
                    for word in wordsa:
                        if word in wordsb:
                            overlap_words[word] = {
                                'worda': wordsa[word], 'wordb': wordsb[word]}
                    n_overlap = len(overlap_words)
                    if n_overlap >= 100:
                        overlap_rate = 0.0
                    else:
                        overlap_rate = 1.0 * (100 - n_overlap) / 100.0

                    all_words = list(overlap_words.keys())
                    worda_dist = []
                    for m in range(len(all_words)):
                        wordi = overlap_words[all_words[m]]['worda']
                        [li, ti, ri, bi] = [
                            1.0 * wordi['box'][0] / dda['data']['size'][1],
                            1.0 * wordi['box'][1] / dda['data']['size'][0],
                            1.0 * wordi['box'][2] / dda['data']['size'][1],
                            1.0 * wordi['box'][3] / dda['data']['size'][0]]
                        for n in range(m+1, len(all_words)):
                            wordj = overlap_words[all_words[n]]['worda']
                            [lj, tj, rj, bj] = [
                                1.0 * wordj['box'][0] / dda['data']['size'][1],
                                1.0 * wordj['box'][1] / dda['data']['size'][0],
                                1.0 * wordj['box'][2] / dda['data']['size'][1],
                                1.0 * wordj['box'][3] / dda['data']['size'][0]]
                            worda_dist.append([li - lj, ti - tj, ri - rj, bi - bj])
                    wordb_dist = []
                    for m in range(len(all_words)):
                        wordi = overlap_words[all_words[m]]['wordb']
                        [li, ti, ri, bi] = [
                            1.0 * wordi['box'][0] / ddb['data']['size'][1],
                            1.0 * wordi['box'][1] / ddb['data']['size'][0],
                            1.0 * wordi['box'][2] / ddb['data']['size'][1],
                            1.0 * wordi['box'][3] / ddb['data']['size'][0]]
                        for n in range(m+1, len(all_words)):
                            wordj = overlap_words[all_words[n]]['wordb']
                            [lj, tj, rj, bj] = [
                                1.0 * wordj['box'][0] / ddb['data']['size'][1],
                                1.0 * wordj['box'][1] / ddb['data']['size'][0],
                                1.0 * wordj['box'][2] / ddb['data']['size'][1],
                                1.0 * wordj['box'][3] / ddb['data']['size'][0]]
                            wordb_dist.append([li - lj, ti - tj, ri - rj, bi - bj])
                    average_dist = []
                    for veca, vecb in zip(worda_dist, wordb_dist):
                        average_dist.append([
                            0.5 * (veca[k] - vecb[k]) ** 2 for k in range(4)])
                    if average_dist:
                        lavg = numpy.mean([average_dist[k][0] \
                            for k in range(len(average_dist))])
                        tavg = numpy.mean([average_dist[k][1] \
                            for k in range(len(average_dist))])
                        ravg = numpy.mean([average_dist[k][2] \
                            for k in range(len(average_dist))])
                        bavg = numpy.mean([average_dist[k][3] \
                            for k in range(len(average_dist))])
                        dist_matrix[i * len(fnames) + j] = numpy.mean([
                            lavg, tavg, ravg, bavg])
                    else:
                        dist_matrix[i * len(fnames) + j] = 1.0

        process_list = []
        for i in range(40):
            process = Process(target=_producer, args=(file_list, file_list.qsize()))
            process_list.append(process)
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()

        dist_matrix = numpy.array(dist_matrix[:]).reshape((len(fnames), len(fnames)))
        numpy.save(distance_path, dist_matrix)
        et = time.time()
        print('calculating distance matrix time: %.2f' % (et - st))

    # 聚类
    clustering = AgglomerativeClustering(
        n_clusters=10, affinity='precomputed', linkage='average').fit(dist_matrix)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(index2fname[i])
    for label in clusters:
        print(label, len(clusters[label]))

    if True:
        # 根据聚类结果拷贝图片
        for label in clusters:
            print('output %d: %d' % (label, len(clusters[label])))
            target_dir = os.path.join(cluster_dir, str(label))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            for fname in clusters[label]:
                target_path = os.path.join(target_dir, fname).replace('json', 'jpg')
                source_path = data_dict[fname]['image_path']
                print(target_path, source_path)
                shutil.copy(source_path, target_path)

    if False:
        # 根据聚类结果分配训练/测试集
        target_train_dir = '/home/caory/data/datasets/extract_information/20201119/train_data/'
        target_valid_dir = '/home/caory/data/datasets/extract_information/20201119/valid_data/'
        n = 0
        for label in clusters:
            random.shuffle(clusters[label])
            n_train = 1
            train_fnames = clusters[label][0:n_train]
            valid_fnames = clusters[label][n_train:]
            for fnames, target_dir in [[train_fnames, target_train_dir], [valid_fnames, target_valid_dir]]:
                for fname in fnames:
                    data_path = data_dict[fname]['data_path']
                    image_path = data_dict[fname]['image_path']
                    docid = data_dict[fname]['docid']
                    if os.path.exists(data_path) and os.path.exists(image_path):
                        n += 1
                        print(n, fname)
                        data = json.load(open(data_path))
                        if data.get('is_wrong'):
                            continue
                        if 'image' in data:
                            del data['image']
                        data = get_new_data(data)
                        if not os.path.exists(os.path.join(target_dir, docid)):
                            os.mkdir(os.path.join(target_dir, docid))
                        target_path = os.path.join(target_dir, docid, fname)
                        target_image_path = os.path.join(
                            target_dir, docid, fname.replace('json', 'jpg'))
                        json.dump(data, open(target_path, 'w'), indent=4)
                        shutil.copy(image_path, target_image_path)


# grater_data()
# wiki_data()
# wiki_table_data()
# wiki_data_to_hyp()
# sroie_data()
# fcc_data()
# grater_statistic()
# wiki_statistic()
# get_keywords()
get_keywords_fcc_data()
# get_keywords_grater_data()
# get_keywords_sroie_data()
# get_cluster_fcc_data()