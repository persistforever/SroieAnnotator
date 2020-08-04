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
import json
import shutil
import wx.lib.scrolledpanel as scrolled


class Application():
    def __init__(self):
        self.point_type = 'start'
        self.information = {}
        self.read_datas()
        self.app = wx.App()
        self.frame = wx.Frame(None, title='SROIE标注', pos=(0,0), size=(2500,1700))
        self.panel = wx.lib.scrolledpanel.ScrolledPanel(
            self.frame, -1, size=(1500,1600), pos=(0,0), style=wx.SIMPLE_BORDER)
        self.panel2 = wx.Panel(self.frame, -1, size=(1000,1700), pos=(1500,0))
        self.panel.SetupScrolling()
        self.panel.SetBackgroundColour('#FFFFFF')
        self.bitmap = wx.StaticBitmap(self.panel, -1, wx.Bitmap(), (0, 0))
        self.bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.bitmap.SetCursor(wx.Cursor(wx.CURSOR_CROSS))

        self.name_text = wx.StaticText(self.panel2, pos=(100,400), size=(100,50))
        self.info_text = wx.StaticText(self.panel2, pos=(100,500), size=(100,200))
        font = wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.info_text.SetFont(font)
	
        self.color_dict = {
            'back': [50,50,50],
            'company|k': [255,50,50], 'company|v': [255,150,150],
            'address|k': [50,255,50], 'address|v': [150,255,150],
            'date|k': [50,50,255], 'date|v': [150,150,255],
            'total|k': [50,255,255], 'total|v': [150,255,255],
            'cash|k': [255,50,255], 'cash|v': [255,150,255],
            'tel|k': [255,255,50], 'tel|v': [255,255,150],
            'fax|k': [25,150,75], 'fax|v': [25,150,150]}

        self.buttons = {}
        for ktype in ['k', 'v']:
            for i, key in enumerate(['company', 'address', 'date', 'total', 'cash', 'tel', 'fax']):
                name = '%s|%s' % (key, ktype)
                color = self.color_dict[name]
                x = 100 + 120 * i
                y = 100 if ktype == 'k' else 200
                button = wx.Button(self.panel2,
                    label='%s|%s' % (key[0:3], ktype), 
                    pos=(x,y), size=[100,50], name=name)
                self.frame.Bind(wx.EVT_BUTTON, self.on_click_button, button)
                self.buttons[name] = button
        button = wx.Button(self.panel2, label='NUL', 
            pos=(820,300), size=[100,50], name='back')
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_button, button)
        self.buttons['back'] = button
		
        self.save_button = wx.Button(self.panel2, label='save', pos=(100,300))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_save_button, self.save_button)
        self.next_button = wx.Button(self.panel2, label='next', pos=(500,300))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_next_button, self.next_button)

        self.bSizer = wx.BoxSizer(wx.VERTICAL)
        self.bSizer.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer)
        self.bSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.bSizer1.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer1)

        self.init_canvas()

    def read_datas(self):
        self.datas = {}
        self.main_dir = 'E://Github//SroieAnnotator//data//train//'
        for file_name in os.listdir(os.path.join(self.main_dir, 'pic')):
            name = file_name.split('.')[0]
            pic_path = os.path.join(self.main_dir, 'pic', '%s.jpg' % (name))
            label_path = os.path.join(self.main_dir, 'label', '%s.txt' % (name))
            ocr_path = os.path.join(self.main_dir, 'ocr', '%s.txt' % (name))
            pkl_path = os.path.join(self.main_dir, 'new_data', '%s.pkl' % (name))
            data_path = os.path.join(self.main_dir, 'new_data_20200731', name, '%s_0.json' % (name))
            if os.path.exists(pic_path) and \
                os.path.exists(ocr_path) and \
                os.path.exists(pkl_path) and \
                not os.path.exists(data_path):
                self.datas[name] = {'is_labeled': False, 
                    'pic_path': pic_path, 'label_path': label_path, 
                    'ocr_path': ocr_path, 'pkl_path': pkl_path,
                    'data_path': data_path}
		
        self.index = 0
        print('all data:', len(self.datas))
        # print(list(self.datas.keys()))
        # exit()

    def init_canvas(self):
        self.choosed_word = {}
        self.point_type = 'start'
        if self.index > len(self.datas) - 1:
            self.index = len(self.datas) - 1
        self.name = list(self.datas.keys())[self.index]

        ocr_path = self.datas[self.name]['ocr_path']
        pkl_path = self.datas[self.name]['pkl_path']
        data = pickle.load(open(pkl_path, 'rb'))
        for word in data['pages'][0]['words']:
            [left, top, right, bottom] = word['box']
            middle = 1.0 * (bottom + top) / 2.0
            height = (bottom - top) * 0.5
            new_top = middle - height / 2.0
            new_bottom = middle + height / 2.0
            word['adjust_box'] = [int(left), int(new_top), int(right), int(new_bottom)]
        self.words = data['pages'][0]['words']

        self.orig_png = wx.Image(
            self.datas[self.name]['pic_path'], wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.png = wx.Image(
            self.datas[self.name]['pic_path'], wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.drawBack(None)
        self.bitmap.SetBitmap(wx.Bitmap())
        self.bitmap.SetBitmap(self.png)

        self.panel.SetupScrolling()
        self.panel.SetBackgroundColour('#FFFFFF')
        self.bSizer = wx.BoxSizer(wx.VERTICAL)
        self.bSizer.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer)
        self.bSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.bSizer1.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer1)
        self.name_text.SetLabel(self.name)
        for bkey in self.buttons:
            self.buttons[bkey].SetBackgroundColour([255,255,255])

    def get_data(self, name):
        data = self.datas[name]
        new_data = {}
        new_data = {'name': name, 'pages': {0: {
            'words': None, 'image': None, 'size': None, 'information': None}}}

        ocr_path = data['ocr_path']
        char_lists = []
        with open(ocr_path, 'r') as fo:
            for line in fo:
                res = line.strip().split(',')
                if len(res) >= 9:
                    [l, t, _, _, r, b, _, _] = [int(t) for t in res[0:8]]
                    string = ','.join(res[8:])
                    char_dict = {'box': [l,t,r,b], 'text': string}
                    char_lists.append(char_dict)
            
        # new_data['pages'][0]['image'] = numpy.array(cv2.imread(data['pic_path']))
        new_data['pages'][0]['size'] = [
            new_data['pages'][0]['image'].shape[1], new_data['pages'][0]['image'].shape[0]]
        new_data['pages'][0]['words'] = char_lists

        return new_data

    def on_mouse_down(self, event):
        c1 = event.GetPosition()
        if self.point_type == 'start':
            self.start_point = [c1.x, c1.y]
        elif self.point_type == 'end':
            self.end_point = [c1.x, c1.y]
        for i, word in enumerate(self.words):
            [l, t, r, b] = word['adjust_box']
            if l < c1.x < r and t < c1.y < b:
                self.choosed_word[i] = None

        self.png = wx.Image(
            self.datas[self.name]['pic_path'], wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.drawBack(None)
        self.bitmap.SetBitmap(wx.Bitmap())
        self.bitmap.SetBitmap(self.png)

    def on_click_button(self, event):
        key_name = event.GetEventObject().GetName()
        self.buttons[key_name].SetBackgroundColour(self.color_dict[key_name])

        for i in self.choosed_word:
            self.words[i]['ss_label'] = key_name
        self.choosed_word = {}

        self.drawBack(None)

    def drawBack(self, event):
        self.png = wx.Image(
            self.datas[self.name]['pic_path'], wx.BITMAP_TYPE_ANY).ConvertToBitmap()

        dc = wx.MemoryDC(self.png)
        
        for key in self.color_dict:
            wid = 2 if key == 'back' else 4
            dc.SetPen(wx.Pen(self.color_dict[key], wid, wx.SOLID))
            dc.SetBrush(wx.Brush([255,255,255], wx.TRANSPARENT))
            for word in self.words:
                if word['ss_label'] == key:
                    [l, t, r, b] = word['adjust_box']
                    dc.DrawRectangle(l-2, t-2, r-l+2, b-t+2)

        dc.SetPen(wx.Pen([200,50,50], 4, wx.PENSTYLE_LONG_DASH))
        dc.SetBrush(wx.Brush([255,255,255], wx.TRANSPARENT))
        for i in self.choosed_word:
            [l, t, r, b] = self.words[i]['adjust_box']
            dc.DrawRectangle(l, t, r-l-2, b-t-2)
        del dc

        self.bitmap.SetBitmap(wx.Bitmap())
        self.bitmap.SetBitmap(self.png)

        self.information = {}
        for word in self.words:
            if word['ss_label'] != 'back':
                if word['ss_label'] not in self.information:
                    self.information[word['ss_label']]= {'words': []}
                self.information[word['ss_label']]['words'].append(word)

        string = ''
        if os.path.exists(self.datas[self.name]['label_path']):
            label_data = json.load(open(self.datas[self.name]['label_path'], 'r'))
        else:
            label_data = {}
        for key in self.color_dict.keys():
            if key == 'back':
                continue
            if key in self.information:
                self.information[key]['text'] = \
                    ' '.join([word['text'] for word in self.information[key]['words']])
                left = min([word['box'][0] for word in self.information[key]['words']])
                top = min([word['box'][1] for word in self.information[key]['words']])
                right = max([word['box'][2] for word in self.information[key]['words']])
                bottom = max([word['box'][3] for word in self.information[key]['words']])
                self.information[key]['box'] = [left, top, right, bottom]
                if key.split('|')[1] == 'v':
                    string += '[TRUE]' + '    ' + key + '    ' + str(label_data.get(key[:-2])) + '\n'
                string += '[PRED]' + '    ' + key + '    ' + str(self.information[key]['text']) + '\n'
            else:
                if key.split('|')[1] == 'v':
                    string += '[TRUE]' + '    ' + key + '    ' + str(label_data.get(key)) + '\n'
                string += '[PRED]' + '    ' + key + '    ' + '' + '\n'
            string += '\n'
        self.info_text.SetLabel(string)
        self.info_text.Wrap(800)

    def on_click_save_button(self, event):
        data = self.datas[self.name]
        new_data = pickle.load(open(self.datas[self.name]['pkl_path'], 'rb'))
        
        new_data['pages'][0]['size'] = [
            numpy.array(cv2.imread(data['pic_path'])).shape[1], 
            numpy.array(cv2.imread(data['pic_path'])).shape[0]]
        new_data['pages'][0]['words'] = self.words
        if not os.path.exists(os.path.join(self.main_dir, 'new_data_20200731', self.name)):
            os.mkdir(os.path.join(self.main_dir, 'new_data_20200731', self.name))
        for word in new_data['pages'][0]['words']:
            word['orig_box'] = word['box']
            word['box'] = word['adjust_box']
            del word['adjust_box']
        json.dump(new_data['pages'][0], open(self.datas[self.name]['data_path'], 'w'), indent=4)
        image_target_path = os.path.join(
        	self.main_dir, 'new_data_20200731', self.name, '%s_0.jpg' % (self.name))
        shutil.copy(data['pic_path'], image_target_path)
        print('%s saved' % (self.name))
        
        string = 'SAVED\n'
        self.info_text.SetLabel(string)
        self.info_text.Wrap(800)

    def on_click_next_button(self, event):
        self.information = {}
        self.choosed_word = {}
        self.point_type = 'start'
        self.index = self.index + 1
        self.name = list(self.datas.keys())[self.index]
        while self.datas[self.name]['is_labeled']:
            self.index = self.index + 1
            self.name = list(self.datas.keys())[self.index]
        self.init_canvas()


if __name__ == '__main__':
    application = Application()
    application.frame.Show()
    application.app.MainLoop()
