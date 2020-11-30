# -*- coding: utf-8 -*-
from __future__ import print_function
import wx
import os
import time
import numpy
import cv2
import pickle
import json
import shutil
import wx.lib.scrolledpanel as scrolled


class Application():
    def __init__(self):
        self.start_point = [-1, -1]
        self.end_point = [-1, -1]
        self.information = {}
        self.read_datas()
        self.app = wx.App()
        self.frame = wx.Frame(None, title='FCC标注', pos=(0, 0), size=(2500, 1300))
        # self.frame.SetDoubleBuffered(True)
        self.panel = wx.lib.scrolledpanel.ScrolledPanel(
            self.frame, -1, size=(1500, 1200), pos=(0, 0), style=wx.SIMPLE_BORDER)
        # self.panel.SetDoubleBuffered(True)
        self.panel2 = wx.Panel(self.frame, -1, size=(1000, 1700), pos=(1500, 0))
        self.panel.SetupScrolling()
        self.panel.SetBackgroundColour('#FFFFFF')
        self.bitmap = wx.StaticBitmap(self.panel, -1, wx.Bitmap(), (0, 0))
        # self.bitmap.SetDoubleBuffered(True)
        self.bitmap.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        self.name_text = wx.StaticText(self.panel2, pos=(100, 400), size=(100, 50))
        self.info_text = wx.StaticText(self.panel2, pos=(100, 500), size=(100, 200))
        font = wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.info_text.SetFont(font)

        self.color_dict = {
            'back': [50,50,50],
            'num|k': [50,50,255], 'num|v': [150,150,255],
            'adv|k': [50,255,255], 'adv|v': [150,255,255],
            'ffrom|k': [255,50,255], 'ffrom|v': [255,150,255],
            'fto|k': [255,255,50], 'fto|v': [255,255,150],
            'total|k': [25,150,75], 'total|v': [25,150,150]
        }
        self.buttons = {}
        for ktype in ['k', 'v']:
            for i, key in enumerate(['num', 'adv', 'ffrom', 'fto', 'total']):
                name = '%s|%s' % (key, ktype)
                color = self.color_dict[name]
                x = 100 + 120 * i
                y = 100 if ktype == 'k' else 200
                button = wx.Button(
                    self.panel2, label='%s|%s' % (key[0:3], ktype),
                    pos=(x,y), size=[100,50], name=name, style=wx.NO_BORDER)
                self.frame.Bind(wx.EVT_BUTTON, self.on_click_button, button)
                self.buttons[name] = button
        button = wx.Button(self.panel2, label='NUL', 
            pos=(820,300), size=[100,50], name='back')
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_clean, button)
        self.buttons['back'] = button

        self.save_button = wx.Button(self.panel2, label='save', pos=(100,300))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_save_button, self.save_button)
        self.next_button = wx.Button(self.panel2, label='next', pos=(300,300))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_next_button, self.next_button)
        self.wrong_button = wx.Button(self.panel2, label='wrong', pos=(500,300))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_wrong_button, self.wrong_button)

        self.bSizer = wx.BoxSizer(wx.VERTICAL)
        self.bSizer.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer)
        self.bSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.bSizer1.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel.SetSizer(self.bSizer1)

        self.init_canvas()

    def read_datas(self):
        self.datas = {}
        self.main_dir = 'E://Github//SroieAnnotator//fcc_data//'
        for file_name in os.listdir(os.path.join(self.main_dir, 'pic')):
            name = file_name.split('.')[0]
            pic_path = os.path.join(self.main_dir, 'pic', '%s.jpg' % (name))
            label_path = os.path.join(self.main_dir, 'label', '%s.json' % (name))
            token_path = os.path.join(self.main_dir, 'token', '%s.pkl' % (name))
            docid = '-'.join(name.split('.')[0].split('-')[:-1])
            pageid = int(name.split('.')[0].split('-')[-1])
            data_path = os.path.join(self.main_dir, 'data', docid, '%s_%d.json' % (docid, pageid))
            if os.path.exists(pic_path) and \
                os.path.exists(token_path) and \
                not os.path.exists(data_path):
                self.datas[name] = {'is_labeled': False, 
                    'pic_path': pic_path, 'label_path': label_path, 
                    'token_path': token_path, 'data_path': data_path}

        self.index = 0
        print('all data:', len(self.datas))

    def init_canvas(self):
        self.choosed_word = {}
        if self.index > len(self.datas) - 1:
            self.index = len(self.datas) - 1
        self.name = list(self.datas.keys())[self.index]

        token_path = self.datas[self.name]['token_path']
        data = pickle.load(open(token_path, 'rb'))
        for word in data['pages'][0]['words']:
            [left, top, right, bottom] = word['box']
            middle = 1.0 * (bottom + top) / 2.0
            height = (bottom - top) * 0.8
            new_top = middle - height / 2.0
            new_bottom = middle + height / 2.0
            word['adjust_box'] = [int(left), int(new_top), int(right), int(new_bottom)]
            word['ss_label'] = {'back': None}
        self.words = data['pages'][0]['words']

        self.bitmap.Destroy()
        self.bitmap = wx.StaticBitmap(self.panel, -1, wx.Bitmap(), (0, 0))
        self.bitmap.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        self.image = wx.Image(
            self.datas[self.name]['pic_path'], wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        dc = wx.MemoryDC(self.image)
        dc.SetPen(wx.Pen([0, 0, 0], 2, wx.SOLID))
        dc.SetBrush(wx.Brush([255, 255, 255], wx.TRANSPARENT))
        for word in self.words:
            [l, t, r, b] = word['adjust_box']
            dc.DrawRectangle(l-2, t-2, r-l+4, b-t+4)
        dc.SelectObject(wx.NullBitmap)
        self.bitmap.SetBitmap(self.image)
        self.drawBack()

        # add button to each word
        self.word_buttons = {}
        for k, word in enumerate(self.words):
            [l, t, r, b] = word['adjust_box']
            bt = wx.Button(
                self.bitmap, label=word['text'], name=str(k),
                pos=(l, t), size=(r-l, b-t), style=wx.NO_BORDER)
            font = wx.Font(int(5.0*(b-t)/15.0), wx.FONTFAMILY_ROMAN, wx.NORMAL, wx.NORMAL)
            bt.SetFont(font)
            bt.SetBackgroundColour([240, 240, 240])
            self.frame.Bind(wx.EVT_BUTTON, self.on_click_word_button, bt)
            self.word_buttons[str(k)] = bt

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
            self.buttons[bkey].SetBackgroundColour([255, 255, 255])

    def on_click_word_button(self, event):
        key_name = event.GetEventObject().GetName()
        self.word_buttons[key_name].SetBackgroundColour([200, 100, 100])
        self.choosed_word[int(key_name)] = None
        self.drawBack()

    def on_key_down(self, event):
        keycode = event.GetEventObject()
        print(keycode)
        event.Skip()

    def on_click_button(self, event):
        key_name = event.GetEventObject().GetName()
        self.buttons[key_name].SetBackgroundColour(self.color_dict[key_name])

        for i in self.choosed_word:
            self.word_buttons[str(i)].SetBackgroundColour(self.color_dict[key_name])
            self.words[i]['ss_label'][key_name] = None
        self.choosed_word = {}
        self.drawBack()

    def on_click_clean(self, event):
        key_name = event.GetEventObject().GetName()
        self.buttons[key_name].SetBackgroundColour(self.color_dict[key_name])

        for i in self.choosed_word:
            self.word_buttons[str(i)].SetBackgroundColour([240, 240, 240])
            self.words[i]['ss_label'] = {'back': None}
        self.choosed_word = {}

        self.drawBack()

    def drawBack(self):
        self.information = {}
        for word in self.words:
            for sl in word['ss_label']:
                if sl != 'back':
                    if sl not in self.information:
                        self.information[sl]= {'words': []}
                    self.information[sl]['words'].append(word)

        string = ''
        if os.path.exists(self.datas[self.name]['label_path']):
            label_data = json.load(open(self.datas[self.name]['label_path'], 'r'))
        else:
            label_data = {}
        cvt_key = {
            'num': 'contract_num', 'adv': 'advertiser',
            'ffrom': 'flight_from', 'fto': 'flight_to', 'total': 'gross_amount'}
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
                    string += '[TRUE]' + '    ' + key + '    ' + \
                    str(label_data.get(cvt_key[key[:-2]])) + '\n'
                string += '[PRED]' + '    ' + key + '    ' + str(self.information[key]['text']) + '\n'
            else:
                if key.split('|')[1] == 'v':
                    string += '[TRUE]' + '    ' + key + '    ' + str(label_data.get(cvt_key[key[:-2]])) + '\n'
                string += '[PRED]' + '    ' + key + '    ' + '' + '\n'
            string += '\n'
        self.info_text.SetLabel(string)
        self.info_text.Wrap(800)

    def on_click_save_button(self, event):
        data = self.datas[self.name]
        new_data = pickle.load(open(self.datas[self.name]['token_path'], 'rb'))
        docid = '-'.join(self.name.split('.')[0].split('-')[:-1])
        pageid = int(self.name.split('.')[0].split('-')[-1])
        
        new_data['pages'][0]['size'] = [
            numpy.array(cv2.imread(data['pic_path'])).shape[1], 
            numpy.array(cv2.imread(data['pic_path'])).shape[0]]
        new_data['pages'][0]['words'] = self.words
        if not os.path.exists(os.path.join(self.main_dir, 'data', docid)):
            os.mkdir(os.path.join(self.main_dir, 'data', docid))
        for word in new_data['pages'][0]['words']:
            word['orig_box'] = word['box']
            word['box'] = word['adjust_box']
            del word['adjust_box']
        json.dump(new_data['pages'][0], open(self.datas[self.name]['data_path'], 'w'), indent=4)
        image_target_path = os.path.join(
            self.main_dir, 'data', docid, '%s_%d.jpg' % (docid, pageid))
        shutil.copy(data['pic_path'], image_target_path)
        print('%s saved' % (self.name))
        
        string = 'SAVED\n'
        self.info_text.SetLabel(string)
        self.info_text.Wrap(800)

    def on_click_wrong_button(self, event):
        data = self.datas[self.name]
        new_data = pickle.load(open(self.datas[self.name]['token_path'], 'rb'))
        docid = '-'.join(self.name.split('.')[0].split('-')[:-1])
        pageid = int(self.name.split('.')[0].split('-')[-1])
        
        new_data['pages'][0]['size'] = [
            numpy.array(cv2.imread(data['pic_path'])).shape[1], 
            numpy.array(cv2.imread(data['pic_path'])).shape[0]]
        new_data['pages'][0]['words'] = self.words
        if not os.path.exists(os.path.join(self.main_dir, 'data', docid)):
            os.mkdir(os.path.join(self.main_dir, 'data', docid))
        for word in new_data['pages'][0]['words']:
            word['orig_box'] = word['box']
            if 'adjust_box' in word:
                word['box'] = word['adjust_box']
                del word['adjust_box']
        new_data['pages'][0]['is_wrong'] = True
        json.dump(new_data['pages'][0], open(self.datas[self.name]['data_path'], 'w'), indent=4)
        print('%s saved' % (self.name))
        
        string = 'SAVED\n'
        self.info_text.SetLabel(string)
        self.info_text.Wrap(800)

    def on_click_next_button(self, event):
        self.information = {}
        self.choosed_word = {}
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
