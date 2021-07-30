# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/07/26
# description: model manager of template matching
from __future__ import print_function
import cv2
import numpy
import wx
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import template.utils as utils


class Model:
    """
    模型类：控制模型训练、验证、预测和应用
    """
    def __init__(
        self,
        option,
        backup_dir,
        logs_dir
    ):
        # 读取配置
        self.option = option
        self.backup_dir = backup_dir
        self.logs_dir = logs_dir

    def init_model(self, processor):
        self.processor = processor
        self.index = 0
        self.template_dict = {}
        self.cluster_dict = {}
        self.run()

    def init_window(self):
        # 实例化app模块
        self.app = wx.App()
        self.frame = wx.Frame(None, title='template系统', pos=(20, 50), size=(1300, 700))
        # query部分-panel
        self.panel_query = wx.lib.scrolledpanel.ScrolledPanel(
            self.frame, -1, size=(400, 600), pos=(0, 50), style=wx.SIMPLE_BORDER)
        self.panel_query.SetupScrolling()
        self.panel_query.SetBackgroundColour('#DDDDDD')
        self.panel_tool = wx.Panel(self.frame, -1, size=(1300, 50), pos=(0, 0))
        self.panel_tool.SetBackgroundColour('#EEEEEE')
        self.bitmap = wx.StaticBitmap(self.panel_tool, -1, wx.Bitmap(), (0, 0))
        self.bitmap.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        # query部分-文本
        self.text_name = wx.StaticText(self.panel_tool, pos=(10, 20), size=(50, 10))
        font = wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.text_name.SetFont(font)
        self.text_name.SetLabel('index')
        # template部分-panel
        self.panel_tpl_a = wx.lib.scrolledpanel.ScrolledPanel(
            self.frame, -1, size=(400, 600), pos=(450, 50), style=wx.SIMPLE_BORDER)
        self.panel_tpl_a.SetupScrolling()
        self.panel_tpl_a.SetBackgroundColour('#DDDDDD')
        self.panel_tpl_b = wx.lib.scrolledpanel.ScrolledPanel(
            self.frame, -1, size=(400, 600), pos=(900, 50), style=wx.SIMPLE_BORDER)
        self.panel_tpl_b.SetupScrolling()
        self.panel_tpl_b.SetBackgroundColour('#DDDDDD')
        # template部分-文本
        self.text_score_a = wx.StaticText(self.panel_tool, pos=(600, 20), size=(100, 10))
        font = wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.text_score_a.SetFont(font)
        self.text_score_a.SetLabel('模板A得分:0.05')
        self.text_score_b = wx.StaticText(self.panel_tool, pos=(1050, 20), size=(100, 10))
        font = wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.text_score_b.SetFont(font)
        self.text_score_b.SetLabel('模板B得分:0.05')
        # 下一个按钮
        self.button_save = wx.Button(self.panel_tool, label='next',
            pos=(100, 10), size=(60, 30))
        # 选模板
        self.button_choose_a = wx.Button(self.panel_tool, label='选模板A',
            pos=(180, 10), size=(60, 30))
        self.button_choose_b = wx.Button(self.panel_tool, label='选模板B',
            pos=(260, 10), size=(60, 30))
        self.button_new_template = wx.Button(self.panel_tool, label='新建模板',
            pos=(340, 10), size=(60, 30))
        self.frame.Bind(wx.EVT_BUTTON, self.on_click_new_template, self.button_new_template)

    def run(self):
        """
        遍历数据集，判断每一个page对应的模板，如果没有，则将该页设置成模板
        """
        # 读取item
        self.item = self.processor.datasets[self.index]
        self.processor.get_data_from_disk(self.item)

        # 渲染
        self.render()

        if False:
            if len(self.template_dict) == 0:
                # 存入template
                self.template_dict[len(self.template_dict)] = \
                    self.convert_edge_dict_to_edge_memory(self.item)[0]
                self.cluster_dict[len(self.cluster_dict)] = [self.index]
            else:
                # 获取离散化距离和角度
                edge_memory, edge_list = self.convert_edge_dict_to_edge_memory(self.item)

                # 遍历template
                match_array = [0] * len(self.template_dict)
                for tplid in self.template_dict:
                    ememory = self.template_dict[tplid]
                    n_match = 0
                    for worda, wordb, euc_dist, angle in edge_list:
                        is_match = False
                        if worda in ememory and wordb in ememory[worda]:
                            for dist_offset in range(-2, 2):
                                for angle_offset in range(-2, 2):
                                    ed = euc_dist + dist_offset
                                    ag = angle + angle_offset
                                    if ed in ememory[worda][wordb] and \
                                        ag in ememory[worda][wordb][ed]:
                                        is_match = True
                                        break
                                if is_match:
                                    break
                        if is_match:
                            n_match += 1
                    match_array[tplid] = n_match

                # 判断是否匹配上template
                max_tplid, max_n_match = max(enumerate(match_array), key=lambda x: x[1])
                if 1.0 * max_n_match / len(edge_list) >= self.option['option']['match_ratio']:
                    # 匹配上template
                    self.cluster_dict[max_tplid].append(self.index)

                    if False:
                        # 更新匹配上的模板
                        new_edge_memory = {}
                        ememory = self.template_dict[max_tplid]
                        for worda, wordb, euc_dist, angle in edge_list:
                            is_match = False
                            if worda in ememory and wordb in ememory[worda]:
                                for dist_offset in range(-2, 2):
                                    for angle_offset in range(-2, 2):
                                        ed = euc_dist + dist_offset
                                        ag = angle + angle_offset
                                        if ed in ememory[worda][wordb] and \
                                            ag in ememory[worda][wordb][ed]:
                                            is_match = True
                                            break
                                    if is_match:
                                        break
                            if is_match:
                                if worda not in new_edge_memory:
                                    new_edge_memory[worda] = {}
                                if wordb not in new_edge_memory[worda]:
                                    new_edge_memory[worda][wordb] = {}
                                if euc_dist not in new_edge_memory[worda][wordb]:
                                    new_edge_memory[worda][wordb][euc_dist] = {}
                                if angle not in new_edge_memory[worda][wordb][euc_dist]:
                                    new_edge_memory[worda][wordb][euc_dist][angle] = None
                        self.template_dict[max_tplid] = new_edge_memory

                else:
                    # 未匹配上template
                    self.template_dict[len(self.template_dict)] = edge_memory
                    self.cluster_dict[len(self.cluster_dict)] = [self.index]

            # 打印debug信息
            if False:
                print('[index=%d] #template: %d' % (self.index, len(self.template_dict)))
                for tplid in self.template_dict:
                    print(tplid, len(self.template_dict[tplid].values()))

    def render(self):
        """
        渲染页面
        """
        # 渲染query文档
        self.bitmap.Destroy()
        self.bitmap = wx.StaticBitmap(self.panel_query, -1, wx.Bitmap(), (0, 0))
        self.bitmap.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        self.image = wx.Image(self.item['pic_path'], wx.BITMAP_TYPE_ANY)
        self.image = self.image.Scale(370, 570, wx.IMAGE_QUALITY_HIGH)
        self.bitmap.SetBitmap(self.image.ConvertToBitmap())
        self.bSizer_v = wx.BoxSizer(wx.VERTICAL)
        self.bSizer_v.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel_query.SetSizer(self.bSizer_v)
        self.bSizer_h = wx.BoxSizer(wx.HORIZONTAL)
        self.bSizer_h.Add(self.bitmap, 0, wx.ALL, 5)
        self.panel_query.SetSizer(self.bSizer_h)

        # 设置index
        self.text_name.SetLabel('index:%d' % (self.index))

    def on_click_new_template(self, event):
        """
        点击信建模板按钮
        """
        self.template_dict[len(self.template_dict)] = \
            self.convert_edge_dict_to_edge_memory(self.item)[0]
        self.cluster_dict[len(self.cluster_dict)] = [self.index]
        print(self.cluster_dict)

    def evaluate(self):
        """
        评估预测结果
        """
        pred_clustering = [-1] * len(self.processor.datasets)
        for cid in cluster_dict:
            for index in cluster_dict[cid]:
                pred_clustering[index] = cid
        true_clustering = [-1] * len(self.processor.datasets)
        tplid_dict = {}
        for index, item in enumerate(self.processor.datasets):
            if item['tplid'] not in tplid_dict:
                tplid_dict[item['tplid']] = len(tplid_dict)
            true_clustering[index] = tplid_dict[item['tplid']]
        amis = adjusted_mutual_info_score(true_clustering, pred_clustering)
        ars = adjusted_rand_score(true_clustering, pred_clustering)
        print(self.option['option']['match_ratio'],
            self.option['option']['n_neighbour'], round(amis, 2), round(ars, 2))

    def convert_edge_dict_to_edge_memory(self, item):
        """
        将item中的edge_dict转换为edge_memory
        """
        edge_memory, edge_list = {}, []
        edge_dict = item['content']['edge_dict']
        for edge_string in edge_dict:
            # 离散化距离
            euc_dist = edge_dict[edge_string]
            euc_dist_dis = int(round(200.0 * euc_dist / item['content']['page_height'], 0))

            # 离散化角度
            wida, widb = int(edge_string.split('&')[0]), int(edge_string.split('&')[1])
            worda = item['content']['words'][wida]
            wordb = item['content']['words'][widb]
            ab_angle = utils.calculate_angle(worda['box'], wordb['box'])
            ab_angle_dis = int(ab_angle // 5)
            ba_angle = utils.calculate_angle(wordb['box'], worda['box'])
            ba_angle_dis = int(ba_angle // 5)

            # 存入edge_memory
            if worda['text'] not in edge_memory:
                edge_memory[worda['text']] = {}
            if wordb['text'] not in edge_memory:
                edge_memory[wordb['text']] = {}
            if wordb['text'] not in edge_memory[worda['text']]:
                edge_memory[worda['text']][wordb['text']] = {}
            if worda['text'] not in edge_memory[wordb['text']]:
                edge_memory[wordb['text']][worda['text']] = {}
            if euc_dist_dis not in edge_memory[worda['text']][wordb['text']]:
                edge_memory[worda['text']][wordb['text']][euc_dist_dis] = {}
            if euc_dist_dis not in edge_memory[wordb['text']][worda['text']]:
                edge_memory[wordb['text']][worda['text']][euc_dist_dis] = {}
            edge_memory[worda['text']][wordb['text']][euc_dist_dis][ab_angle_dis] = None
            edge_memory[wordb['text']][worda['text']][euc_dist_dis][ba_angle_dis] = None

            # 存入edge_list
            edge_list.append([worda['text'], wordb['text'], euc_dist_dis, ab_angle_dis])
            edge_list.append([wordb['text'], worda['text'], euc_dist_dis, ba_angle_dis])

        return [edge_memory, edge_list]
