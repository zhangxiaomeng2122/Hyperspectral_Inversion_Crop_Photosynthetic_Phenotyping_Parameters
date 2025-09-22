"""
@Author: zmq
@Description:科技论文绘图
@date: 2024-07-24
图中字体Arial，字号10 a/b/c/d编号的字号选10，加粗
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.ticker import MaxNLocator, FuncFormatter

class setfig():

    def __init__(self, column,x, y):

        self.column = column  # 设置栏数
        self.x = x#画布长
        self.y = y#画布宽
        # 对尺寸和 dpi参数进行调整
        plt.rcParams['figure.dpi'] = 300

        # 字体调整
        plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS，Calibri
        plt.rcParams['font.weight'] = 'light'
        plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
        plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
        plt.rcParams['axes.labelsize'] = 8 # 坐标轴标签字体大小
        plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
        #缩短刻度和坐标轴的距离
        plt.rcParams['axes.titlepad'] = 4
        plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
        # plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.figsize'] = (x, y)#画布大小
        #边缘不留空白
        plt.rcParams['figure.autolayout'] = True
        # 线条调整
        plt.rcParams['axes.linewidth'] = 0.5

        # 刻度在内，设置刻度字体大小
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        #设置刻度粗细
        plt.rcParams['xtick.major.width'] = 0.5
        plt.rcParams['ytick.major.width'] = 0.5
        # 设置刻度长度
        plt.rcParams['xtick.major.size'] = 2
        plt.rcParams['ytick.major.size'] = 2

        # 设置输出格式为PDF
        plt.rcParams['savefig.format'] = 'PDF'
        plt.rcParams['figure.autolayout'] = True

    @property
    def tickfont(self):
        plt.tight_layout()
        ax1 = plt.gca()  # 获取当前图像的坐标轴
        # 更改坐标轴字体，避免出现指数为负的情况
        tick_font = font_manager.FontProperties(family='Arial', size=8.0)
        for labelx in ax1.get_xticklabels():
            labelx.set_fontproperties(tick_font)
        for labely in ax1.get_yticklabels():
            labely.set_fontproperties(tick_font)

    @property
    def Global_font(self):
        # 设置基本字体
        plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
        plt.rcParams['font.weight'] = 'light'

    def create(self):
        # 创建图形和坐标轴对象
        fig, ax = plt.subplots(figsize=(self.x, self.y))  # 使用 figsize 来控制画布大小
        return fig, ax  # 返回图形和坐标轴对象

    def show(self):
        # 改变字体
        # self.Global_font
        # self.tickfont
        # 设置图例显示在右下角
        ax1 = plt.gca()  # 获取当前图像的坐标轴
        ax1.spines['top'].set_visible(False)  # 隐藏上边框
        ax1.spines['right'].set_visible(False)  # 隐藏右边框
        # plt.legend(loc='lower right')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))  # 保留一位小数

        # 改变图像大小
        cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
        gcf = plt.gcf()  # 获取当前图像
        if self.column == 1:
            gcf.set_size_inches(7 * cm_to_inc, 6 * cm_to_inc)
        if self.column == 2:
            gcf.set_size_inches(14 * cm_to_inc, 6 * cm_to_inc)
        else:
            gcf.set_size_inches(21 * cm_to_inc, 6 * cm_to_inc)#设置图像大小

        # 调整边距，确保纵坐标标签不会被截断
        plt.subplots_adjust(right=0.85)  # 右边距设置为0.85，增加空间
        plt.show()
        plt.close()
#     plt.close()
# plt.close()
