"""why/ 下所有图共用的配色和样式。

配色取自可视化规范的参考调色板，已用其校验脚本检查过：四个系列色相邻可区分（含色盲），
浅绿、黄两色对背景对比度不足 3:1，所以折线一律在线尾直接标注数值。
"""
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402,F401


SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # 蓝、橙、绿、黄，按顺序分配给各条线
LIGHT = '#86b6ef'  # 与蓝色同色系的浅蓝，表示“未达到门槛”
NEGATIVE = '#e34948'  # 发散色的红端，表示亏损、负值
NEUTRAL = '#f0efec'  # 发散色的中点
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'  # 主要文字、次要文字、刻度与注释
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'  # 网格线、坐标轴、背景


def style():
    plt.rcParams.update({'text.parse_math': False, 'font.sans-serif': ['Microsoft YaHei', 'Segoe UI', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10.5, 'axes.edgecolor': AXIS,
                         'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})


def tidy(axis):
    """只留横向网格线和底边坐标轴，去掉刻度线。"""
    axis.grid(axis='y', color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        axis.spines[side].set_visible(False)
    axis.tick_params(length=0)


def end_labels(axis, ends, gap, offset=pd.Timedelta(days=5)):
    """在线尾画点并标注文字；ends 是 (终值, 颜色, 日期, 文字) 的列表，按终值排开，相邻两条至少隔 gap。"""
    placed = []
    for value, color, day, text in sorted(ends, key=lambda end: end[0]):
        y = value if not placed or value - placed[-1] > gap else placed[-1] + gap
        placed.append(y)
        axis.plot([day], [value], marker='o', markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.text(day + offset, y, text, color=INK, fontsize=9.5, va='center')
