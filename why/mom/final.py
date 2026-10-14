"""动量研究的最终口径：把所有修正合在一起，出一张总对比图。

走到这一步前面纠正过四处偏差，这里统一用最严谨的口径：

| | 之前 | 现在 |
|---|---|---|
| 成分股 | 维基抓的**当前** 503 只（有幸存者偏差） | CRSP 历史成员期，含被剔除的 1,453 只 |
| 动量信号 | 只用成员期内历史，7% 成分股缺信号 | 股票**自身的完整历史**，覆盖率 98.1% |
| 加权 | 等权 | **市值加权**（论文和 French 的口径） |
| 样本 | 14 次调仓 | **811 次调仓、68 年** |

成本固定单边 10bp。注意这对 1975 年 5 月之前（NYSE 固定佣金制）是乐观的——
实测按年代调整（1975 前 100bp）会把全样本夏普从 0.86 压到 0.73 左右，见 README。

输出 `data/mom/final_summary.csv`、`final_daily.csv` 和 `docs/momentum/5_final.png`。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.backtest import hold
from ..factors.metrics import COST_BPS
from . import crsp
from .barroso import TARGET
from .breadth import run, wanted
from .momentum import momentum, period_ends, periods
from .overlay import apply_scale, monthly, realized_vol, summarise

WHY = Path(__file__).resolve().parent.parent
BEST_TARGET = 0.19   # 和原始策略波动相当的那一档，见 overlay 的阶梯表


def build(start='1957-03-01', cost_bps=COST_BPS, target=BEST_TARGET):
    """跑出最终要对比的五条序列。"""
    returns, caps, members, sectors, _ = crsp.full(start=start)
    prices = (1 + returns.fillna(0)).cumprod().where(returns.notna())
    signal = momentum(prices)
    days = period_ends(signal.index, 'M')
    cost = cost_bps / 1e4
    books = {}
    # caps=None 就是等权。市值加权前 5 大能占到 58%，等权是很自然的对照。
    for label, options, weight in [('前 10 只', dict(top=10, short=False), caps),
                                   ('前 10% 市值加权', dict(fraction=0.1, short=False), caps),
                                   ('前 10% 等权', dict(fraction=0.1, short=False), None),
                                   ('前 10% 多空', dict(fraction=0.1, short=True), caps)]:
        books[label] = run(signal, members, returns, days, cost, sectors, caps=weight, **options)[1]
    window = books['前 10% 市值加权'].index
    marks = [window[i] for i in range(len(window))
             if i == 0 or window[i].to_period('M') != window[i - 1].to_period('M')]
    for base in ('前 10% 市值加权', '前 10% 等权'):
        sigma = monthly(realized_vol(books[base]), marks)
        books[f'{base} + 波动目标 {target:.0%}'] = apply_scale(
            books[base], (target / sigma).clip(upper=3.0), cost_bps)
    books['标普 500 等权（基准）'] = pd.concat(
        [hold(returns.iloc[span], set(members.columns[members.iloc[day].to_numpy()]))
         for day, span in periods(signal.index, days)])
    return pd.DataFrame(books).reindex(window).dropna()


def plot(frame, table, path, target=BEST_TARGET):
    """左：累计净值（对数轴）；右：回撤。右图才是"能不能拿得住"的答案。"""
    from matplotlib.ticker import FuncFormatter

    from ..charts import AXIS, INK, INK2, MUTED, SERIES, plt, style, tidy
    style()
    # 配色只有 4 个，图上只画「加权方式 × 要不要缩放」这四种；前 10 只和多空留在表里。
    order = [f'前 10% 等权 + 波动目标 {target:.0%}', f'前 10% 市值加权 + 波动目标 {target:.0%}',
             '前 10% 等权', '前 10% 市值加权']
    winner = table['夏普'].idxmax()
    colours = dict(zip(order, SERIES))
    benchmark = '标普 500 等权（基准）'
    colours[benchmark] = MUTED
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(wspace=0.2))
    fig.suptitle(f'最好的是「{winner}」（夏普 {table.loc[winner, "夏普"]:.3f}）', x=0.042,
                 ha='left', y=0.97, fontsize=15, color=INK, weight='bold')
    fig.text(0.042, 0.888,
             f'CRSP 标普 500 成分股（含被剔除的，无幸存者偏差），{frame.index[0]:%Y-%m} 至 '
             f'{frame.index[-1]:%Y-%m}。11-1 动量、每月调仓、单边 10bp，等权和市值加权两种都画。\n'
             '动量用股票自身的完整历史算，不受入指时间截断。'
             '波动率目标按自身过去 126 天的已实现波动缩放。'
             '「前 10 只」和「前 10% 多空」只在表里、不在图上（配色只有四个）。',
             fontsize=9.5, color=INK2, va='top', linespacing=1.5)

    for axis in (left, right):
        tidy(axis)
    labels = []
    for name in order + [benchmark]:
        value = (1 + frame[name]).cumprod()
        style_ = dict(color=colours[name], linestyle=(0, (5, 2)) if name == benchmark else '-')
        left.plot(value.index, value, linewidth=1.7, label=name, **style_)
        right.plot(value.index, (value / value.cummax() - 1) * 100, linewidth=1.3, **style_)
        labels.append(((value / value.cummax() - 1).min() * 100, colours[name]))
    placed = []                   # 最深回撤的标注按纵坐标排开，相邻至少隔 3 个百分点
    for worst, colour in sorted(labels):
        y = worst if not placed or worst - placed[-1] > 3 else placed[-1] + 3
        placed.append(y)
        right.annotate(f'{worst:.0f}%', (frame.index[-1], y), (7, 0), textcoords='offset points',
                       fontsize=9, color=colour, va='center')
    left.set_yscale('log')
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
    left.axhline(1, color=AXIS, linewidth=1)
    left.set_title('1 美元的累计增长（扣成本，对数轴）', loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc='upper left')
    right.axhline(0, color=AXIS, linewidth=1)
    right.set_title('回撤（%）', loc='left', fontsize=11.5, color=INK, pad=10)
    fig.subplots_adjust(top=0.73, bottom=0.09, left=0.055, right=0.95)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def main():
    parser = argparse.ArgumentParser(description='Final momentum comparison on the strictest setup')
    parser.add_argument('--start', default='1957-03-01')
    parser.add_argument('--cost-bps', type=float, default=COST_BPS)
    parser.add_argument('--target', type=float, default=BEST_TARGET)
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--plot', type=Path, default=WHY / 'docs' / 'momentum' / '5_final.png')
    args = parser.parse_args()

    frame = build(args.start, args.cost_bps, args.target)
    table = pd.DataFrame([summarise(frame[name], name) for name in frame.columns]).set_index('策略')
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'final_daily.csv', float_format='%.8f')
    table.to_csv(args.output / 'final_summary.csv', float_format='%.6f')
    pd.set_option('display.width', 200)
    print(f'{frame.index[0].date()} 至 {frame.index[-1].date()}，'
          f'完整历史 + 市值加权 + 单边 {args.cost_bps:g}bp')
    print(table.round(4).to_string())
    if wanted(args.plot):
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(frame, table, args.plot, args.target)
    print(f"\n写入 {args.output / 'final_summary.csv'}")


if __name__ == '__main__':
    main()
