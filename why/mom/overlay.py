"""给 68 年的动量策略加对冲，看夏普能提多少。

被对冲的是 `breadth.py` 的「前 10% 纯多头」（48 只、每月调仓、单边 10bp，
1958-03 至 2025-12，夏普 0.852、波动 19.2%、回撤 −56.3%）。它是纯多头，beta 接近 1，
所以大部分波动来自市场本身。

试三条路，以及它们的组合：

1. **按 beta 做空市场**：用过去 252 天的滚动 beta（只用历史数据），每月调仓时按它做空
   等权基准。beta 是估出来的，会滞后，所以对冲不干净。
2. **波动率目标**（Barroso 那套）：按策略自己过去 126 个交易日的已实现波动率缩放仓位，
   目标年化 12%。和 `barroso.py` 的式(5)(6) 完全一样，只是这次作用在**我们自己构建的**
   组合上，不是 French 的现成因子。
3. **固定比例做空市场**：不估 beta，直接按固定比例空，作为 1 的对照——如果固定比例和
   按 beta 一样好，说明 beta 估计没带来价值。

对冲和缩放都要花钱：做空市场按名义金额收单边成本，调整杠杆同理。默认 10bp，和正文一致。

输出 `data/mom/overlay_summary.csv` 和 `docs/mom_overlay.png`。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.backtest import hold
from ..factors.metrics import COST_BPS
from .barroso import DAYS, TARGET
from .breadth import run, wanted
from .crsp import full as crsp_full
from .crsp import load as load_crsp
from .momentum import momentum, period_ends, periods

WHY = Path(__file__).resolve().parent.parent
BETA_WINDOW = 252   # 滚动 beta 的窗口，一年


def trailing_beta(strategy, market, window=BETA_WINDOW):
    """只用历史数据的滚动 beta：cov(策略, 市场) / var(市场)，整体后移一天。

    后移一天是关键——t 日的 beta 必须只用到 t−1 日为止的数据，否则是未来函数。
    """
    covariance = strategy.rolling(window).cov(market)
    variance = market.rolling(window).var()
    return (covariance / variance).shift(1).rename('beta')


def realized_vol(strategy, days=DAYS):
    """策略自己的已实现波动率（年化），只用历史数据。和 barroso 的式(5) 同一个口径。"""
    return np.sqrt(strategy.pow(2).rolling(days).mean() * 252).shift(1).rename('sigma')


def monthly(series, marks):
    """把逐日的对冲比例/杠杆压成「每月调仓时定一次、月内不变」。"""
    held = series.where(series.index.isin(marks)).ffill()
    return held


def apply_hedge(strategy, market, ratio, cost_bps):
    """按 ratio 做空市场。ratio 每月更新一次，变动时按名义金额收成本。"""
    ratio = ratio.reindex(strategy.index).fillna(0.0)
    gross = strategy - ratio * market.reindex(strategy.index).fillna(0.0)
    turnover = ratio.diff().abs().fillna(ratio.abs())
    return gross - turnover * cost_bps / 1e4


def apply_scale(strategy, weight, cost_bps):
    """按 weight 缩放仓位。weight 每月更新一次，变动时按名义金额收成本。"""
    weight = weight.reindex(strategy.index).fillna(0.0)
    turnover = weight.diff().abs().fillna(weight.abs())
    return weight * strategy - turnover * cost_bps / 1e4


def summarise(net, name):
    net = net.dropna()
    curve = (1 + net).cumprod()
    deviation = net.std(ddof=1)
    return dict(策略=name, 年化收益=float(net.mean() * 252),
                年化波动=float(deviation * np.sqrt(252)),
                夏普=float(net.mean() / deviation * np.sqrt(252)) if deviation > 1e-12 else 0.0,
                最大回撤=float((curve / curve.cummax() - 1).min()),
                偏度=float(net.skew()), 终值=float(curve.iloc[-1]),
                t值=float(net.mean() / deviation * np.sqrt(len(net))) if deviation > 1e-12 else 0.0)


def build(start='1957-03-01', cost_bps=COST_BPS, fraction=0.1, target=TARGET, neutral=False,
          full=True, value=True):
    """跑出基础策略、市场基准，以及各种对冲后的序列。"""
    source = crsp_full if full else load_crsp
    returns, sizes, members, sectors, _ = source(start=start)
    prices = (1 + returns.fillna(0)).cumprod().where(returns.notna())
    caps = sizes if value else None
    signal = momentum(prices)
    days = period_ends(signal.index, 'M')
    cost = cost_bps / 1e4
    _, strategy, _ = run(signal, members, returns, days, cost, sectors, fraction=fraction,
                         short=False, neutral=neutral, caps=caps)
    # 基准 = 同样每月调仓、等权持有当期全部成分股，和策略同一套持有期切片
    market = pd.concat([hold(returns.iloc[span], set(members.columns[members.iloc[day].to_numpy()]))
                        for day, span in periods(signal.index, days)])
    window = strategy.index.intersection(market.index)
    strategy, market = strategy.reindex(window), market.reindex(window)
    marks = [strategy.index[i] for i in range(len(strategy)) if i == 0 or
             strategy.index[i].to_period('M') != strategy.index[i - 1].to_period('M')]
    return strategy, market, marks


def plot(books, table, ladder, path):
    """左：累计净值；右：目标波动率这一档的收益-风险权衡。

    左图只画「原始、按 beta 对冲、波动率目标」三条加基准——固定比例那几档是中间态，
    画上去只会挤。右图用散点说明缩放为什么是免费的午餐：夏普不变，所以沿着一条直线走，
    想要多少收益就承担多少波动，而同等波动下它比原始更靠上。
    """
    from matplotlib.ticker import FuncFormatter

    from ..charts import AXIS, INK, INK2, MUTED, SERIES, SURFACE, plt, style, tidy
    style()
    show = ([name for name in books if name.startswith('原始')]
            + ['波动率目标 12%', '按 beta 空市场', '标普 500 等权（基准）'])
    show = [name for name in show if name in books]
    colours = dict(zip(show, SERIES))
    colours['标普 500 等权（基准）'] = MUTED
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(wspace=0.24))
    fig.suptitle('按 beta 对冲最差，按自身波动率缩放最好', x=0.042, ha='left', y=0.97,
                 fontsize=15, color=INK, weight='bold')
    fig.text(0.042, 0.888, '对冲对象是 68 年的「前 10% 纯多头」（48 只、每月调仓、单边 10bp）。'
             '对冲比例和杠杆都每月更新一次，变动按名义金额收费。\n'
             '滚动 beta 用过去 252 天、整体后移一天；波动率用过去 126 天，都只含历史数据。',
             fontsize=9.5, color=INK2, va='top', linespacing=1.5)

    tidy(left)
    for name in show:
        value = (1 + books[name].dropna()).cumprod()
        left.plot(value.index, value, color=colours[name], linewidth=1.6,
                  linestyle=(0, (5, 2)) if '基准' in name else '-', label=name)
    left.set_yscale('log')
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
    left.axhline(1, color=AXIS, linewidth=1)
    left.set_title('1 美元的累计增长（扣成本，对数轴）', loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc='upper left')

    tidy(right)
    right.plot(ladder['年化波动'] * 100, ladder['年化收益'] * 100, color=SERIES[1], linewidth=2,
               marker='o', markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for _, row in ladder.iterrows():
        right.annotate(row['标签'], (row['年化波动'] * 100, row['年化收益'] * 100), (8, -4),
                       textcoords='offset points', fontsize=9, color=INK2)
    base = table.iloc[0]
    right.plot([base['年化波动'] * 100], [base['年化收益'] * 100], marker='o', markersize=11,
               color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    right.annotate('原始（无缩放）', (base['年化波动'] * 100, base['年化收益'] * 100), (10, -14),
                   textcoords='offset points', fontsize=9.5, color=INK)
    right.set_xlabel('年化波动（%）')
    right.set_title('缩放档位的收益-风险权衡：同等波动下比原始更高', loc='left',
                    fontsize=11.5, color=INK, pad=10)
    right.text(0.03, 0.95, '夏普沿这条线几乎不变（0.879–0.882），\n所以它是一条直线；'
               '原始那点落在线的下方。', transform=right.transAxes, fontsize=9, color=MUTED,
               va='top', linespacing=1.5)
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.055, right=0.98)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def main():
    parser = argparse.ArgumentParser(description='Hedge the long-history momentum strategy')
    parser.add_argument('--start', default='1957-03-01')
    parser.add_argument('--cost-bps', type=float, default=COST_BPS)
    parser.add_argument('--target', type=float, default=TARGET, help='波动率目标（年化）')
    parser.add_argument('--members-only', action='store_true',
                        help='动量只用成员期内的历史（默认用完整历史）')
    parser.add_argument('--equal', action='store_true', help='等权（默认市值加权）')
    parser.add_argument('--neutral', action='store_true',
                        help='被对冲的改成行业中性版本（每个行业内部各取前 10%）')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--plot', type=Path, default=WHY / 'docs' / 'mom_overlay.png')
    args = parser.parse_args()

    strategy, market, marks = build(args.start, args.cost_bps, neutral=args.neutral,
                                    full=not args.members_only, value=not args.equal)
    beta = monthly(trailing_beta(strategy, market), marks)
    sigma = monthly(realized_vol(strategy), marks)
    scale = (args.target / sigma).clip(upper=3.0)        # 杠杆上限 3 倍，避免早期低波动时离谱

    base = '原始（前 10% 纯多头' + ('，行业中性）' if args.neutral else '）')
    books = {base: strategy, '标普 500 等权（基准）': market}
    books['按 beta 空市场'] = apply_hedge(strategy, market, beta, args.cost_bps)
    for ratio in (0.3, 0.5, 0.7, 1.0):
        books[f'固定空 {ratio:g} 倍市场'] = apply_hedge(strategy, market,
                                                        pd.Series(ratio, index=strategy.index), args.cost_bps)
    books[f'波动率目标 {args.target:.0%}'] = apply_scale(strategy, scale, args.cost_bps)
    hedged = apply_hedge(strategy, market, beta, args.cost_bps)
    both = apply_scale(hedged, monthly(args.target / realized_vol(hedged), marks).clip(upper=3.0), args.cost_bps)
    books['按 beta 空市场 + 波动率目标'] = both

    table = pd.DataFrame([summarise(v, k) for k, v in books.items()]).set_index('策略')
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'overlay_summary.csv', float_format='%.6f')
    pd.DataFrame(books).to_csv(args.output / 'overlay_daily.csv', float_format='%.8f')
    pd.set_option('display.width', 220)
    print(f'{strategy.index[0].date()} 至 {strategy.index[-1].date()}，单边 {args.cost_bps:g}bp')
    print(table.round(4).to_string())
    ladder = pd.DataFrame([dict(summarise(apply_scale(strategy, (tgt / sigma).clip(upper=3.0),
                                                      args.cost_bps), f'目标 {tgt:.0%}'),
                                标签=f'{tgt:.0%}')
                           for tgt in (0.12, 0.16, 0.19, 0.22, 0.25)])
    ladder.to_csv(args.output / 'overlay_ladder.csv', index=False, float_format='%.6f')
    print()
    print('把目标波动率往上调（夏普不变，所以这是一条收益-风险的直线）：')
    print(ladder.set_index('标签')[['年化收益', '年化波动', '夏普', '最大回撤', '终值']].round(4).to_string())
    if wanted(args.plot):
        plot(books, table, ladder, args.plot)
    print(f'\n滚动 beta：中位 {beta.median():.2f}  5% 分位 {beta.quantile(.05):.2f}  95% 分位 {beta.quantile(.95):.2f}')
    print(f'波动率目标的杠杆：中位 {scale.median():.2f}  95% 分位 {scale.quantile(.95):.2f}  顶格比例 {(scale >= 3).mean():.1%}')
    print(f"\n写入 {args.output / 'overlay_summary.csv'}")


if __name__ == '__main__':
    main()
