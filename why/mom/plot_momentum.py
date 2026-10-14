"""画 11-1 动量策略的结果，输出 why/docs/mom_momentum.png：累计收益、回撤、每月收益、持仓的行业分布。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from ..charts import AXIS, INK, INK2, MUTED, NEGATIVE, SERIES, SURFACE, end_labels, plt, style, tidy

WHY = Path(__file__).resolve().parent.parent


def plot(daily, holdings, summary, top, cost, output):
    net, first, last = daily['net'], daily.index[0], daily.index[-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.24))
    fig.suptitle(f'AQR 式 11-1 动量：标普 500 里每月买动量最强的 {top} 只，等权持有',
                 x=0.05, ha='left', y=0.985, fontsize=15, color=INK, weight='bold')
    fig.text(0.05, 0.945, f'{first:%Y-%m-%d} 至 {last:%Y-%m-%d} · 动量 = 过去 12 个月跳过最近 1 个月的涨幅 · '
                          f'每月最后一个交易日出信号，下一交易日收盘成交 · 扣成本单边 {cost:g}bp · '
                          f'共 {summary["rebalances"]} 次调仓',
             fontsize=10.5, color=INK2, va='top')

    growth, drawdown = axes[0]
    ends = []
    for color, (column, name) in zip(SERIES, [('net', f'动量前 {top} 只（扣成本）'), ('spy', 'SPY 买入持有'),
                                              ('equal_weight', '成分股等权')]):
        curve = daily[column].fillna(0).cumsum() * 100
        growth.plot(curve.index, curve.to_numpy(), color=color, linewidth=2, label=name)
        ends.append((curve.iloc[-1], color, curve.index[-1], f'{curve.iloc[-1]:+.0f}%'))
    low, high = growth.get_ylim()
    end_labels(growth, ends, (high - low) * 0.06)
    growth.axhline(0, color=AXIS, linewidth=1)
    growth.set_title('累计收益（%，逐日相加）', loc='left', fontsize=11.5, color=INK, pad=10)
    growth.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)

    curve = net.cumsum() * 100
    dip = curve - curve.cummax()
    drawdown.plot(dip.index, dip.to_numpy(), color=SERIES[0], linewidth=1.8)
    drawdown.annotate(f'最大回撤 {dip.min():.1f}%', (dip.idxmin(), dip.min()),
                      (first + (last - first) * 0.45, dip.min() * 0.85), fontsize=9.5, color=INK, va='center',
                      arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    drawdown.set_title('回撤（%）', loc='left', fontsize=11.5, color=INK, pad=10)

    monthly_axis, sector_axis = axes[1]
    monthly = net.groupby(net.index.to_period('M')).sum() * 100
    positions = np.arange(len(monthly))
    monthly_axis.bar(positions, monthly.to_numpy(), color=[SERIES[0] if v >= 0 else NEGATIVE for v in monthly],
                     width=0.7, edgecolor=SURFACE, linewidth=1.5)
    for index in [int(np.argmax(monthly.to_numpy())), int(np.argmin(monthly.to_numpy()))]:
        value = monthly.iloc[index]
        monthly_axis.text(index, value + (1.2 if value >= 0 else -1.2), f'{value:+.1f}%', ha='center',
                          va='bottom' if value >= 0 else 'top', fontsize=9.5, color=INK)
    monthly_axis.set_ylim(monthly.min() * 1.35 - 2, monthly.max() * 1.25)
    monthly_axis.set_xticks(positions[::2])
    monthly_axis.set_xticklabels([period.strftime('%Y-%m') for period in monthly.index[::2]])
    monthly_axis.axhline(0, color=AXIS, linewidth=1)
    monthly_axis.set_title(f'每月收益（%）：{int((monthly > 0).sum())}/{len(monthly)} 个月为正，最后一个月只到月中',
                           loc='left', fontsize=11.5, color=INK, pad=10)

    share = holdings['sector'].value_counts(normalize=True).sort_values() * 100
    spots = np.arange(len(share))
    sector_axis.barh(spots, share.to_numpy(), color=SERIES[0], height=0.65, edgecolor=SURFACE, linewidth=1.2)
    for spot, value in zip(spots, share):
        sector_axis.text(value + 1, spot, f'{value:.0f}%', va='center', fontsize=9.5, color=INK)
    sector_axis.set_yticks(spots)
    sector_axis.set_yticklabels(share.index)
    sector_axis.set_xlim(0, share.max() * 1.18)
    sector_axis.set_title(f'持仓的行业分布（{summary["rebalances"]} 次调仓合计，共 {summary["distinct_names"]} 只股票）',
                          loc='left', fontsize=11.5, color=INK, pad=10)
    sector_axis.grid(axis='x', color=AXIS, linewidth=0.8, alpha=0.5)
    sector_axis.set_axisbelow(True)
    for side in ['top', 'right', 'bottom']:
        sector_axis.spines[side].set_visible(False)
    sector_axis.tick_params(length=0)

    for axis in [growth, drawdown]:
        axis.set_xlim(first, last + pd.Timedelta(days=60))
        axis.set_xticks(pd.date_range(first, last, freq='2MS'))
        axis.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    for axis in [growth, drawdown, monthly_axis]:
        tidy(axis)
    fig.text(0.05, 0.012, f'检验期只有 {summary["rebalances"]} 次调仓，只持有 {top} 只股票，'
                          '结果受个别股票和单一行业影响很大；未做行业和风格中性。', fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.06)
    fig.savefig(output, dpi=150)
    plt.close(fig)


NAMES = {'none': '不对冲', 'spy_1x': '卖空等额 SPY', 'spy_beta': '按 beta 卖空 SPY', 'qqq_1x': '卖空等额 QQQ',
         'qqq_beta': '按 beta 卖空 QQQ', 'sector': '卖空同行业篮子', 'long_short': '卖空最弱的 10 只',
         'vixy_5': '加 5% VIXY', 'vixy_10': '加 10% VIXY', 'vixm_10': '加 10% VIXM（中期）'}
LINES = ['none', 'qqq_beta', 'spy_beta']  # 左图画的三条


def plot_hedges(books, summary, benchmark, output):
    """对冲比较：左边是累计收益，右边是各方式的年化收益和夏普。"""
    first, last = books.index[0], books.index[-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.3))
    fig.suptitle('给动量组合加对冲：降风险有效，但这一年没能提高风险调整后的收益', x=0.05, ha='left', y=0.985,
                 fontsize=15, color=INK, weight='bold')
    fig.text(0.05, 0.925, f'{first:%Y-%m-%d} 至 {last:%Y-%m-%d} · 多头都是动量最强的 10 只，只改对冲方式 · '
                          'beta 在每个调仓日用过去 252 天估计，只用当天及之前的数据 · 扣成本单边 10bp',
             fontsize=10.5, color=INK2, va='top')
    ends = []
    for color, name in zip(SERIES, LINES):
        curve = books[name].cumsum() * 100
        left.plot(curve.index, curve.to_numpy(), color=color, linewidth=2, label=NAMES[name])
        ends.append((curve.iloc[-1], color, curve.index[-1], f'{curve.iloc[-1]:+.0f}%'))
    curve = benchmark.reindex(books.index).fillna(0).cumsum() * 100
    left.plot(curve.index, curve.to_numpy(), color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), label='SPY 买入持有')
    ends.append((curve.iloc[-1], INK2, curve.index[-1], f'{curve.iloc[-1]:+.0f}%'))
    low, high = left.get_ylim()
    end_labels(left, ends, (high - low) * 0.06)
    left.axhline(0, color=AXIS, linewidth=1)
    left.set_xlim(first, last + pd.Timedelta(days=60))
    left.set_xticks(pd.date_range(first, last, freq='2MS'))
    left.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    left.set_title('累计收益（%，逐日相加）', loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(loc='upper left', frameon=False, fontsize=9.5, labelcolor=INK2)
    tidy(left)

    order = [name for name in NAMES if name in summary]
    annual = [summary[name]['annual'] * 100 for name in order]
    spots = np.arange(len(order))
    right.barh(spots, annual, color=SERIES[0], height=0.62, edgecolor=SURFACE, linewidth=1.2)
    for spot, name, value in zip(spots, order, annual):
        right.text(value + 1.5, spot, f'{value:+.0f}%   夏普 {summary[name]["sharpe"]:.2f}   '
                                      f'回撤 {summary[name]["max_drawdown"] * 100:.0f}%', va='center', fontsize=9,
                   color=INK)
    right.set_yticks(spots)
    right.set_yticklabels([NAMES[name] for name in order])
    right.invert_yaxis()
    right.set_xlim(0, max(annual) * 1.75)
    right.set_title('年化收益（扣成本）', loc='left', fontsize=11.5, color=INK, pad=10)
    right.grid(axis='x', color=AXIS, linewidth=0.8, alpha=0.5)
    right.set_axisbelow(True)
    for side in ['top', 'right', 'bottom']:
        right.spines[side].set_visible(False)
    right.tick_params(length=0)
    fig.text(0.05, 0.012, '按 beta 卖空 QQQ 后 beta 只剩 0.08，波动从 54% 降到 34%，最差单月从 -25% 收窄到 -10%，'
                          '代价是年化收益从 +74% 降到 +31%；买 VIX 期货 ETP 几乎没有降低风险，只是持续亏损。',
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.79, bottom=0.1)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_qqq_hedge(books, betas, market, output):
    """按 beta 卖空 QQQ 的细节：累计收益、回撤、每次调仓用的 beta、对冲前后与 QQQ 的关系。"""
    plain, hedged = books['none'], books['qqq_beta']
    first, last = books.index[0], books.index[-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.24))
    fig.suptitle('按 beta 卖空 QQQ：把市场涨跌剥掉之后，动量组合还剩多少', x=0.05, ha='left', y=0.985,
                 fontsize=15, color=INK, weight='bold')
    fig.text(0.05, 0.945, f'{first:%Y-%m-%d} 至 {last:%Y-%m-%d} · 多头是动量最强的 10 只 · '
                          '每个调仓日用过去 252 天估计持仓对 QQQ 的 beta，按这个比例卖空 QQQ，持有到下次调仓 · '
                          '扣成本单边 10bp', fontsize=10.5, color=INK2, va='top')

    growth, drawdown = axes[0]
    ends = []
    for color, (series, name) in zip(SERIES, [(plain, '不对冲'), (hedged, '按 beta 卖空 QQQ'),
                                              (market['SPY'].reindex(books.index).fillna(0), 'SPY 买入持有')]):
        curve = series.cumsum() * 100
        growth.plot(curve.index, curve.to_numpy(), color=color, linewidth=2, label=name)
        ends.append((curve.iloc[-1], color, curve.index[-1], f'{curve.iloc[-1]:+.0f}%'))
    low, high = growth.get_ylim()
    end_labels(growth, ends, (high - low) * 0.06)
    growth.axhline(0, color=AXIS, linewidth=1)
    growth.set_title('累计收益（%，逐日相加）', loc='left', fontsize=11.5, color=INK, pad=10)
    growth.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)

    for color, (series, name) in zip(SERIES, [(plain, '不对冲'), (hedged, '按 beta 卖空 QQQ')]):
        curve = series.cumsum() * 100
        dip = curve - curve.cummax()
        drawdown.plot(dip.index, dip.to_numpy(), color=color, linewidth=1.8, label=f'{name}  最深 {dip.min():.0f}%')
    drawdown.set_title('回撤（%）', loc='left', fontsize=11.5, color=INK, pad=10)
    drawdown.legend(loc='lower left', frameon=False, fontsize=10, labelcolor=INK2)

    ladder, scatter = axes[1]
    dates = pd.to_datetime(betas['trade_date'])
    for color, column, name in [(SERIES[0], 'beta_qqq', '对 QQQ 的 beta（实际用的）'), (SERIES[1], 'beta_spy', '对 SPY 的 beta（参考）')]:
        ladder.step(dates, betas[column], where='post', color=color, linewidth=2, label=name)
        ladder.plot(dates, betas[column], marker='o', markersize=4, linestyle='none', color=color)
    ladder.axhline(1, color=AXIS, linewidth=1)
    ladder.set_ylim(0, betas[['beta_qqq', 'beta_spy']].to_numpy().max() * 1.15)
    ladder.set_title('每次调仓用的对冲比例（持仓的 beta）', loc='left', fontsize=11.5, color=INK, pad=10)
    ladder.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)

    qqq = market['QQQ'].reindex(books.index).fillna(0) * 100
    for color, (series, name) in zip(SERIES, [(plain, '不对冲'), (hedged, '对冲后')]):
        slope = np.polyfit(qqq, series * 100, 1)[0]
        scatter.scatter(qqq, series * 100, s=16, color=color, edgecolor=SURFACE, linewidth=0.4, alpha=0.75,
                        label=f'{name}：斜率 {slope:+.2f}')
        grid = np.linspace(qqq.min(), qqq.max(), 10)
        scatter.plot(grid, np.polyval(np.polyfit(qqq, series * 100, 1), grid), color=color, linewidth=1.6)
    scatter.axhline(0, color=AXIS, linewidth=1)
    scatter.axvline(0, color=AXIS, linewidth=1)
    scatter.set_xlabel('QQQ 当日收益（%）')
    scatter.set_ylabel('组合当日收益（%）')
    scatter.set_title('每日收益对 QQQ：斜率越小，受大盘影响越小', loc='left', fontsize=11.5, color=INK, pad=10)
    scatter.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)

    for axis in [growth, drawdown, ladder]:
        axis.set_xlim(first, last + pd.Timedelta(days=60))
        axis.set_xticks(pd.date_range(first, last, freq='2MS'))
        axis.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    for axis in [growth, drawdown, ladder, scatter]:
        tidy(axis)
    fig.text(0.05, 0.012, '对冲比例只用调仓日之前的数据估计。持仓越来越集中在高 beta 的半导体，'
                          '所以对冲比例从 1.2 升到 2.5：每 $100 的多头要卖空约 $250 的 QQQ。'
                          '事后看对冲后对 QQQ 还剩 0.3 的斜率，因为 beta 一路在升，用历史估的比例偏低。',
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.06)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot the 11-1 momentum backtest')
    parser.add_argument('--results', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--output', type=Path, default=WHY / 'docs' / 'mom_momentum.png')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--cost-bps', type=float, default=10.0)
    args = parser.parse_args()
    daily = pd.read_csv(args.results / 'momentum_daily.csv', index_col=0, parse_dates=True)
    holdings = pd.read_csv(args.results / 'momentum_holdings.csv', parse_dates=['signal_date', 'trade_date'])
    summary = json.loads((args.results / 'momentum_summary.json').read_text(encoding='utf-8'))
    style()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(daily, holdings, summary, args.top, args.cost_bps, args.output)
    print(f'saved {args.output}')
    hedges = args.results / 'hedge_daily.csv'
    if hedges.exists():
        books = pd.read_csv(hedges, index_col=0, parse_dates=True)
        hedge_summary = json.loads((args.results / 'hedge_summary.json').read_text(encoding='utf-8'))
        output = args.output.with_name('mom_hedge.png')
        plot_hedges(books, hedge_summary, daily['spy'], output)
        print(f'saved {output}')
        market = pd.read_parquet(args.results.parent / 'benchmarks.parquet').pivot(
            index='date', columns='symbol', values='ret')
        betas = pd.read_csv(args.results / 'hedge_betas.csv', parse_dates=['signal_date', 'trade_date'])
        output = args.output.with_name('mom_qqq_hedge.png')
        plot_qqq_hedge(books, betas, market, output)
        print(f'saved {output}')


if __name__ == '__main__':
    main()
