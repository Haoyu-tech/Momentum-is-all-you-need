"""把 Barroso & Santa-Clara 的口径套到我们自己的标普 500 池子上，和「前 10 只」直接对比。

论文的 WML 是**前 10% 减后 10%**，最近一个月实际持有 1145 只（多头 462、空头 683）。
我们的 `momentum.py` 是**前 10 只纯多头**，最近三个月 100% 信息技术。两者差 100 倍。

这个脚本在**同一个池子、同一个信号、同一段时间**上跑几种宽度，回答两个问题：

1. 前 10 只的超额到底是**动量**还是**行业贝塔**？→ 看「行业中性」那一行：
   在每个行业内部各取前 10%，行业权重和基准一致，押注就被剥掉了。
2. 分散到前 10%（约 50 只）之后还剩多少？

口径和 `momentum.py` 对齐：11-1 动量、月末出信号、次日收盘成交、单边 10bp、等权
（我们没有市值数据，所以不能像 French 那样市值加权）。

**数据只够约 14 次调仓**（动量要 252 天预热，全市场数据最早到 2024-07），统计功效很低，
结论只能当方向性参考。这一点在 README 里也写了。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.backtest import hold, trade_size
from ..factors.metrics import COST_BPS
from . import crsp, universe
from .momentum import COVERAGE, SKIP, WINDOW, load, momentum, period_ends, periods

WHY = Path(__file__).resolve().parent.parent
DECILE = 0.1  # 论文口径：前 10% 和后 10%


def pick(scores, sectors, fraction=DECILE, neutral=False):
    """选出做多和做空的两头。

    neutral=True 时在**每个行业内部**各取前后 fraction，于是组合的行业权重和池子一致，
    行业押注被剥掉，剩下的才是纯粹的截面动量。
    """
    if not neutral:
        count = max(1, int(len(scores) * fraction))
        return list(scores.nlargest(count).index), list(scores.nsmallest(count).index)
    longs, shorts = [], []
    for _, group in scores.groupby(scores.index.map(lambda i: sectors.get(i, '未知'))):
        count = max(1, int(len(group) * fraction))
        if len(group) < 1 / fraction:      # 行业太小，取不出一档就跳过
            continue
        longs += list(group.nlargest(count).index)
        shorts += list(group.nsmallest(count).index)
    return longs, shorts


def weigh(names, caps, day):
    """按市值加权，权重和为 1。caps 为 None 时等权。"""
    if caps is None:
        return pd.Series(1.0 / len(names), index=list(names))
    size = caps.iloc[day].reindex(list(names)).fillna(0.0)
    return size / size.sum() if size.sum() > 0 else pd.Series(1.0 / len(names), index=list(names))


def carry(window, weights):
    """按给定权重买入后一直持有，返回组合每天的收益。等权时和 hold() 等价。"""
    growth = (1 + window[list(weights.index)].fillna(0)).cumprod().mul(weights, axis=1).sum(axis=1)
    return growth / growth.shift(1, fill_value=1.0) - 1


def shift_size(old, new):
    """按权重算的换手：两次持仓权重之差的绝对值之和。首次建仓等于 1。"""
    keys = set(old.index) | set(new.index)
    return float((new.reindex(keys).fillna(0) - old.reindex(keys).fillna(0)).abs().sum())


def run(signal, members, returns, days, cost, sectors, fraction=DECILE, neutral=False, short=True,
        top=None, caps=None):
    """跑一种宽度。top 给定时忽略 fraction，直接取前 top 只（复刻 momentum.py）。

    caps 给定时按市值加权（论文和 French 的口径），否则等权。等权走原来的代码路径，
    数字和之前完全一致；市值加权走权重路径，换手按权重差额算。
    """
    gross, net, records = [], [], []
    held_long, held_short = set(), set()
    weights_long, weights_short = pd.Series(dtype=float), pd.Series(dtype=float)
    for day, span in periods(signal.index, days):
        scores = signal.iloc[day].where(members.iloc[day]).dropna()
        if top is not None:
            longs, shorts = list(scores.nlargest(top).index), []
        else:
            longs, shorts = pick(scores, sectors, fraction, neutral)
        if not longs:
            continue
        window = returns.iloc[span]
        if caps is None:
            traded = trade_size(held_long, set(longs)) + (trade_size(held_short, set(shorts)) if short else 0.0)
            held_long, held_short = set(longs), set(shorts) if short else set()
            spread = hold(window, held_long) - (hold(window, held_short) if short and shorts else 0.0)
        else:
            now_long = weigh(longs, caps, day)
            now_short = weigh(shorts, caps, day) if short and shorts else pd.Series(dtype=float)
            traded = shift_size(weights_long, now_long) + shift_size(weights_short, now_short)
            weights_long, weights_short = now_long, now_short
            spread = carry(window, now_long) - (carry(window, now_short) if len(now_short) else 0.0)
        gross.append(spread)
        charged = spread.copy()
        charged.iloc[0] -= traded * cost
        net.append(charged)
        weights = pd.Series([sectors.get(i, '未知') for i in longs]).value_counts(normalize=True)
        records.append(dict(signal_date=signal.index[day], 行业权重=weights, 持仓=list(longs),
                            多头只数=len(longs),
                            空头只数=len(shorts) if short else 0, traded=traded,
                            多头行业数=len({sectors.get(i, '未知') for i in longs}),
                            多头最大行业占比=float(weights.iloc[0])))
    return pd.concat(gross), pd.concat(net), pd.DataFrame(records)


def summarise(net, records, benchmark=None):
    """年化收益、波动、夏普、最大回撤，外加持股数和行业集中度。"""
    curve = (1 + net).cumprod()
    result = dict(调仓次数=len(records), 交易日数=len(net),
                  年化收益=float(net.mean() * 252), 年化波动=float(net.std(ddof=1) * np.sqrt(252)),
                  夏普=float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if net.std(ddof=1) > 1e-12 else 0.0,
                  最大回撤=float((curve / curve.cummax() - 1).min()),
                  多头只数=float(records['多头只数'].median()) if len(records) else np.nan,
                  空头只数=float(records['空头只数'].median()) if len(records) else np.nan,
                  多头行业数=float(records['多头行业数'].median()) if len(records) else np.nan,
                  最大行业占比=float(records['多头最大行业占比'].mean()) if len(records) else np.nan,
                  每次换手=float(records['traded'].mean()) if len(records) else np.nan)
    result['t值'] = (float(net.mean() / net.std(ddof=1) * np.sqrt(len(net)))
                    if net.std(ddof=1) > 1e-12 else 0.0)
    if benchmark is not None:          # 只对纯多头有意义；多空是市场中性的，减多头基准没道理
        extra = (net - benchmark.reindex(net.index).fillna(0)).dropna()
        result['超额年化'] = float(extra.mean() * 252)
        result['超额t值'] = (float(extra.mean() / extra.std(ddof=1) * np.sqrt(len(extra)))
                           if extra.std(ddof=1) > 1e-12 else 0.0)
    return result


def plot(series, summary, path, title, note):
    """左：累计净值；右：多头腿的最大行业占比。两张图合起来才是这节的结论。"""
    from matplotlib.ticker import FuncFormatter

    from ..charts import AXIS, INK, INK2, MUTED, SERIES, SURFACE, plt, style, tidy
    style()
    frame = pd.DataFrame(series).dropna()
    # 配色只有 4 个，循环使用会让两条线撞色。只画 4 条策略线，基准用灰色（它不参与身份竞争），
    # 前 30% 多空留在表里不上图。
    benchmark = '标普 500 等权（对照）'
    order = [name for name in frame.columns if name != benchmark and '前 30%' not in name]
    colours = dict(zip(order, SERIES))
    colours[benchmark] = MUTED
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(wspace=0.22))
    fig.suptitle(title, x=0.042, ha='left', y=0.97, fontsize=15, color=INK, weight='bold')
    fig.text(0.042, 0.888, note, fontsize=9.5, color=INK2, va='top', linespacing=1.5)

    tidy(left)
    for label in order + [benchmark]:
        growth = (1 + frame[label]).cumprod()
        left.plot(growth.index, growth, color=colours[label], linewidth=1.8,
                  linestyle=(0, (5, 2)) if label == benchmark else '-', label=label)
    left.axhline(1, color=AXIS, linewidth=1)
    long_history = len(frame) > 3000
    if long_history:                       # 68 年用对数轴，否则早期几十年全挤在底部
        left.set_yscale('log')
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 10 else f'{v:.1f}'))
    left.set_title('累计净值（扣成本' + ('，对数轴）' if long_history else '）'),
                   loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(frameon=False, fontsize=9, labelcolor=INK, loc='upper left')

    tidy(right)
    # 池子本身也画成一根柱子（灰色），用同一套方法量出来，不用硬编码的参考线
    share = summary['最大行业占比'].reindex(order + [benchmark]).dropna()
    positions = np.arange(len(share))
    right.bar(positions, share * 100, color=[colours[i] for i in share.index],
              width=0.62, edgecolor=SURFACE, linewidth=1)
    if benchmark in share.index:
        right.axhline(share[benchmark] * 100, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    for position, value in zip(positions, share * 100):
        right.text(position, value + 1.5, f'{value:.0f}%', ha='center', fontsize=10, color=INK)
    right.set_xticks(positions)
    right.set_xticklabels([name.replace('（', '\n（').replace(' 多空', '\n多空') for name in share.index],
                          fontsize=8.5)
    right.set_ylim(0, share.max() * 100 + 10)
    right.set_title('多头腿的最大行业占比（越低越分散）', loc='left', fontsize=11.5, color=INK, pad=10)
    fig.subplots_adjust(top=0.74, bottom=0.17, left=0.055, right=0.98)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def focus(net, records, rivals, path, label='前 10% 纯多头'):
    """「前 10% 纯多头」这一种的详细图：累计、回撤、每月收益、行业权重随时间。

    行业那一栏用堆叠面积，因为这节真正的问题是**行业权重怎么漂移的**，静态柱状图看不出来。
    """
    from matplotlib.ticker import FuncFormatter

    from ..charts import AXIS, GRID, INK, INK2, MUTED, NEGATIVE, SERIES, SURFACE, plt, style, tidy
    style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.22))
    growth, fall, months, mix = axes.ravel()
    # 文案和坐标轴全部按样本长度自适应：68 年和 14 个月不能用同一套
    long_history = len(net) > 3000
    years = (net.index[-1] - net.index[0]).days / 365.25
    sharpe = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 1e-12 else 0.0
    fig.suptitle(f'{label}：{int(records["多头只数"].median())} 只、每月调仓、单边 10bp',
                 x=0.05, ha='left', y=0.975, fontsize=15, color=INK, weight='bold')
    fig.text(0.05, 0.935, f'{net.index[0]:%Y-%m-%d} 至 {net.index[-1]:%Y-%m-%d}，{years:.0f} 年、'
             f'{len(records)} 次调仓（{len(net):,} 个交易日），夏普 {sharpe:.3f}。'
             + ('' if long_history else '样本太短，统计上和基准分不开。'),
             fontsize=9.5, color=INK2, va='top')

    curves = {label: net, **rivals}
    colours = dict(zip(curves, SERIES))
    tidy(growth)
    for name, series in curves.items():
        value = (1 + series.reindex(net.index).fillna(0)).cumprod()
        if long_history:                       # 68 年用对数轴画净值，线性轴会把前 30 年压成一条线
            growth.plot(value.index, value, color=colours[name], linewidth=1.5,
                        linestyle=(0, (5, 2)) if '对照' in name else '-', label=name)
        else:
            growth.plot(value.index, (value - 1) * 100, color=colours[name], linewidth=1.8,
                        linestyle=(0, (5, 2)) if '对照' in name else '-', label=name)
    if long_history:
        growth.set_yscale('log')
        growth.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
        growth.axhline(1, color=AXIS, linewidth=1)
        growth.set_title('1 美元的累计增长（扣成本，对数轴）', loc='left', fontsize=11.5, color=INK, pad=10)
    else:
        growth.axhline(0, color=AXIS, linewidth=1)
        growth.set_title('累计收益（%，扣成本）', loc='left', fontsize=11.5, color=INK, pad=10)
    growth.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc='upper left')

    tidy(fall)
    for name, series in curves.items():
        value = (1 + series.reindex(net.index).fillna(0)).cumprod()
        fall.plot(value.index, (value / value.cummax() - 1) * 100, color=colours[name],
                  linewidth=1.5, linestyle=(0, (5, 2)) if '对照' in name else '-')
    fall.axhline(0, color=AXIS, linewidth=1)
    fall.set_title('回撤（%）', loc='left', fontsize=11.5, color=INK, pad=10)

    tidy(months)
    # 814 个月的柱子挤成一团、横轴标签糊成色块；超过 120 个月就改按年汇总
    freq, unit = ('Y', '年') if long_history else ('M', '月')
    bars = ((1 + net).groupby(net.index.to_period(freq)).prod() - 1) * 100
    positions = np.arange(len(bars))
    months.bar(positions, bars, color=[SERIES[0] if v >= 0 else NEGATIVE for v in bars],
               width=0.78 if long_history else 0.68,
               edgecolor=SURFACE, linewidth=0.4 if long_history else 1)
    months.axhline(0, color=AXIS, linewidth=1)
    step = max(1, len(bars) // 14)
    months.set_xticks(positions[::step])
    months.set_xticklabels([str(m) for m in bars.index[::step]], fontsize=8.5,
                           rotation=0 if long_history else 60, ha='center' if long_history else 'right')
    months.set_title(f'每{unit}收益（%）：{int((bars > 0).sum())}/{len(bars)} 个{unit}为正',
                     loc='left', fontsize=11.5, color=INK, pad=10)

    tidy(mix)
    weights = pd.DataFrame([r['行业权重'] for r in records.to_dict('records')],
                           index=records['signal_date']).fillna(0.0)
    weights = weights[weights.mean().sort_values(ascending=False).index]
    keep = list(weights.columns[:4])
    narrowed = weights[keep].copy()
    # 残差桶不能叫「其他」——Fama-French 12 里本来就有一个「其他」行业（而且常在前几名），
    # 重名会把真正的那一个覆盖掉。
    rest = '其余行业'
    narrowed[rest] = weights.drop(columns=keep).sum(axis=1)
    palette = SERIES[:4] + [MUTED]
    mix.stackplot(narrowed.index, *(narrowed[c] * 100 for c in narrowed.columns),
                  colors=palette, labels=list(narrowed.columns), edgecolor=SURFACE, linewidth=0.6)
    mix.set_ylim(0, 100)
    mix.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
    mix.set_title('持仓的行业权重（每次调仓）', loc='left', fontsize=11.5, color=INK, pad=10)
    mix.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower left', ncol=2)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.07)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def wanted(path):
    """命令行传空字符串表示不要这张图。Path('') 会变成 Path('.')，不能直接当真值用。"""
    return path is not None and str(path) not in ('.', '')


def main():
    parser = argparse.ArgumentParser(description='Compare momentum portfolio breadth on our own universe')
    parser.add_argument('--data', type=Path, default=WHY / 'data' / 'us_daily.parquet')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--cost-bps', type=float, default=COST_BPS)
    parser.add_argument('--schedule', default='M', choices=['M', 'W'])
    parser.add_argument('--crsp', action='store_true',
                        help='改用 WRDS 的 CRSP 标普 500 成分股（1957 年起、含被剔除的股票）')
    parser.add_argument('--start', default=None, help='只跑这个日期之后')
    parser.add_argument('--full', action='store_true',
                        help='动量用股票自身的完整历史算（修「入指不满 252 天没信号」的偏差）')
    parser.add_argument('--value', action='store_true', help='按市值加权（论文和 French 的口径），默认等权')
    parser.add_argument('--plot', type=Path, default=WHY / 'docs' / 'mom_breadth.png')
    parser.add_argument('--focus', type=Path, default=WHY / 'docs' / 'mom_breadth_decile.png',
                        help='「前 10% 纯多头」的详细图')
    args = parser.parse_args()

    caps = None
    if args.crsp:
        source = crsp.full if args.full else crsp.load
        returns, sizes, members, sectors, _ = source(start=args.start)
        prices = (1 + returns.fillna(0)).cumprod().where(returns.notna())
        caps = sizes if args.value else None
    else:
        daily, prices, returns = load(args.data)
        table = universe.load(args.output / 'sp500_constituents.csv', refresh=False)
        members = universe.membership(daily, table, prices)
        sector_by_symbol = dict(zip(table['symbol'], table['sector']))
        symbols = daily.drop_duplicates('instrument_id').set_index('instrument_id')['symbol']
        sectors = {i: sector_by_symbol.get(s, '未知') for i, s in symbols.items()}
        if args.start:
            keep = prices.index >= args.start
            prices, returns, members = prices[keep], returns[keep], members[keep]
    signal = momentum(prices)
    days = period_ends(signal.index, args.schedule)
    cost = args.cost_bps / 1e4

    setups = [('前 10 只 纯多头（momentum.py）', dict(top=10, short=False)),
              ('前 10% 纯多头', dict(fraction=DECILE, short=False)),
              ('前 10% 多空（论文口径）', dict(fraction=DECILE, short=True)),
              ('前 10% 多空 行业中性', dict(fraction=DECILE, short=True, neutral=True)),
              ('前 30% 多空', dict(fraction=0.3, short=True))]
    equal = pd.concat([hold(returns.iloc[span], set(members.columns[members.iloc[day].to_numpy()]))
                       for day, span in periods(signal.index, days)])
    runs = {label: run(signal, members, returns, days, cost, sectors, caps=caps, **options)
            for label, options in setups}
    window = min((net.index for _, net, _ in runs.values()), key=len)   # 全部对齐到最短的那段
    equal = equal.reindex(window).dropna()
    rows, series = {}, {}
    for label, (_, net, records) in runs.items():
        long_only = '纯多头' in label
        rows[label] = summarise(net.reindex(window).dropna(), records, equal if long_only else None)
        series[label] = net.reindex(window)
    # 池子本身也按同样的方法量一次行业集中度，这样右图就有了一根真实的基准柱子，
    # 不用再画一条硬编码的参考线（之前那条 16.5% 是从短样本的 GICS 口径来的，用在这里是错的）
    pool = []
    for day, _ in periods(signal.index, days):
        names = members.columns[members.iloc[day].to_numpy()]
        share = pd.Series([sectors.get(i, '未知') for i in names]).value_counts(normalize=True)
        pool.append(dict(多头只数=len(names), 空头只数=0, 多头行业数=len(share),
                         多头最大行业占比=float(share.iloc[0]), traded=np.nan))
    rows['标普 500 等权（对照）'] = summarise(equal, pd.DataFrame(pool))
    series['标普 500 等权（对照）'] = equal

    summary = pd.DataFrame(rows).T
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / 'breadth_summary.csv', float_format='%.6f')
    pd.DataFrame(series).to_csv(args.output / 'breadth_daily.csv', float_format='%.8f')
    pd.set_option('display.width', 240)
    print(f'同一池子、同一信号、单边 {args.cost_bps:g}bp、{"每月" if args.schedule == "M" else "每周"}调仓')
    print(summary.round(4).to_string())
    if wanted(args.plot):
        span = pd.DataFrame(series).dropna().index           # 区间从数据里取，不写死
        years = (span[-1] - span[0]).days / 365.25
        tail = ('行业用 Fama-French 12 分类，灰柱是池子本身。'
                '为避开配色重复，「前 30% 多空」只在表里、不在图上。')
        if args.crsp:
            title = (f'{years:.0f} 年、{int(summary["调仓次数"].max())} 次调仓：'
                     '分散到 48 只，收益略降但夏普和回撤都更好')
            note = (f'CRSP 标普 500 成分股（含被剔除的，无幸存者偏差）。曲线从 {span[0]:%Y-%m-%d} 起——'
                    f'{args.start} 之后的前 252 个交易日用于动量预热。'
                    '11-1 动量、单边 10bp、每月调仓、等权。\n' + tail)
        else:
            title = '持股数从 10 只放宽到 48 只：收益腰斩，但夏普更高、回撤减半'
            note = ('同一个标普 500 池子、同一个 11-1 动量信号、单边 10bp、每月调仓。'
                    '「行业中性」是在每个行业内部各取前后 10%。\n'
                    f'只有 {int(summary["调仓次数"].max())} 次调仓（{len(span)} 个交易日），'
                    '没有一个 t 值超过 2，差异都在噪声范围内。' + tail)
        plot(series, summary, args.plot, title, note)
    if wanted(args.focus):
        target = '前 10% 纯多头'
        _, net, records = runs[target]
        rivals = {'前 10 只 纯多头（momentum.py）': series['前 10 只 纯多头（momentum.py）'],
                  '标普 500 等权（对照）': equal}
        focus(net.reindex(window).dropna(), records, rivals, args.focus, target)
    print(f"\n写入 {args.output / 'breadth_summary.csv'}")


if __name__ == '__main__':
    main()
