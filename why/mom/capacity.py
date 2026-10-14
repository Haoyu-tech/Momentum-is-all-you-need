"""策略容量：按成交额算这个策略最多能装多少钱。

回测里的成本是固定单边 10bp，这等于假设「交易量对价格没有影响」。规模一大这个假设就破了，
所以容量要单独算，不能靠回测。口径是业界常用的那条：

    单只股票单日的成交金额 ≤ 该股过去 21 日**中位**日均成交额 × 参与率

参与率默认 10%。取中位而不是均值，是为了不让财报日、指数调整日那几天的异常放大容量。
于是某一次调仓能承载的规模是

    容量 = min_i ( 参与率 × ADV_i ) / ( 杠杆 × |Δw_i| )

`min` 是因为**最紧的那只票决定全局**——组合要等权，买不满那一只就得整体缩小。这是保守口径。
另外给一个「可以在 N 天里摊开交易」的版本（`--spread`），以及一个总量口径
（允许在不同股票之间调配参与率），三个数一起看才知道约束到底卡在哪。

容量是**名义美元**，所以 1960 年的 3 千万和 2025 年的 3 千万不是一回事。主表按年代分段给，
并同时报 CPI 调整到今天的口径；真正用于决策的是最近十年那一行。

输出 `data/mom/capacity_summary.csv`、`capacity_series.csv` 和 `docs/momentum/6_capacity.png`。
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.metrics import COST_BPS
from . import crsp
from .breadth import run, wanted, weigh
from .final import BEST_TARGET
from .momentum import momentum, period_ends
from .overlay import monthly, realized_vol

WHY = Path(__file__).resolve().parent.parent
VOLUME_COLUMNS = ['PERMNO', 'DlyCalDt', 'DlyVol', 'DlyClose']
ADV_DAYS = 21          # 一个月的交易日
PARTICIPATION = 0.10   # 单日最多吃掉一只股票日均成交额的 10%
SPREAD = 1             # 默认要求一天之内换完（和回测的「次日收盘成交」一致）

# 美国 CPI 年均值（1982-84=100），用来把名义容量折算到 2025 年的购买力。
# 来源：BLS CUUR0000SA0 年度平均，取十年一档够用了——我们要的是量级，不是精度。
CPI = {1960: 29.6, 1970: 38.8, 1980: 82.4, 1990: 130.7, 2000: 172.2, 2010: 218.1, 2025: 320.6}


def deflate(year, base=2025):
    """把 year 年的名义美元折算成 base 年的购买力，线性插值。"""
    years = sorted(CPI)
    return np.interp(base, years, [CPI[y] for y in years]) / np.interp(year, years, [CPI[y] for y in years])


def extract(source=crsp.SOURCE, cache=None, force=False):
    """从 CRSP 成分股文件里抽出日频成交金额（股数 × 收盘价），缓存成 parquet。

    只取成员期内的记录就够了——我们只在股票是成分股的时候买它。
    """
    cache = cache or WHY / 'data' / 'mom' / 'crsp_volume.parquet'
    if cache.exists() and not force:
        return cache
    if not source.exists():
        raise FileNotFoundError(f'找不到 CRSP 源文件：{source}')
    archive = zipfile.ZipFile(source)
    pieces = []
    for chunk in pd.read_csv(archive.open(archive.namelist()[0]), usecols=VOLUME_COLUMNS,
                             chunksize=2_000_000, low_memory=False):
        chunk['date'] = pd.to_datetime(chunk.pop('DlyCalDt'), errors='coerce')
        chunk['dollars'] = chunk.pop('DlyVol') * chunk.pop('DlyClose')
        pieces.append(chunk.dropna(subset=['date', 'dollars'])[['date', 'PERMNO', 'dollars']])
        print(f'  已读 {sum(len(p) for p in pieces):,} 行', flush=True)
    frame = pd.concat(pieces, ignore_index=True).rename(columns={'PERMNO': 'permno'})
    frame = frame.sort_values(['date', 'permno'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    print(f'缓存 {len(frame):,} 行到 {cache}')
    return cache


def advance(cache=None, days=ADV_DAYS, start=None):
    """每只股票过去 `days` 天的中位日成交金额，整体后移一天（只用历史数据）。"""
    frame = pd.read_parquet(cache or extract())
    if start:
        # 多留一年，给滚动窗口预热
        frame = frame[frame['date'] >= pd.Timestamp(start) - pd.Timedelta(days=400)]
    wide = frame.pivot_table(index='date', columns='permno', values='dollars', aggfunc='last')
    return wide.rolling(days, min_periods=days // 2).median().shift(1)


def book(start='1957-03-01', cost_bps=COST_BPS, target=BEST_TARGET, value=False):
    """重建最终策略的持仓和杠杆。默认等权——那正是容量最紧的那一版。"""
    returns, caps, members, sectors, _ = crsp.full(start=start)
    prices = (1 + returns.fillna(0)).cumprod().where(returns.notna())
    signal = momentum(prices)
    days = period_ends(signal.index, 'M')
    _, net, records = run(signal, members, returns, days, cost_bps / 1e4, sectors,
                          fraction=0.1, short=False, caps=caps if value else None)
    marks = [net.index[i] for i in range(len(net))
             if i == 0 or net.index[i].to_period('M') != net.index[i - 1].to_period('M')]
    leverage = monthly(target / realized_vol(net), marks).clip(upper=3.0)
    # 容量算的是**每只票要成交多少钱**，所以必须拿到真实权重，不能一律按 1/N——
    # 市值加权下最小的那只票权重远小于 1/N，约束松得多。
    spots = signal.index.get_indexer(records['signal_date'])
    books = [(date, weigh(names, caps if value else None, day))
             for date, names, day in zip(records['signal_date'], records['持仓'], spots)]
    return books, leverage.reindex(net.index).ffill()


def changes(books):
    """每次调仓每只票的权重变动 |Δw|。"""
    rows, before = [], pd.Series(dtype=float)
    for date, now in books:
        keys = set(now.index) | set(before.index)
        delta = (now.reindex(keys).fillna(0.0) - before.reindex(keys).fillna(0.0)).abs()
        rows.append((date, delta[delta > 1e-12]))
        before = now
    return rows


def capacity(books, leverage, volume, participation=PARTICIPATION, spread=SPREAD):
    """每次调仓能承载的规模（名义美元）。三个口径一起给。

    `紧约束`：最紧的那只票决定全局，组合严格等权。
    `总量`  ：允许在股票之间调配参与率，只要总交易额不超过总 ADV 的参与率。
    """
    rows = []
    for date, delta in changes(books):
        if date not in volume.index or not len(delta):
            continue
        adv = volume.loc[date].reindex(delta.index).dropna()
        delta = delta.reindex(adv.index)
        gear = float(leverage.get(date, 1.0)) if np.isfinite(leverage.get(date, np.nan)) else 1.0
        if not len(adv) or adv.min() <= 0 or gear <= 0:
            continue
        allowed = participation * adv * spread          # 这只票这次最多能成交多少钱
        tight = float((allowed / (gear * delta)).min())
        total = float(allowed.sum() / (gear * delta.sum()))
        rows.append(dict(调仓日=date, 持股数=len(delta), 杠杆=gear,
                         换手=float(delta.sum()), 紧约束容量=tight, 总量容量=total,
                         最紧的票=int(delta.index[(allowed / (gear * delta)).to_numpy().argmin()]),
                         持仓中位ADV=float(adv.median()), 持仓最小ADV=float(adv.min())))
    frame = pd.DataFrame(rows).set_index('调仓日')
    frame['紧约束容量_2025'] = [v * deflate(d.year) for d, v in frame['紧约束容量'].items()]
    frame['总量容量_2025'] = [v * deflate(d.year) for d, v in frame['总量容量'].items()]
    return frame


ERAS = [('1958-1974', 1958, 1974), ('1975-1999', 1975, 1999),
        ('2000-2015', 2000, 2015), ('2016-2025', 2016, 2025)]


def summarise(frame):
    """按年代分段给容量的中位数和最差的那一年。"""
    rows = []
    for label, low, high in ERAS:
        part = frame[(frame.index.year >= low) & (frame.index.year <= high)]
        if not len(part):
            continue
        rows.append(dict(区间=label, 调仓次数=len(part),
                         紧约束中位=part['紧约束容量'].median(),
                         紧约束下四分位=part['紧约束容量'].quantile(0.25),
                         总量中位=part['总量容量'].median(),
                         紧约束中位_2025年美元=part['紧约束容量_2025'].median(),
                         持仓中位ADV=part['持仓中位ADV'].median(),
                         换手中位=part['换手'].median(), 杠杆中位=part['杠杆'].median()))
    return pd.DataFrame(rows).set_index('区间')


def plot(frame, table, path, participation=PARTICIPATION):
    """左：容量随时间（对数轴，名义 vs 实际）；右：最近十年的分布。"""
    from matplotlib.ticker import FuncFormatter

    from ..charts import AXIS, INK, INK2, MUTED, SERIES, plt, style, tidy
    style()
    recent = table.index[-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                      gridspec_kw=dict(width_ratios=[1.9, 1], wspace=0.22))
    fig.suptitle(f'策略容量：最近十年中位 {table.loc[recent, "紧约束中位"] / 1e6:,.0f} 百万美元',
                 x=0.042, ha='left', y=0.97, fontsize=15, color=INK, weight='bold')
    fig.text(0.042, 0.885,
             f'口径：单只股票单日成交额不超过它过去 {ADV_DAYS} 日中位成交额的 {participation:.0%}，一天内换完。\n'
             '「紧约束」是最紧的那只票决定全局；「总量」允许在股票之间调配参与率，'
             '两条线的差距就是「被一只小票卡住」的代价。\n'
             '容量是名义美元，灰线是折算到 2025 年购买力后的紧约束口径。',
             fontsize=9.5, color=INK2, va='top', linespacing=1.5)
    for axis in (left, right):
        tidy(axis)
    for name, colour, width in [('总量容量', SERIES[1], 1.4), ('紧约束容量', SERIES[0], 1.8)]:
        left.plot(frame.index, frame[name], linewidth=width, color=colour, label=name)
    left.plot(frame.index, frame['紧约束容量_2025'], linewidth=1.2, color=MUTED,
              linestyle=(0, (5, 2)), label='紧约束（2025 年美元）')
    left.set_yscale('log')
    left.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f'{v / 1e9:,.0f}B' if v >= 1e9 else (f'{v / 1e6:,.0f}M' if v >= 1e6 else f'{v / 1e3:,.0f}K')))
    left.set_title('每次调仓能承载的规模（对数轴）', loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc='upper left')

    last = frame[frame.index.year >= 2016]['紧约束容量'] / 1e6
    right.hist(last, bins=24, color=SERIES[0], edgecolor='white', linewidth=0.8)
    ceiling = right.get_ylim()[1]
    # 两条参考线挨得近，标注一高一低、一左一右分开，不然会叠在一起
    for value, label, colour, height, side in [(last.median(), '中位', INK, 0.96, 5),
                                               (last.quantile(0.1), '最差 10%', SERIES[2], 0.82, -5)]:
        right.axvline(value, color=colour, linewidth=1.3, linestyle=(0, (4, 2)))
        right.annotate(f'{label} {value:,.0f}M', (value, ceiling * height), (side, 0),
                       textcoords='offset points', fontsize=9, color=colour, va='top',
                       ha='left' if side > 0 else 'right')
    right.set_title('2016 年以来每次调仓的容量（百万美元）', loc='left', fontsize=11.5, color=INK, pad=10)
    right.axhline(0, color=AXIS, linewidth=1)
    fig.subplots_adjust(top=0.70, bottom=0.09, left=0.055, right=0.97)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def main():
    parser = argparse.ArgumentParser(description='Capacity of the momentum strategy from traded volume')
    parser.add_argument('--start', default='1957-03-01')
    parser.add_argument('--participation', type=float, default=PARTICIPATION)
    parser.add_argument('--spread', type=int, default=SPREAD, help='允许摊到几天里换完')
    parser.add_argument('--value', action='store_true', help='市值加权版（容量大得多）')
    parser.add_argument('--extract-only', action='store_true', help='只抽成交额缓存，不跑容量')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--plot', type=Path, default=WHY / 'docs' / 'momentum' / '6_capacity.png')
    args = parser.parse_args()

    cache = extract(force=args.force)
    if args.extract_only:
        return
    volume = advance(cache, start=args.start)
    books, leverage = book(args.start, value=args.value)
    frame = capacity(books, leverage, volume, args.participation, args.spread)
    table = summarise(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'capacity_series.csv', float_format='%.2f')
    table.to_csv(args.output / 'capacity_summary.csv', float_format='%.2f')
    pd.set_option('display.width', 220)
    print(f'\n{frame.index[0].date()} 至 {frame.index[-1].date()}，{len(frame)} 次调仓，'
          f'参与率 {args.participation:.0%}、{args.spread} 天换完，'
          f'{"市值加权" if args.value else "等权"}')
    print((table / 1e6).round(1).to_string())
    if wanted(args.plot):
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(frame, table, args.plot, args.participation)
    print(f"\n写入 {args.output / 'capacity_summary.csv'}")


if __name__ == '__main__':
    main()
