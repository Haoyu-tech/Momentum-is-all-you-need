"""AQR 风格的 11-1 动量：每月底在标普 500 成分股里选动量最强的 10 只，等权持有一个月。

- **动量**：过去 12 个月、跳过最近 1 个月的涨幅，即后复权价 t-21 日 ÷ t-252 日 − 1（11 个月的涨幅）。
  跳过最近一个月是 AQR 的标准做法，避开一个月的短期反转。
- **调仓**：每个月最后一个交易日出信号，下一个交易日收盘成交，持有到下个月的成交日。
- **成本**：单边 10bp（与 why/factors 的默认一致），按每次调仓实际买卖的比例计算，首次建仓也计入。
- **对照**：SPY 买入持有，以及同一批成分股等权、同样每月调仓。

数据需要 12 个月预热，所以第一个信号日是 2025-07-02，回测只有约 14 个月、14 次调仓，结论只能当参考。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.backtest import hold, trade_size
from ..factors.metrics import COST_BPS
from . import universe

WHY = Path(__file__).resolve().parent.parent
WINDOW, SKIP, TOP = 252, 21, 10  # 12 个月、跳过 1 个月、选 10 只
COVERAGE = 0.8  # 窗口内至少要有这么大比例的交易日有价格，才算有足够历史


def load(path):
    """读 us_daily.parquet，返回原表和三张宽表：后复权价、收益、每天有没有收盘价。"""
    daily = pd.read_parquet(path, columns=['date', 'instrument_id', 'symbol', 'close', 'adj_close_pit', 'ret'])
    prices = daily.pivot(index='date', columns='instrument_id', values='adj_close_pit')
    returns = daily.pivot(index='date', columns='instrument_id', values='ret')
    return daily, prices, returns


def momentum(prices, window=WINDOW, skip=SKIP):
    """11-1 动量：t-skip 日的价格 ÷ t-window 日的价格 − 1；窗口内价格太少的股票留空。"""
    signal = prices.shift(skip) / prices.shift(window) - 1
    enough = prices.notna().rolling(window).sum() >= window * COVERAGE
    return signal.where(enough)


def period_ends(dates, schedule='M'):
    """每个自然月（'M'）或每周（'W'）最后一个交易日的位置。"""
    last = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period(schedule)).max()
    return list(last)


def periods(dates, days):
    """依次返回（信号日位置，持有期切片）：信号日 + 1 收盘成交，收益从 +2 天算到下一个成交日（含）。"""
    last = len(dates) - 1
    for n, day in enumerate(days):
        leave = min(days[n + 1] + 1, last) if n + 1 < len(days) else last
        if day + 2 > leave:
            return
        yield day, slice(day + 2, leave + 1)


def backtest(signal, members, returns, days, cost, top=TOP):
    """每次调仓买入动量最强的 top 只，等权持有到下次成交。返回每日收益（扣成本前后）和每次调仓的持仓。"""
    gross, net, records = [], [], []
    held = set()
    for day, span in periods(signal.index, days):
        chosen = signal.iloc[day].where(members.iloc[day]).dropna().nlargest(top)
        window = returns.iloc[span]
        traded = trade_size(held, set(chosen.index))
        held = set(chosen.index)
        spread = hold(window, held)
        gross.append(spread)
        charged = spread.copy()
        charged.iloc[0] -= traded * cost
        net.append(charged)
        records.append(dict(signal_date=signal.index[day], trade_date=signal.index[day + 1], traded=traded,
                            names=list(chosen.index), momentum=list(chosen.to_numpy())))
    return pd.concat(gross), pd.concat(net), pd.DataFrame(records)


def equal_weight(members, returns, days):
    """同一批成分股等权、同样每月调仓，作为对照。"""
    frames = [hold(returns.iloc[span], set(members.columns[members.iloc[day].to_numpy()]))
              for day, span in periods(members.index, days)]
    return pd.concat(frames)


def statistics(gross, net, records, benchmark):
    curve = net.cumsum()
    extra = net - benchmark.reindex(net.index).fillna(0)
    return dict(days=len(net), rebalances=len(records), start=str(net.index[0].date()), end=str(net.index[-1].date()),
                gross_annual=gross.mean() * 252, net_annual=net.mean() * 252,
                net_sharpe=net.mean() / net.std() * np.sqrt(252), volatility=net.std() * np.sqrt(252),
                max_drawdown=(curve - curve.cummax()).min(), total=net.sum(),
                annual_cost=(gross - net).sum() / len(net) * 252,
                turnover_per_rebalance=records['traded'].iloc[1:].mean() / 2,
                excess_annual=extra.mean() * 252, tracking_error=extra.std() * np.sqrt(252),
                beta=np.cov(net, benchmark.reindex(net.index).fillna(0))[0, 1] / benchmark.reindex(net.index).var())


def main():
    parser = argparse.ArgumentParser(description='AQR-style 11-1 momentum on S&P 500 constituents')
    parser.add_argument('--data', type=Path, default=WHY / 'data' / 'us_daily.parquet')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--top', type=int, default=TOP, help='Number of stocks held')
    parser.add_argument('--schedule', default='M', choices=['M', 'W'], help='Rebalance at each month or week end')
    parser.add_argument('--cost-bps', type=float, default=COST_BPS, help='One-way trading cost per side')
    parser.add_argument('--offsets', type=int, nargs='+', default=None,
                        help='Rebalance this many trading days after each period end; the first one is the main result')
    parser.add_argument('--refresh-universe', action='store_true', help='Fetch the constituent list again')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cost = args.cost_bps / 1e4

    daily, prices, returns = load(args.data)
    table = universe.load(args.output / 'sp500_constituents.csv', args.refresh_universe)
    members = universe.membership(daily, table, prices)
    signal = momentum(prices)
    ends = period_ends(prices.index, args.schedule)
    offsets = args.offsets if args.offsets else ([0, 5, 10, 15] if args.schedule == 'M' else [0, 1, 2, 3])
    suffix = '' if args.schedule == 'M' else '_week'
    benchmark = pd.read_parquet(args.data.parent / 'benchmarks.parquet').query("symbol == 'SPY'").set_index('date')['ret']

    results = {}
    for offset in offsets:
        days = [day + offset for day in ends if day + offset < len(prices.index) - 2]
        days = [day for day in days if signal.iloc[day].where(members.iloc[day]).notna().sum() >= args.top]
        gross, net, records = backtest(signal, members, returns, days, cost, args.top)
        results[offset] = (gross, net, records)
        print(f'offset {offset:2d}d: net annual {net.mean() * 252:+.1%}  sharpe '
              f'{net.mean() / net.std() * np.sqrt(252):.2f}  rebalances {len(records)}', flush=True)

    gross, net, records = results[offsets[0]]
    summary = statistics(gross, net, records, benchmark)
    summary['schedule'] = args.schedule
    summary['offsets'] = {f'{offset}d': results[offset][1].mean() * 252 for offset in offsets}
    equal = equal_weight(members, returns, [day for day in ends if day < len(prices.index) - 2])
    summary['rebalances_per_year'] = len(records) / (len(net) / 252)
    summary['equal_weight_annual'] = equal.reindex(net.index).mean() * 252
    summary['benchmark_annual'] = benchmark.reindex(net.index).fillna(0).mean() * 252
    summary['members_per_day'] = int(members.loc[net.index].sum(axis=1).median())

    names = daily.pivot(index='date', columns='instrument_id', values='symbol')
    sectors = table.set_index('symbol')['sector']
    rows = []
    for record in records.itertuples():
        for rank, (instrument, value) in enumerate(zip(record.names, record.momentum), start=1):
            symbol = names.loc[record.signal_date, instrument]
            rows.append(dict(signal_date=record.signal_date, trade_date=record.trade_date, rank=rank, symbol=symbol,
                             instrument_id=instrument, sector=sectors.get(symbol), momentum=value,
                             traded=record.traded))
    holdings = pd.DataFrame(rows)
    summary['sector_share'] = (holdings['sector'].value_counts() / len(holdings)).round(4).to_dict()
    summary['distinct_names'] = int(holdings['symbol'].nunique())
    holdings.to_csv(args.output / f'momentum_holdings{suffix}.csv', index=False, float_format='%.4f')
    pd.DataFrame({'gross': gross, 'net': net, 'equal_weight': equal.reindex(net.index),
                  'spy': benchmark.reindex(net.index)}).rename_axis('date').to_csv(
        args.output / f'momentum_daily{suffix}.csv', float_format='%.6f')
    (args.output / f'momentum_summary{suffix}.json').write_text(json.dumps(summary, indent=2, default=float),
                                                                encoding='utf-8')

    pd.set_option('display.width', 200)
    print(json.dumps(summary, indent=2, default=lambda v: round(float(v), 4)))
    print('latest holdings:')
    print(holdings[holdings['signal_date'] == holdings['signal_date'].max()][['rank', 'symbol', 'momentum']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
