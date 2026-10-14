"""给 11-1 动量组合加对冲，比较几种做法。

多头都是同一个组合（标普 500 里动量最强的 10 只，等权，每月调仓），只改对冲的方式：

- `none`：不对冲。
- `spy_1x` / `qqq_1x`：卖空等额的 SPY / QQQ（金额中性）。
- `spy_beta` / `qqq_beta`：按 beta 卖空 SPY / QQQ（beta 中性）。beta 在每个调仓日用过去 252 天的日收益估计，
  取 10 只持仓各自 beta 的平均，只用当天及之前的数据。
- `sector`：卖空同行业的成分股等权篮子，行业权重和多头一致（行业中性、金额中性）。
- `long_short`：卖空动量最弱的 10 只（AQR 的经典做法）。
- `vixy_5` / `vixy_10` / `vixm_10`：额外买入 5% 或 10% 的 VIX 期货 ETP（VIXY 是近月，VIXM 是中期），
  每次调仓把这部分仓位调回目标比例。这是额外加的仓位，不减少多头。

对冲的买卖也按单边 10bp 计成本：ETF 按每次调仓调整的名义金额计，篮子按实际换掉的比例计。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.backtest import hold, trade_size
from ..factors.metrics import COST_BPS
from . import universe
from .momentum import TOP, WINDOW, load, momentum, period_ends, periods

WHY = Path(__file__).resolve().parent.parent
HEDGES = ['none', 'spy_1x', 'spy_beta', 'qqq_1x', 'qqq_beta', 'sector', 'long_short',
          'vixy_5', 'vixy_10', 'vixm_10']
VIX_SLEEVES = {'vixy_5': ('VIXY', 0.05), 'vixy_10': ('VIXY', 0.10), 'vixm_10': ('VIXM', 0.10)}


def trailing_beta(returns, benchmark, day, names, window=WINDOW, minimum=60):
    """调仓日用过去 window 天的日收益估计持仓的 beta：每只股票对基准回归，再取等权平均。"""
    past = returns.iloc[max(0, day - window + 1):day + 1][sorted(names)]
    market = benchmark.reindex(past.index)
    variance = market.var()
    betas = [past[name].cov(market) / variance for name in past.columns
             if past[name].notna().sum() >= minimum]
    return float(np.mean(betas)) if betas else 1.0


def sector_basket(members, sectors, day, names):
    """和多头行业权重相同的空头篮子：每个行业挑该行业的全部成分股等权，行业权重按多头持仓的比例。"""
    held = pd.Series([sectors.get(name) for name in names], index=list(names)).dropna()
    weights = held.value_counts(normalize=True)
    available = members.columns[members.iloc[day].to_numpy()]
    basket = {}
    for sector, weight in weights.items():
        peers = [name for name in available if sectors.get(name) == sector]
        for name in peers:
            basket[name] = basket.get(name, 0.0) + weight / len(peers)
    return pd.Series(basket)


def run(signal, members, returns, days, benchmarks, sectors, cost, top=TOP):
    """一次遍历所有调仓期，算出各种对冲方式的每日收益（已扣成本）。"""
    books, ratios = {name: [] for name in HEDGES}, []
    held, short_held, notional = set(), set(), {name: 0.0 for name in HEDGES}
    sleeve = {name: 0.0 for name in VIX_SLEEVES}
    basket_held = pd.Series(dtype=float)
    for day, span in periods(signal.index, days):
        scores = signal.iloc[day].where(members.iloc[day]).dropna()
        chosen = set(scores.nlargest(top).index)
        weakest = set(scores.nsmallest(top).index)
        window = returns.iloc[span]
        long_leg = hold(window, chosen)
        traded = trade_size(held, chosen)
        basket = sector_basket(members, sectors, day, chosen)
        beta = {'spy': trailing_beta(returns, benchmarks['SPY'], day, chosen),
                'qqq': trailing_beta(returns, benchmarks['QQQ'], day, chosen)}
        legs = {'none': (pd.Series(0.0, index=window.index), 0.0),
                'spy_1x': (benchmarks['SPY'].reindex(window.index).fillna(0), 1.0),
                'spy_beta': (benchmarks['SPY'].reindex(window.index).fillna(0) * beta['spy'], beta['spy']),
                'qqq_1x': (benchmarks['QQQ'].reindex(window.index).fillna(0), 1.0),
                'qqq_beta': (benchmarks['QQQ'].reindex(window.index).fillna(0) * beta['qqq'], beta['qqq']),
                'sector': ((window[sorted(basket.index)].fillna(0) * basket[sorted(basket.index)]).sum(axis=1), 1.0),
                'long_short': (hold(window, weakest), 1.0)}
        for name, (symbol, weight) in VIX_SLEEVES.items():
            # 买入的仓位随 ETP 涨跌漂移，下次调仓再调回目标比例；收益按每天的仓位计
            growth = (1 + benchmarks[symbol].reindex(window.index).fillna(0)).cumprod()
            legs[name] = (-weight * (growth - growth.shift(1, fill_value=1.0)), weight)
        for name, (short_leg, size) in legs.items():
            charge = traded
            if name in ('sector',):
                charge += trade_size(set(basket_held.index), set(basket.index))
            elif name == 'long_short':
                charge += trade_size(short_held, weakest)
            elif name in VIX_SLEEVES:
                charge += abs(size - sleeve[name])
            else:
                charge += abs(size - notional[name])
            book = long_leg - short_leg
            book = book.copy()
            book.iloc[0] -= charge * cost
            books[name].append(book)
            notional[name] = size
        for name, (symbol, weight) in VIX_SLEEVES.items():
            sleeve[name] = weight * (1 + benchmarks[symbol].reindex(window.index).fillna(0)).prod()
        ratios.append(dict(signal_date=signal.index[day], trade_date=signal.index[day + 1], names=len(chosen),
                           beta_spy=beta['spy'], beta_qqq=beta['qqq']))
        held, short_held, basket_held = chosen, weakest, basket
    return {name: pd.concat(parts) for name, parts in books.items()}, pd.DataFrame(ratios)


def statistics(net, benchmark):
    curve = net.cumsum()
    market = benchmark.reindex(net.index).fillna(0)
    return dict(annual=net.mean() * 252, sharpe=net.mean() / net.std() * np.sqrt(252),
                volatility=net.std() * np.sqrt(252), max_drawdown=(curve - curve.cummax()).min(),
                beta=np.cov(net, market)[0, 1] / market.var(), total=net.sum(),
                worst_month=net.groupby(net.index.to_period('M')).sum().min())


def main():
    parser = argparse.ArgumentParser(description='Compare hedges for the 11-1 momentum portfolio')
    parser.add_argument('--data', type=Path, default=WHY / 'data' / 'us_daily.parquet')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--top', type=int, default=TOP)
    parser.add_argument('--cost-bps', type=float, default=COST_BPS)
    args = parser.parse_args()
    cost = args.cost_bps / 1e4

    daily, prices, returns = load(args.data)
    table = universe.load(args.output / 'sp500_constituents.csv')
    members = universe.membership(daily, table, prices)
    signal = momentum(prices)
    ends = period_ends(prices.index)
    days = [day for day in ends if day < len(prices.index) - 2
            and signal.iloc[day].where(members.iloc[day]).notna().sum() >= 2 * args.top]
    names = daily.pivot(index='date', columns='instrument_id', values='symbol')
    sector_by_symbol = table.set_index('symbol')['sector']
    sectors = {instrument: sector_by_symbol.get(names[instrument].dropna().iloc[-1])
               for instrument in members.columns[members.any().to_numpy()]}
    benchmarks = pd.read_parquet(args.data.parent / 'benchmarks.parquet').pivot(
        index='date', columns='symbol', values='ret')

    books, ratios = run(signal, members, returns, days, benchmarks, sectors, cost, args.top)
    ratios.to_csv(args.output / 'hedge_betas.csv', index=False, float_format='%.4f')
    summary = {name: statistics(net, benchmarks['SPY']) for name, net in books.items()}
    summary['SPY'] = statistics(benchmarks['SPY'].reindex(books['none'].index).fillna(0), benchmarks['SPY'])
    pd.DataFrame(books).rename_axis('date').to_csv(args.output / 'hedge_daily.csv', float_format='%.6f')
    (args.output / 'hedge_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    frame = pd.DataFrame(summary).T
    pd.set_option('display.width', 200)
    print(frame.round(3).to_string())
    print(f'saved {args.output / "hedge_summary.json"}')


if __name__ == '__main__':
    main()
