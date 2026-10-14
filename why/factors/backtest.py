"""按固定频率调仓的多空回测。

每 every 个交易日调一次仓：t 日收盘出信号，t+1 日收盘成交，做多信号最高的 quantile、做空最低的 quantile，
等权买入后一直持有到下次成交。持有期内缺失的收益（退市、停牌）按 0 计。成本按每次调仓实际买卖的比例计算。

只用一份资金时，结果取决于调仓定在哪一天（每周调仓有 5 种选法，每月有 20 种），差别可以很大。
staggered 把资金分成 every 份、错开日子调仓（分批调仓），等于把这些选法平均，低频回测默认用它。
"""
import numpy as np
import pandas as pd

from .metrics import QUANTILE


def periods(dates, start, every, offset=0):
    """从 start 起每 every 个交易日一个信号日，依次返回（信号日位置，持有期切片）。

    offset > 0 时是分批调仓里的第 offset 份：start 当天照常建仓，之后从 start 后第 offset 个交易日起每 every 天调仓。
    持有期从成交日（信号日 + 1）的下一天开始，到下一次成交日为止（含），最后一期到数据最后一天。
    """
    first = int(np.flatnonzero(dates >= start)[0])
    signal_days = np.arange(first + offset, len(dates), every)
    if offset:
        signal_days = np.r_[first, signal_days]
    last = len(dates) - 1
    for n, day in enumerate(signal_days):
        leave = min(signal_days[n + 1] + 1, last) if n + 1 < len(signal_days) else last
        if day + 2 > leave:
            return
        yield day, slice(day + 2, leave + 1)


def split_books(score, quantile):
    """按截面分位数取多头（前 quantile）和空头（后 quantile）；score 中的空值不参与排名。"""
    percentile = score.rank(pct=True)
    return set(percentile.index[percentile > 1 - quantile]), set(percentile.index[percentile <= quantile])


def hold(window, names):
    """等权买入 names 后一直持有，返回组合在 window 每天的收益。"""
    growth = (1 + window[sorted(names)].fillna(0)).cumprod().mean(axis=1)
    return growth / growth.shift(1, fill_value=1.0) - 1


def trade_size(old, new):
    """调仓时卖出离开的、买入新进的，两者各占组合的比例之和；首次建仓为 1。"""
    return len(new - old) / len(new) + (len(old - new) / len(old) if old else 0.0)


def backtest(signal, universe, returns, every, start, cost, quantile=QUANTILE, offset=0):
    """返回每日多空收益（未扣成本、扣成本）和每次调仓的记录（持有起止日、换手、两边的持仓）。成本记在持有期第一天。"""
    gross, net, records = [], [], []
    held_long, held_short = set(), set()
    for day, span in periods(signal.index, start, every, offset):
        longs, shorts = split_books(signal.iloc[day].where(universe.iloc[day]), quantile)
        window = returns.iloc[span]
        traded = trade_size(held_long, longs) + trade_size(held_short, shorts)
        held_long, held_short = longs, shorts
        spread = hold(window, longs) - hold(window, shorts)
        gross.append(spread)
        charged = spread.copy()
        charged.iloc[0] -= traded * cost
        net.append(charged)
        records.append(dict(date=signal.index[day], start=window.index[0], end=window.index[-1], turnover=traded / 4,
                            traded=traded, longs=len(longs), shorts=len(shorts), long_ids=sorted(longs),
                            short_ids=sorted(shorts)))
    return pd.concat(gross), pd.concat(net), pd.DataFrame(records)


def staggered(signal, universe, returns, every, start, cost, quantile=QUANTILE):
    """分批调仓：资金等分成 every 份，都在 start 建仓，第 k 份从 start 后第 k 个交易日起每 every 天调仓，每天的收益取各份平均。

    返回每日多空收益（未扣成本、扣成本）、各份的调仓记录（book 列是第几份），以及各份单独的扣成本每日收益（每列一份）。
    every = 1 时只有一份，和 backtest 完全相同。
    """
    books = [backtest(signal, universe, returns, every, start, cost, quantile, offset) for offset in range(every)]
    nets = pd.concat([net for _, net, _ in books], axis=1)
    gross = pd.concat([gross for gross, _, _ in books], axis=1).mean(axis=1)
    trades = pd.concat([trades.assign(book=k) for k, (_, _, trades) in enumerate(books)], ignore_index=True)
    return gross, nets.mean(axis=1), trades, nets


def book_returns(signal, universe, returns, every, start, quantile=QUANTILE):
    """多头、空头和整个股票池各自的每日收益（都按等权买入、持有到下次调仓，不扣成本），按分批调仓取各份平均，用来拆解收益来源。"""
    books = []
    for offset in range(every):
        frames = []
        for day, span in periods(signal.index, start, every, offset):
            score = signal.iloc[day].where(universe.iloc[day])
            longs, shorts = split_books(score, quantile)
            window = returns.iloc[span]
            frames.append(pd.DataFrame({'long': hold(window, longs), 'short': hold(window, shorts),
                                        'universe': hold(window, set(score.dropna().index))}))
        books.append(pd.concat(frames))
    return sum(books) / len(books)


def equal_groups(score, groups, seed=0):
    """把 score 等分成 groups 组（1 为最低）。分数相同的股票按固定种子随机打散，保证每组股票数相同：
    CNTP5 这类只有几种取值的因子，否则一整批并列的股票会挤进同一组，别的组就空了。"""
    shuffled = score.sample(frac=1, random_state=seed)
    order = shuffled.rank(method='first', pct=True).reindex(score.index)
    return np.ceil(order * groups).astype(int)


def group_returns(signal, universe, returns, every, start, groups=10):
    """分层回测：每次调仓把股票池按信号等分成 groups 组（第 1 组信号最低），每组等权买入、持有到下次调仓，不扣成本。

    和 staggered 一样分批调仓，每天的收益取各份平均。返回每天、每组的收益，列名是 1 至 groups。
    """
    books = []
    for offset in range(every):
        frames = []
        for day, span in periods(signal.index, start, every, offset):
            score = signal.iloc[day].where(universe.iloc[day]).dropna()
            group = equal_groups(score, groups, seed=day)
            window = returns.iloc[span]
            frames.append(pd.DataFrame({k: hold(window, set(score.index[group == k])) for k in range(1, groups + 1)}))
        books.append(pd.concat(frames))
    return sum(books) / len(books)


def statistics(gross, net, trades, every):
    """汇总指标。trades 可以带 book 列（分批调仓）：每份的第一次是建仓，不计入换手；调仓次数按一份计。"""
    curve = net.cumsum()
    book = trades['book'] if 'book' in trades else pd.Series(0, index=trades.index)
    return dict(gross_annual=gross.mean() * 252, net_annual=net.mean() * 252,
                gross_sharpe=gross.mean() / gross.std() * np.sqrt(252), net_sharpe=net.mean() / net.std() * np.sqrt(252),
                max_drawdown=(curve - curve.cummax()).min(),
                turnover_per_rebalance=trades.loc[book.duplicated(), 'turnover'].mean(),
                annual_cost=(gross - net).sum() / len(net) * 252,
                names_per_side=trades[['longs', 'shorts']].median().mean(), rebalances=int((book == 0).sum()),
                days=len(net), every=every)
