"""因子评估共用的工具：读数据、截面 Rank IC、分组多空收益、换手、持有期收益。

所有宽表都是 行 = 交易日、列 = instrument_id；股票池 universe 是同形状的布尔表。
"""
import numpy as np
import pandas as pd


COLUMNS = ['date', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'ret', 'adj_close_pit', 'in_universe']
MIN_STOCKS = 100  # 当天股票池内有效股票少于这个数，不算 IC 和多空收益
QUANTILE = 0.05  # 默认做多前 5%、做空后 5%
COST_BPS = 10.0  # 默认单边交易成本，单位 bp
# sklearn HistGradientBoostingRegressor 的参数，没有调过；evaluate 和 low_frequency 共用
GBDT = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=500, l2_regularization=1.0,
            early_stopping=False, random_state=0)


def load(path, columns=COLUMNS):
    """读 us_daily.parquet，返回原表、股票池宽表和收益（已复权的 ret）宽表。"""
    daily = pd.read_parquet(path, columns=columns)
    universe = daily.pivot(index='date', columns='instrument_id', values='in_universe').fillna(False).astype(bool)
    returns = daily.pivot(index='date', columns='instrument_id', values='ret')
    return daily, universe, returns


def holding_return(returns, days):
    """对齐到信号日 t：t+1 日收盘买入、持有到 t+1+days 日收盘的收益。days = 1 就是 Qlib 默认标签。"""
    return np.expm1(np.log1p(returns).rolling(days, min_periods=days).sum().shift(-(days + 1)))


def last_training_day(dates, split, days):
    """前半段里标签不跨过切分日的最后一个信号日：持有 days 天的标签要用到 t+1 至 t+1+days 日的收益，
    所以最后 days + 1 个交易日的标签会用到后半段，挑因子、定方向、训练模型都只能用到这一天。"""
    return dates[dates < split][-(days + 2)]


def to_wide(values, rows, cols, like):
    """把按 (rows, cols) 位置排列的一维结果放回与 like 同形状的宽表。"""
    wide = np.full(like.shape, np.nan)
    wide[rows, cols] = values
    return pd.DataFrame(wide, index=like.index, columns=like.columns)


def row_corr_frame(a, b):
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


def rank_ic(signal, label, universe):
    """每天在股票池内、两边都有值的股票上算 Spearman 相关（精确版，每次重新排名）。"""
    valid = universe & signal.notna() & label.notna()
    ic = row_corr_frame(signal.where(valid).rank(axis=1), label.where(valid).rank(axis=1))
    return ic.where(valid.sum(axis=1) >= MIN_STOCKS)


def ranked(frame, universe):
    """股票池内每天的截面名次（float32 数组），供 row_corr 反复使用。"""
    return frame.where(universe).rank(axis=1).to_numpy(dtype=np.float32)


def row_corr(a, b):
    """逐行相关系数，只用两边都有值的格子（快速版）。

    输入是事先在各自有效集合上排好的名次，所以结果近似于交集上的 Spearman；
    实测与 rank_ic 的差异最大 0.00006，可以忽略。
    """
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = np.where(valid, a, 0.0).astype(np.float64), np.where(valid, b, 0.0).astype(np.float64)
    n = valid.sum(axis=1)
    sa, sb = a.sum(axis=1), b.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = (a * b).sum(axis=1) - sa * sb / n
        var = ((a * a).sum(axis=1) - sa ** 2 / n) * ((b * b).sum(axis=1) - sb ** 2 / n)
        return np.where(n >= MIN_STOCKS, cov / np.sqrt(var), np.nan)


def long_short(signal, label, universe, quantile=QUANTILE):
    """每天做多信号最高的 quantile、做空最低的 quantile，等权，返回标签收益之差（不扣成本）。"""
    valid = universe & signal.notna() & label.notna()
    percentile = signal.where(valid).rank(axis=1, pct=True)
    spread = label.where(percentile > 1 - quantile).mean(axis=1) - label.where(percentile <= quantile).mean(axis=1)
    return spread.where(valid.sum(axis=1) >= MIN_STOCKS)


def turnover(signal, universe, quantile=QUANTILE):
    """每天调仓时，多空两边各自换掉的股票比例的平均值；年化成本约为 252 × 4 × 换手 × 单边成本。"""
    percentile = signal.where(universe).rank(axis=1, pct=True)
    legs = []
    for held in [percentile > 1 - quantile, percentile <= quantile]:
        kept = (held & held.shift(1, fill_value=False)).sum(axis=1)
        legs.append((1 - kept / held.sum(axis=1)).where(held.sum(axis=1) > 0))
    return (legs[0] + legs[1]) / 2


def summarize(ic, spread):
    ic, spread = ic.dropna(), spread.dropna()
    return dict(rank_ic=ic.mean(), rank_icir=ic.mean() / ic.std(), t_stat=ic.mean() / ic.std() * np.sqrt(len(ic)),
                win_rate=(ic > 0).mean(), days=len(ic), long_short_annual=spread.mean() * 252,
                long_short_sharpe=spread.mean() / spread.std() * np.sqrt(252))


def halves(series, split):
    """全期、前半段（切分日之前）、后半段（切分日及之后）。"""
    return {'full': series, 'first_half': series[series.index < split], 'second_half': series[series.index >= split]}
