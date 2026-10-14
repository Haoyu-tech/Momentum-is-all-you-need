"""用 WRDS 的 CRSP 标普 500 成分股文件做长期回测，解决幸存者偏差。

之前所有在我们自己数据上跑的策略都卡在两个问题上：

1. **只有 14 次调仓**（Databento 全市场数据最早到 2024-07，动量还要 252 天预热）；
2. **幸存者偏差**（成分股名单是从维基抓的**当前** 503 只，被剔除的那些一只都没有）。

CRSP 的「S&P 500 Index Constituents」一次解决两个，还顺带给了我们一直缺的市值：

- `MbrStartDt` / `MbrEndDt`：每段成员期的起止，**被剔除的股票也在里面**。
  实测 2015 年起出现过 756 只，其中 257 只在期间被剔除。
- 日频的 `DlyRet`（含分红的总收益）、`DlyCap`（市值）、`DlyClose`、`DlyVol`。
- 每天成分股数中位 505（双重股权类别会让它超过 500），日频记录 100% 落在成员期内。
- 覆盖 1928-08-24 至 2025-12-31。

**动量用累计总收益算**，和论文一致（论文脚注 4 说的是 cumulative returns from t−12 to t−2），
不用收盘价和复权因子——`DlyRet` 本身已经把拆股、分红都算进去了。

源文件不在仓库里，默认在 `why/data/raw/`。
第一次运行会抽取需要的列缓存成 parquet（约 940 万行，读 zip 要半分钟），之后直接读缓存。
"""
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WHY = Path(__file__).resolve().parent.parent
SOURCE = Path(os.environ.get('MOM_CRSP_MEMBERS_ZIP', str(WHY / 'data' / 'raw' / 'sp500_constituents.zip')))
COLUMNS = ['PERMNO', 'Ticker', 'SICCD', 'MbrStartDt', 'MbrEndDt', 'DlyCalDt', 'DlyRet', 'DlyCap', 'DlyClose']

# SIC 码 → Fama-French 12 行业。用 SIC 大类不行——「制造」一档就占指数的 37%，
# 行业集中度会失去意义。FF12 是学术标准，分得均匀得多。区间取自 French 的数据库说明。
FF12 = [
    ('非耐用消费', [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ('耐用消费', [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
                  (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ('制造', [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
              (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
              (3830, 3839), (3860, 3899)]),
    ('能源', [(1200, 1399), (2900, 2999)]),
    ('化工', [(2800, 2829), (2840, 2899)]),
    ('设备科技', [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ('电信', [(4800, 4899)]),
    ('公用事业', [(4900, 4949)]),
    ('批发零售', [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ('医疗', [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ('金融', [(6000, 6999)]),
]


def industry(code):
    """4 位 SIC 码 → Fama-French 12 行业；落不进任何一档的归「其他」。"""
    try:
        value = int(code)
    except (TypeError, ValueError):
        return '未知'
    for name, spans in FF12:
        if any(low <= value <= high for low, high in spans):
            return name
    return '其他'


def extract(source=SOURCE, cache=None, force=False):
    """从 zip 里抽出需要的列，缓存成 parquet。源文件 2.6 GB，缓存后小得多。"""
    cache = cache or WHY / 'data' / 'mom' / 'crsp_sp500.parquet'
    if cache.exists() and not force:
        return cache
    if not source.exists():
        raise FileNotFoundError(f'找不到 CRSP 源文件：{source}')
    archive = zipfile.ZipFile(source)
    pieces = []
    for chunk in pd.read_csv(archive.open(archive.namelist()[0]), usecols=COLUMNS,
                             chunksize=2_000_000, low_memory=False):
        chunk['date'] = pd.to_datetime(chunk.pop('DlyCalDt'), errors='coerce')
        for column in ('MbrStartDt', 'MbrEndDt'):
            chunk[column] = pd.to_datetime(chunk[column], errors='coerce')
        pieces.append(chunk.dropna(subset=['date']))
    frame = pd.concat(pieces, ignore_index=True)
    frame['sector'] = frame['SICCD'].map(industry)
    frame = frame.rename(columns={'PERMNO': 'permno', 'Ticker': 'symbol', 'DlyRet': 'ret',
                                  'DlyCap': 'cap', 'DlyClose': 'close'})
    frame = frame[['date', 'permno', 'symbol', 'sector', 'ret', 'cap', 'close',
                   'MbrStartDt', 'MbrEndDt']].sort_values(['date', 'permno'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    print(f'缓存 {len(frame):,} 行到 {cache}')
    return cache


STOCK_FILE = Path(os.environ.get('MOM_CRSP_STOCK_ZIP', str(WHY / 'data' / 'raw' / 'daily_stock.zip')))
STOCK_COLUMNS = ['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyCap', 'Ticker', 'SICCD']


def extract_full(source=STOCK_FILE, cache=None, members_cache=None, force=False):
    """抽出「曾经进过标普 500 的股票」的**完整**日频历史，用来修一个口径偏差。

    成分股文件只记录成员期内的数据，所以一只股票刚被纳入时，我们看不到它入指之前的价格，
    动量的 252 天窗口填不满——实测平均每月有 35 只成分股（7%）因此没有信号。而论文是用
    股票自身的完整收益历史算动量，不管它当时在不在指数里。

    这个偏差的方向是保守的（新纳入的往往正是涨得猛的高动量票，被漏掉了），但它是真实的偏离。

    源文件 6.78 GB、84 列、全部美股。这里只留曾进过标普 500 的约 1,955 个 PERMNO 和 6 列。
    """
    cache = cache or WHY / 'data' / 'mom' / 'crsp_full.parquet'
    if cache.exists() and not force:
        return cache
    if not source.exists():
        raise FileNotFoundError(f'找不到 CRSP 日频股票文件：{source}')
    keep = set(pd.read_parquet(members_cache or WHY / 'data' / 'mom' / 'crsp_sp500.parquet',
                               columns=['permno'])['permno'].unique())
    archive = zipfile.ZipFile(source)
    pieces, scanned = [], 0
    for chunk in pd.read_csv(archive.open(archive.namelist()[0]), usecols=STOCK_COLUMNS,
                             chunksize=2_000_000, low_memory=False):
        scanned += len(chunk)
        chunk = chunk[chunk['PERMNO'].isin(keep)]
        if len(chunk):
            chunk['date'] = pd.to_datetime(chunk.pop('DlyCalDt'), errors='coerce')
            pieces.append(chunk.dropna(subset=['date']))
        print(f'  已扫 {scanned:,} 行，留下 {sum(len(p) for p in pieces):,} 行', flush=True)
    frame = pd.concat(pieces, ignore_index=True)
    frame['sector'] = frame['SICCD'].map(industry)
    frame = frame.rename(columns={'PERMNO': 'permno', 'Ticker': 'symbol',
                                  'DlyRet': 'ret', 'DlyCap': 'cap'})
    frame = frame[['date', 'permno', 'symbol', 'sector', 'ret', 'cap']].sort_values(['date', 'permno'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    print(f'缓存 {len(frame):,} 行、{frame["permno"].nunique():,} 只股票到 {cache}')
    return cache


def load(cache=None, start=None, end=None):
    """返回逐日的宽表：收益、市值、成员标记，外加 permno → 行业的映射。

    成员标记直接用 MbrStartDt/MbrEndDt——CRSP 的日频记录本来就只在成员期内，
    所以只要这一天这只股票有记录，它当天就是成分股。
    """
    frame = pd.read_parquet(cache or WHY / 'data' / 'mom' / 'crsp_sp500.parquet')
    if start:
        frame = frame[frame['date'] >= start]
    if end:
        frame = frame[frame['date'] < end]
    returns = frame.pivot_table(index='date', columns='permno', values='ret', aggfunc='last')
    caps = frame.pivot_table(index='date', columns='permno', values='cap', aggfunc='last')
    members = returns.notna()
    sectors = dict(frame.drop_duplicates('permno', keep='last')[['permno', 'sector']].to_numpy())
    tickers = dict(frame.drop_duplicates('permno', keep='last')[['permno', 'symbol']].to_numpy())
    return returns, caps, members, sectors, tickers


# 单边交易成本（bp）按年代分档。1975-05-01 之前 NYSE 是固定佣金制，全程 10bp 不可能。
# 这组数字是量级估计，不是精确值——真实成本随股票规模、订单大小差别很大。
COSTS = [(2000, 10.0), (1990, 25.0), (1975, 50.0), (0, 100.0)]


def cost_at(day, ladder=COSTS):
    """某一天的单边成本（小数）。"""
    return next(bps for year, bps in ladder if day.year >= year) / 1e4


def full(cache=None, start=None, end=None):
    """完整历史版：动量用股票**自身**的全部收益算，成员标记另外来自成分股文件。

    成分股文件只记成员期内的数据，所以刚纳入的股票动量窗口填不满——实测平均每月有
    35 只（7%）因此没有信号。这里用 `crsp_full.parquet`（曾进过标普 500 的股票的完整
    日频历史，比成员期内多 95% 的行）算价格和动量，再用成分股文件决定哪天能买。
    """
    whole = pd.read_parquet(cache or WHY / 'data' / 'mom' / 'crsp_full.parquet')
    # 行业取自成分股文件，不取完整历史文件：退市股票最后一条记录的 SICCD 是 0，
    # 而 70% 的股票早已退市，按 keep='last' 会把它们全归进「其他」（实测 1960 年高达 85%）。
    # 成分股文件里的行业是**在指数内那段时间**的，正是我们关心的。
    inside = pd.read_parquet(WHY / 'data' / 'mom' / 'crsp_sp500.parquet',
                             columns=['date', 'permno', 'sector', 'symbol'])
    if start:
        whole, inside = whole[whole['date'] >= start], inside[inside['date'] >= start]
    if end:
        whole, inside = whole[whole['date'] < end], inside[inside['date'] < end]
    returns = whole.pivot_table(index='date', columns='permno', values='ret', aggfunc='last')
    caps = whole.pivot_table(index='date', columns='permno', values='cap', aggfunc='last')
    members = (inside.assign(member=True).pivot_table(index='date', columns='permno', values='member',
                                                      aggfunc='last')
               .reindex(index=returns.index, columns=returns.columns).fillna(False).astype(bool))
    recent = inside.drop_duplicates('permno', keep='last')
    sectors = dict(recent[['permno', 'sector']].to_numpy())
    tickers = dict(recent[['permno', 'symbol']].to_numpy())
    return returns, caps, members, sectors, tickers


def prices(returns):
    """把总收益连乘成「后复权价」，起点 1。动量就在这个序列上算。

    CRSP 的 DlyRet 已经含分红和拆股调整，所以连乘出来的就是总收益指数。
    股票不在成分股里的那些天是 NaN，连乘前先填 0 会把不同成员期接在一起——
    这没关系，因为动量信号只在它是成分股的日子才会被用到，而且窗口里要求 80% 覆盖。
    """
    return (1 + returns.fillna(0)).cumprod().where(returns.notna())


def main():
    parser = argparse.ArgumentParser(description='Extract and inspect the CRSP S&P 500 constituent file')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--cache', type=Path, default=WHY / 'data' / 'mom' / 'crsp_sp500.parquet')
    parser.add_argument('--force', action='store_true', help='重新抽取，忽略已有缓存')
    parser.add_argument('--start', default='1957-03-01', help='标普 500 扩到 500 只是 1957 年 3 月')
    args = parser.parse_args()

    path = extract(args.source, args.cache, args.force)
    frame = pd.read_parquet(path, columns=['date', 'permno', 'symbol', 'sector', 'MbrEndDt'])
    print(f'全部 {len(frame):,} 行，{frame["date"].min().date()} 至 {frame["date"].max().date()}，'
          f'{frame["permno"].nunique():,} 只股票')
    counts = frame.groupby('date')['permno'].nunique()
    print(f'每天成分股数：中位 {counts.median():.0f}  最少 {counts.min()}  最多 {counts.max()}')
    recent = frame[frame['date'] >= args.start]
    exits = recent[recent['MbrEndDt'] < recent['date'].max()]
    print(f'{args.start} 起：{recent["permno"].nunique():,} 只股票出现过，'
          f'其中 {exits["permno"].nunique():,} 只在期间被剔除')
    print('\n行业分布（按股票数，最近一个交易日）：')
    last = frame[frame['date'] == frame['date'].max()]
    print((last['sector'].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == '__main__':
    main()
