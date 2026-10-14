"""标普 500 成分股名单：从 Wikipedia 抓当天的名单并缓存，再对到我们数据里的 instrument_id。

只有当天的名单，没有历史成员文件，所以：
- 名单里有「加入日期」，早于加入日期的日子不算成分股，避免把还没进指数的股票算进去；
- 但**已经被剔除的公司完全不在名单里**，这是幸存者偏差，见 README。
"""
from html.parser import HTMLParser

import pandas as pd
import requests

URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
COLUMNS = {'Symbol': 'symbol', 'Security': 'company', 'GICS Sector': 'sector', 'Date added': 'added'}
EXPECTED = (490, 515)  # 名单长度的合理范围，超出就认为页面结构变了


class WikiTable(HTMLParser):
    """取页面里第一个 wikitable 的所有单元格。"""

    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.depth, self.done = [], None, None, 0, False

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and not self.done:
            self.depth += 1 if self.depth else int('wikitable' in dict(attrs).get('class', ''))
        if not self.depth:
            return
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []

    def handle_endtag(self, tag):
        if tag == 'table' and self.depth:
            self.depth -= 1
            self.done = not self.depth
        if not self.depth:
            return
        if tag in ('td', 'th') and self.cell is not None and self.row is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def fetch(url=URL):
    """抓成分股表，返回 symbol、company、sector、added 四列。"""
    response = requests.get(url, headers={'User-Agent': 'hk-quant research contact local-user'}, timeout=60)
    response.raise_for_status()
    parser = WikiTable()
    parser.feed(response.text)
    header, *rows = parser.rows
    table = pd.DataFrame(rows, columns=header)[list(COLUMNS)].rename(columns=COLUMNS)
    table['symbol'] = table['symbol'].str.strip()
    table['added'] = pd.to_datetime(table['added'], errors='coerce')
    if not EXPECTED[0] <= len(table) <= EXPECTED[1] or table['symbol'].duplicated().any():
        raise ValueError(f'Unexpected S&P 500 table: {len(table)} rows')
    return table


def load(path, refresh=False):
    """读缓存的名单；没有缓存或者 refresh 时重新抓取并写入缓存。"""
    if refresh or not path.exists():
        table = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'fetched {len(table)} S&P 500 symbols to {path}', flush=True)
        return table
    table = pd.read_csv(path, parse_dates=['added'])
    print(f'loaded {len(table)} S&P 500 symbols from {path}', flush=True)
    return table


def membership(daily, table, like):
    """逐日的成分股标记（行 = 交易日，列 = instrument_id）：当天的代码在名单里，且已经过了加入日期。

    代码用的是当天的证券定义，所以改代码、代码被别的公司复用都能对上。
    """
    added = table.set_index('symbol')['added']
    joined = daily['symbol'].map(added)
    # 名单里没有加入日期的（少数几个）只按代码算成分股
    inside = daily['symbol'].isin(added.index) & (joined.isna() | (daily['date'] >= joined))
    frame = daily.loc[inside, ['date', 'instrument_id']].assign(member=True)
    return frame.pivot(index='date', columns='instrument_id', values='member').reindex(
        index=like.index, columns=like.columns).fillna(False).astype(bool)
