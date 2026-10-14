import unittest

import numpy as np
import pandas as pd

from why.mom.hedge import run, sector_basket, trailing_beta
from why.mom.momentum import backtest, equal_weight, momentum, period_ends, periods
from why.mom.universe import WikiTable, membership

PAGE = """<html><body>
<table class="wikitable sortable"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th></tr>
<tr><td>AAA</td><td>Alpha</td><td>Industrials</td><td>1990-01-02</td></tr>
<tr><td>BBB</td><td>Beta</td><td>Financials</td><td>2026-01-05</td></tr></table>
<table class="wikitable"><tr><th>Date</th></tr><tr><td>ignored</td></tr></table></body></html>"""


def market(days=30, names=6):
    """信号 = 0..names-1，最高的两只每天涨 1%，其余不动。"""
    index = pd.bdate_range('2025-09-01', periods=days)
    signal = pd.DataFrame(np.tile(np.arange(float(names)), (days, 1)), index=index)
    returns = pd.DataFrame(0.0, index=index, columns=signal.columns)
    returns[[names - 2, names - 1]] = 0.01
    members = pd.DataFrame(True, index=index, columns=signal.columns)
    return index, signal, members, returns


class MomentumTests(unittest.TestCase):
    def test_momentum_skips_the_most_recent_month(self):
        index = pd.bdate_range('2024-01-01', periods=300)
        prices = pd.DataFrame({'a': np.arange(1.0, 301.0)}, index=index)
        signal = momentum(prices, window=252, skip=21)
        # 第 260 天（下标 259）：用第 239 天的价格除以第 8 天的价格
        self.assertAlmostEqual(signal['a'].iloc[259], prices['a'].iloc[259 - 21] / prices['a'].iloc[259 - 252] - 1)
        self.assertTrue(signal['a'].iloc[:251].isna().all())

    def test_momentum_needs_enough_history(self):
        index = pd.bdate_range('2024-01-01', periods=300)
        prices = pd.DataFrame({'a': np.arange(1.0, 301.0), 'b': np.arange(1.0, 301.0)}, index=index)
        prices.loc[prices.index[:100], 'b'] = np.nan  # b 只有 200 天价格，窗口内覆盖不足
        signal = momentum(prices, window=252, skip=21)
        self.assertFalse(np.isnan(signal['a'].iloc[-1]))
        self.assertTrue(np.isnan(signal['b'].iloc[-1]))

    def test_period_ends_take_the_last_trading_day(self):
        dates = pd.DatetimeIndex(['2025-09-29', '2025-09-30', '2025-10-01', '2025-10-31', '2025-11-03'])
        self.assertEqual(period_ends(dates), [1, 3, 4])

    def test_periods_trade_the_day_after_the_signal(self):
        index = pd.bdate_range('2025-09-01', periods=12)
        spans = [(day, span.start, span.stop) for day, span in periods(index, [0, 5, 10])]
        # 第 0 天出信号、第 1 天成交，收益从第 2 天算到下一个成交日（第 6 天）为止
        self.assertEqual(spans, [(0, 2, 7), (5, 7, 12)])

    def test_backtest_buys_the_strongest_names(self):
        index, signal, members, returns = market()
        gross, net, records = backtest(signal, members, returns, [0, 10, 20], cost=0.001, top=2)
        self.assertTrue(np.allclose(gross, 0.01))
        self.assertEqual(records['names'].iloc[0], [5, 4])   # nlargest: rank 1 is the strongest
        # 首次建仓买入 100%，之后持仓不变，不再交易
        self.assertAlmostEqual(net.iloc[0], 0.01 - 0.001)
        self.assertTrue(np.allclose(net.iloc[1:], 0.01))
        self.assertEqual(records['traded'].iloc[1:].tolist(), [0.0, 0.0])

    def test_backtest_only_holds_members(self):
        index, signal, members, returns = market()
        members[[4, 5]] = False  # 最强的两只不在成分股里
        _, _, records = backtest(signal, members, returns, [0], cost=0.0, top=2)
        self.assertEqual(records['names'].iloc[0], [3, 2])

    def test_equal_weight_holds_every_member(self):
        index, signal, members, returns = market()
        members[0] = False
        result = equal_weight(members, returns, [0, 10, 20])
        # 5 只成分股等权，其中 2 只每天涨 1%：建仓当天正好是 2/5 × 1%，之后涨的两只权重变大，收益略升
        self.assertAlmostEqual(result.iloc[0], 2 * 0.01 / 5)
        self.assertTrue((result >= 2 * 0.01 / 5 - 1e-12).all() and result.iloc[-1] > result.iloc[0])

    def test_membership_starts_at_the_date_added(self):
        dates = pd.to_datetime(['2025-12-31', '2026-01-31'])
        daily = pd.DataFrame(dict(date=dates.repeat(2), instrument_id=[1, 2, 1, 2],
                                  symbol=['AAA', 'BBB', 'AAA', 'BBB']))
        table = pd.DataFrame(dict(symbol=['AAA', 'BBB'], added=pd.to_datetime(['1990-01-02', '2026-01-05'])))
        like = pd.DataFrame(index=dates, columns=[1, 2], dtype=float)
        result = membership(daily, table, like)
        self.assertEqual(result.to_numpy().tolist(), [[True, False], [True, True]])

    def test_trailing_beta_uses_only_past_returns(self):
        index = pd.bdate_range('2025-01-01', periods=300)
        market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=index)
        returns = pd.DataFrame({'a': 2 * market, 'b': 0.5 * market})
        returns.iloc[250:] = 0.0  # 之后的收益改掉；用第 249 天估计时不应该受影响
        self.assertAlmostEqual(trailing_beta(returns, market, 249, {'a', 'b'}, window=252), 1.25, places=6)

    def test_sector_basket_matches_the_long_side_sectors(self):
        index = pd.bdate_range('2025-09-01', periods=3)
        members = pd.DataFrame(True, index=index, columns=[1, 2, 3, 4, 5])
        sectors = {1: 'Tech', 2: 'Tech', 3: 'Tech', 4: 'Banks', 5: 'Banks'}
        basket = sector_basket(members, sectors, 0, [1, 2, 4])  # 多头 2/3 科技、1/3 银行
        self.assertAlmostEqual(basket.sum(), 1.0)
        self.assertAlmostEqual(basket[[1, 2, 3]].sum(), 2 / 3)
        self.assertTrue(np.allclose(basket[[1, 2, 3]], 2 / 9) and np.allclose(basket[[4, 5]], 1 / 6))

    def test_vix_sleeve_adds_its_own_return(self):
        index, signal, members, returns = market(days=12)
        benchmarks = pd.DataFrame({'SPY': 0.0, 'QQQ': 0.0, 'VIXY': -0.02, 'VIXM': 0.0}, index=index)
        sectors = {name: 'Tech' for name in signal.columns}
        books, ratios = run(signal, members, returns, [0, 5], benchmarks, sectors, cost=0.0, top=2)
        # 5% 的 VIXY 仓位：第一天贡献 5% × -2%，之后仓位缩水，贡献略小
        self.assertAlmostEqual(books['vixy_5'].iloc[0] - books['none'].iloc[0], 0.05 * -0.02)
        self.assertTrue((books['vixy_5'] < books['none']).all())
        self.assertAlmostEqual(books['vixm_10'].iloc[0], books['none'].iloc[0])
        self.assertEqual(list(ratios.columns), ['signal_date', 'trade_date', 'names', 'beta_spy', 'beta_qqq'])

    def test_wiki_table_reads_the_first_table(self):
        parser = WikiTable()
        parser.feed(PAGE)
        self.assertEqual(parser.rows[0], ['Symbol', 'Security', 'GICS Sector', 'Date added'])
        self.assertEqual([row[0] for row in parser.rows[1:]], ['AAA', 'BBB'])


if __name__ == '__main__':
    unittest.main()
