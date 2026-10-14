import unittest

import numpy as np
import pandas as pd

from why.mom.breadth import pick, run, summarise


def scores(values):
    """instrument_id 从 0 开始的动量分数。"""
    return pd.Series(values, index=range(len(values)), dtype=float)


SECTORS = {i: ('科技' if i < 10 else '金融' if i < 20 else '能源') for i in range(30)}


class PickTests(unittest.TestCase):
    def setUp(self):
        self.scores = scores(np.arange(30, dtype=float))     # 分数就是编号，排序一目了然

    def test_takes_the_top_and_bottom_fraction(self):
        longs, shorts = pick(self.scores, SECTORS, fraction=0.1)
        self.assertEqual(sorted(longs), [27, 28, 29])        # 30 × 10% = 3 只
        self.assertEqual(sorted(shorts), [0, 1, 2])

    def test_always_picks_at_least_one(self):
        longs, shorts = pick(scores([1.0, 2.0]), SECTORS, fraction=0.1)
        self.assertEqual(len(longs), 1)
        self.assertEqual(len(shorts), 1)

    def test_sector_neutral_picks_inside_each_sector(self):
        longs, shorts = pick(self.scores, SECTORS, fraction=0.1, neutral=True)
        # 每个行业 10 只，各取 1 只：科技 0-9、金融 10-19、能源 20-29
        self.assertEqual(sorted(longs), [9, 19, 29])
        self.assertEqual(sorted(shorts), [0, 10, 20])

    def test_sector_neutral_covers_every_sector(self):
        longs, _ = pick(self.scores, SECTORS, fraction=0.1, neutral=True)
        self.assertEqual({SECTORS[i] for i in longs}, {'科技', '金融', '能源'})

    def test_sector_neutral_skips_sectors_too_small_for_one_slot(self):
        # 行业只有 5 只，但 1/0.1 = 10，取不出一档，整个行业跳过
        tiny = {i: ('小' if i < 5 else '大') for i in range(30)}
        longs, _ = pick(self.scores, tiny, fraction=0.1, neutral=True)
        self.assertEqual({tiny[i] for i in longs}, {'大'})


class RunTests(unittest.TestCase):
    """用一个每天涨跌固定的小盘面，验证成交、持仓和多空相减。"""

    def setUp(self):
        self.dates = pd.bdate_range('2020-01-01', periods=12)
        self.returns = pd.DataFrame(0.0, index=self.dates, columns=range(30))
        self.returns.loc[:, 27:29] = 0.01                 # 高分那三只每天 +1%
        self.returns.loc[:, 0:2] = -0.01                  # 低分那三只每天 −1%
        self.signal = pd.DataFrame(np.tile(np.arange(30, dtype=float), (12, 1)),
                                   index=self.dates, columns=range(30))
        self.members = pd.DataFrame(True, index=self.dates, columns=range(30))
        self.days = [3, 7]

    def test_long_only_holds_the_top_names(self):
        _, net, records = run(self.signal, self.members, self.returns, self.days, 0.0,
                              SECTORS, fraction=0.1, short=False)
        self.assertEqual(list(records['多头只数']), [3, 3])
        self.assertEqual(list(records['空头只数']), [0, 0])
        self.assertTrue((net > 0).all())                  # 只买涨的，每天都是正的

    def test_long_short_subtracts_the_short_leg(self):
        _, long_only, _ = run(self.signal, self.members, self.returns, self.days, 0.0,
                              SECTORS, fraction=0.1, short=False)
        _, both, _ = run(self.signal, self.members, self.returns, self.days, 0.0,
                         SECTORS, fraction=0.1, short=True)
        # 多头 +1%、空头 −1%，多空应该是 +2%
        self.assertAlmostEqual(both.iloc[-1], 0.02, places=10)
        self.assertAlmostEqual(long_only.iloc[-1], 0.01, places=10)

    def test_turnover_counts_both_legs(self):
        _, _, long_only = run(self.signal, self.members, self.returns, self.days, 0.0,
                              SECTORS, fraction=0.1, short=False)
        _, _, both = run(self.signal, self.members, self.returns, self.days, 0.0,
                         SECTORS, fraction=0.1, short=True)
        self.assertAlmostEqual(both['traded'].iloc[0], long_only['traded'].iloc[0] * 2)

    def test_cost_is_only_charged_when_holdings_change(self):
        _, free, records = run(self.signal, self.members, self.returns, self.days, 0.0,
                               SECTORS, fraction=0.1, short=False)
        _, paid, _ = run(self.signal, self.members, self.returns, self.days, 1e-3,
                         SECTORS, fraction=0.1, short=False)
        gap = (free - paid).round(12)
        # 信号不变，第二次调仓持仓一样、换手为 0，所以只有首次建仓扣了钱
        self.assertEqual(int((gap > 0).sum()), 1)
        self.assertAlmostEqual(records['traded'].iloc[0], 1.0)
        self.assertAlmostEqual(records['traded'].iloc[1], 0.0)

    def test_cost_is_charged_once_per_period_when_names_rotate(self):
        rotating = self.signal.copy()
        rotating.iloc[7:] = rotating.iloc[7:].to_numpy()[:, ::-1]   # 第二期把排序整个翻转
        _, free, records = run(rotating, self.members, self.returns, self.days, 0.0,
                               SECTORS, fraction=0.1, short=False)
        _, paid, _ = run(rotating, self.members, self.returns, self.days, 1e-3,
                         SECTORS, fraction=0.1, short=False)
        gap = (free - paid).round(12)
        self.assertEqual(int((gap > 0).sum()), len(records))

    def test_top_overrides_the_fraction(self):
        _, _, records = run(self.signal, self.members, self.returns, self.days, 0.0,
                            SECTORS, top=5, short=False)
        self.assertEqual(list(records['多头只数']), [5, 5])

    def test_records_sector_concentration(self):
        _, _, records = run(self.signal, self.members, self.returns, self.days, 0.0,
                            SECTORS, fraction=0.1, short=False)
        # 前 3 名（27、28、29）全在能源
        self.assertEqual(list(records['多头行业数']), [1, 1])
        self.assertAlmostEqual(records['多头最大行业占比'].iloc[0], 1.0)

    def test_sector_neutral_spreads_across_sectors(self):
        _, _, records = run(self.signal, self.members, self.returns, self.days, 0.0,
                            SECTORS, fraction=0.1, short=False, neutral=True)
        self.assertEqual(list(records['多头行业数']), [3, 3])
        self.assertAlmostEqual(records['多头最大行业占比'].iloc[0], 1 / 3)


class SummariseTests(unittest.TestCase):
    def test_metrics_of_a_steady_series(self):
        net = pd.Series(0.001, index=pd.bdate_range('2020-01-01', periods=252))
        records = pd.DataFrame([dict(多头只数=10, 空头只数=0, 多头行业数=3,
                                     多头最大行业占比=0.5, traded=0.4)])
        result = summarise(net, records)
        self.assertAlmostEqual(result['年化收益'], 0.252)
        self.assertAlmostEqual(result['年化波动'], 0.0)
        self.assertEqual(result['夏普'], 0.0)              # 零波动时不除零（浮点残差也不行）
        self.assertEqual(result['t值'], 0.0)
        self.assertAlmostEqual(result['最大回撤'], 0.0)

    def test_drawdown_is_measured_on_compounded_value(self):
        net = pd.Series([0.5, -0.5], index=pd.bdate_range('2020-01-01', periods=2))
        self.assertAlmostEqual(summarise(net, pd.DataFrame())['最大回撤'], -0.5)

    def test_benchmark_gives_the_excess(self):
        index = pd.bdate_range('2020-01-01', periods=50)
        net = pd.Series(0.002, index=index)
        base = pd.Series(0.001, index=index)
        result = summarise(net, pd.DataFrame(), base)
        self.assertAlmostEqual(result['超额年化'], 0.001 * 252)


if __name__ == '__main__':
    unittest.main()
