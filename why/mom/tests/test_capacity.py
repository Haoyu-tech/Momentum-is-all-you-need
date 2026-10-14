import unittest

import numpy as np
import pandas as pd

from why.mom.capacity import ADV_DAYS, advance, capacity, changes, deflate


def days(count, start='2020-01-01'):
    return pd.bdate_range(start, periods=count)


class ChangeTests(unittest.TestCase):
    """|Δw| 决定每只票要成交多少钱，算错了容量就全错。"""

    def test_first_rebalance_is_a_full_build(self):
        books = [(days(1)[0], pd.Series({1: 0.5, 2: 0.5}))]
        _, delta = changes(books)[0]
        self.assertAlmostEqual(delta.sum(), 1.0)

    def test_unchanged_holdings_trade_nothing(self):
        weights = pd.Series({1: 0.5, 2: 0.5})
        books = [(d, weights) for d in days(2)]
        self.assertEqual(len(changes(books)[1][1]), 0)

    def test_swapping_one_name_trades_both_legs(self):
        books = [(days(2)[0], pd.Series({1: 0.5, 2: 0.5})),
                 (days(2)[1], pd.Series({1: 0.5, 3: 0.5}))]
        _, delta = changes(books)[1]
        self.assertEqual(set(delta.index), {2, 3})
        self.assertAlmostEqual(delta.sum(), 1.0)

    def test_weights_are_used_not_name_counts(self):
        """市值加权下小票权重小，|Δw| 必须跟着小——这正是之前写死 1/N 的那个 bug。"""
        books = [(days(2)[0], pd.Series({1: 0.9, 2: 0.1})),
                 (days(2)[1], pd.Series({1: 0.9, 3: 0.1}))]
        _, delta = changes(books)[1]
        self.assertAlmostEqual(delta.sum(), 0.2)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.dates = days(2)
        self.books = [(self.dates[0], pd.Series({1: 0.5, 2: 0.5}))]
        self.leverage = pd.Series(1.0, index=self.dates)
        # 1 号票每天成交 100 万，2 号票只有 10 万——2 号是卡脖子的那只
        self.volume = pd.DataFrame({1: [1e6, 1e6], 2: [1e5, 1e5]}, index=self.dates)

    def test_tight_constraint_is_set_by_the_least_liquid_name(self):
        frame = capacity(self.books, self.leverage, self.volume, participation=0.1, spread=1)
        # 0.1 × 10 万 / 0.5 = 2 万
        self.assertAlmostEqual(frame['紧约束容量'].iloc[0], 2e4)
        self.assertEqual(frame['最紧的票'].iloc[0], 2)

    def test_total_capacity_pools_liquidity_across_names(self):
        frame = capacity(self.books, self.leverage, self.volume, participation=0.1, spread=1)
        # 0.1 × (100 万 + 10 万) / 1.0 = 11 万，比紧约束宽得多
        self.assertAlmostEqual(frame['总量容量'].iloc[0], 1.1e5)
        self.assertGreater(frame['总量容量'].iloc[0], frame['紧约束容量'].iloc[0])

    def test_leverage_shrinks_capacity_proportionally(self):
        geared = capacity(self.books, pd.Series(2.0, index=self.dates), self.volume,
                          participation=0.1, spread=1)
        flat = capacity(self.books, self.leverage, self.volume, participation=0.1, spread=1)
        self.assertAlmostEqual(geared['紧约束容量'].iloc[0] * 2, flat['紧约束容量'].iloc[0])

    def test_spreading_over_more_days_scales_capacity(self):
        wide = capacity(self.books, self.leverage, self.volume, participation=0.1, spread=5)
        flat = capacity(self.books, self.leverage, self.volume, participation=0.1, spread=1)
        self.assertAlmostEqual(wide['紧约束容量'].iloc[0], flat['紧约束容量'].iloc[0] * 5)


class AdvanceTests(unittest.TestCase):
    def test_adv_uses_only_past_days(self):
        """ADV 必须后移一天，否则调仓当天就用上了当天的成交量——那是未来函数。"""
        index = days(ADV_DAYS * 3)
        frame = pd.DataFrame({'date': index, 'permno': 1,
                              'dollars': np.arange(len(index), dtype=float) + 1.0})
        wide = frame.pivot_table(index='date', columns='permno', values='dollars', aggfunc='last')
        rolled = wide.rolling(ADV_DAYS, min_periods=ADV_DAYS // 2).median().shift(1)
        self.assertTrue(np.isnan(rolled.iloc[0, 0]))
        # 第 t 天的值只能由 t 之前的数据决定
        self.assertLess(rolled.iloc[ADV_DAYS, 0], wide.iloc[ADV_DAYS, 0])

    def test_advance_shifts_and_matches_manual_median(self):
        import tempfile
        from pathlib import Path
        index = days(ADV_DAYS * 2)
        frame = pd.DataFrame({'date': index, 'permno': 7,
                              'dollars': np.arange(1.0, len(index) + 1.0)})
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'volume.parquet'
            frame.to_parquet(cache, index=False)
            result = advance(cache)
        expected = frame['dollars'].rolling(ADV_DAYS).median().shift(1).iloc[-1]
        self.assertAlmostEqual(result[7].iloc[-1], expected)


class DeflateTests(unittest.TestCase):
    def test_older_dollars_are_worth_more_after_deflating(self):
        self.assertGreater(deflate(1960), deflate(2000))
        self.assertGreater(deflate(2000), 1.0)

    def test_base_year_is_unchanged(self):
        self.assertAlmostEqual(deflate(2025), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
