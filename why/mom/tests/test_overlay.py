import unittest

import numpy as np
import pandas as pd

from why.mom.overlay import (apply_hedge, apply_scale, monthly, realized_vol, summarise,
                             trailing_beta)


def series(values, start='2020-01-01'):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


class BetaTests(unittest.TestCase):
    """滚动 beta 必须只用历史数据——这是这个模块唯一可能出未来函数的地方。"""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.market = series(rng.normal(0, 0.01, 600))
        self.strategy = self.market * 1.5 + series(rng.normal(0, 0.002, 600))

    def test_recovers_a_known_beta(self):
        beta = trailing_beta(self.strategy, self.market, window=252).dropna()
        self.assertAlmostEqual(beta.median(), 1.5, places=1)

    def test_no_lookahead_future_data_cannot_move_an_earlier_beta(self):
        beta = trailing_beta(self.strategy, self.market, window=252)
        tampered = self.strategy.copy()
        tampered.iloc[-50:] = 9.99
        after = trailing_beta(tampered, self.market, window=252)
        early = beta.index < tampered.index[-50]
        pd.testing.assert_series_equal(beta[early], after[early])

    def test_first_window_has_no_estimate(self):
        beta = trailing_beta(self.strategy, self.market, window=252)
        self.assertTrue(beta.iloc[:252].isna().all())   # 252 天窗口 + 后移一天


class VolatilityTests(unittest.TestCase):
    def test_constant_returns_give_the_textbook_volatility(self):
        flat = series([0.01] * 300)
        self.assertAlmostEqual(realized_vol(flat, days=126).dropna().iloc[0], 0.01 * np.sqrt(252), places=10)

    def test_no_lookahead(self):
        rng = np.random.default_rng(1)
        base = series(rng.normal(0, 0.01, 400))
        sigma = realized_vol(base, days=126)
        tampered = base.copy()
        tampered.iloc[-30:] = 5.0
        after = realized_vol(tampered, days=126)
        early = sigma.index < tampered.index[-30]
        pd.testing.assert_series_equal(sigma[early], after[early])


class MonthlyTests(unittest.TestCase):
    def test_value_is_held_flat_between_marks(self):
        raw = series(np.arange(40, dtype=float))
        marks = [raw.index[0], raw.index[20]]
        held = monthly(raw, marks)
        self.assertEqual(held.iloc[:20].nunique(), 1)      # 第一段只有一个值
        self.assertEqual(held.iloc[20:].nunique(), 1)
        self.assertNotEqual(held.iloc[0], held.iloc[20])


class HedgeTests(unittest.TestCase):
    def setUp(self):
        self.strategy = series([0.02] * 10)
        self.market = series([0.01] * 10)

    def test_full_hedge_removes_the_market_leg(self):
        ratio = pd.Series(1.0, index=self.strategy.index)
        net = apply_hedge(self.strategy, self.market, ratio, cost_bps=0.0)
        self.assertTrue(np.allclose(net, 0.01))            # 2% − 1×1%

    def test_zero_ratio_leaves_the_strategy_alone(self):
        ratio = pd.Series(0.0, index=self.strategy.index)
        net = apply_hedge(self.strategy, self.market, ratio, cost_bps=0.0)
        pd.testing.assert_series_equal(net, self.strategy, check_names=False)

    def test_cost_is_charged_when_the_ratio_changes(self):
        steady = pd.Series(0.5, index=self.strategy.index)
        moving = steady.copy()
        moving.iloc[5:] = 0.9
        flat = apply_hedge(self.strategy, self.market, steady, cost_bps=10.0)
        shifted = apply_hedge(self.strategy, self.market, moving, cost_bps=10.0)
        # 第 5 天比例从 0.5 变成 0.9，要交易 0.4 的名义金额
        self.assertAlmostEqual(flat.iloc[5] - shifted.iloc[5], 0.4 * 1e-3 + 0.4 * 0.01, places=10)

    def test_initial_position_is_charged(self):
        ratio = pd.Series(0.5, index=self.strategy.index)
        free = apply_hedge(self.strategy, self.market, ratio, cost_bps=0.0)
        paid = apply_hedge(self.strategy, self.market, ratio, cost_bps=10.0)
        self.assertAlmostEqual(free.iloc[0] - paid.iloc[0], 0.5 * 1e-3)
        self.assertAlmostEqual(free.iloc[1], paid.iloc[1])   # 之后不变就不再扣


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.strategy = series([0.02] * 10)

    def test_weight_multiplies_the_return(self):
        weight = pd.Series(0.5, index=self.strategy.index)
        net = apply_scale(self.strategy, weight, cost_bps=0.0)
        self.assertAlmostEqual(net.iloc[-1], 0.01)

    def test_scaling_does_not_change_the_sharpe(self):
        rng = np.random.default_rng(2)
        noisy = series(rng.normal(0.0004, 0.01, 500))
        low = apply_scale(noisy, pd.Series(0.5, index=noisy.index), cost_bps=0.0)
        high = apply_scale(noisy, pd.Series(2.0, index=noisy.index), cost_bps=0.0)
        self.assertAlmostEqual(summarise(low, 'a')['夏普'], summarise(high, 'b')['夏普'], places=10)

    def test_cost_follows_the_change_in_leverage(self):
        weight = pd.Series(1.0, index=self.strategy.index)
        weight.iloc[5:] = 1.3
        free = apply_scale(self.strategy, weight, cost_bps=0.0)
        paid = apply_scale(self.strategy, weight, cost_bps=10.0)
        self.assertAlmostEqual(free.iloc[5] - paid.iloc[5], 0.3 * 1e-3, places=12)


class SummariseTests(unittest.TestCase):
    def test_zero_variance_does_not_divide(self):
        result = summarise(series([0.001] * 100), 'flat')
        self.assertEqual(result['夏普'], 0.0)
        self.assertEqual(result['t值'], 0.0)

    def test_terminal_value_compounds(self):
        result = summarise(series([0.5, -0.5]), 'path')
        self.assertAlmostEqual(result['终值'], 0.75)        # 1.5 × 0.5
        self.assertAlmostEqual(result['最大回撤'], -0.5)


if __name__ == '__main__':
    unittest.main()
