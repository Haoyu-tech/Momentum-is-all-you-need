import unittest

import numpy as np
import pandas as pd

from why.mom.barroso import (MONTH, alpha, costs, daily_weights, drawdown, manage, moments,
                             monthly_variance, parse, predictability, realized, schedules)

HEADER = """This file was created using the 202608 CRSP database.
  Some blurb about construction.

,Mom
"""


def daily_series(values, start='2020-01-01'):
    """按交易日排列的日收益。"""
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, name='wml')


class ParseTests(unittest.TestCase):
    """French 的文件前面有说明文字，后面可能跟一张年度表，只能取第一张。"""

    def test_skips_the_preamble_and_converts_percent(self):
        text = HEADER + '20200102,    1.50\n20200103,   -0.25\n'
        frame = parse(text, monthly=False)
        self.assertEqual(list(frame.index), [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')])
        self.assertAlmostEqual(frame.iloc[0, 0], 0.015)
        self.assertAlmostEqual(frame.iloc[1, 0], -0.0025)

    def test_stops_before_the_annual_table(self):
        text = HEADER + '20200102,    1.00\n\n  Annual Factors:\n\n,Mom\n2020,   10.00\n'
        self.assertEqual(len(parse(text, monthly=False)), 1)

    def test_missing_codes_become_nan(self):
        text = HEADER + '20200102,  -99.99\n20200103,    1.00\n'
        self.assertTrue(np.isnan(parse(text, monthly=False).iloc[0, 0]))

    def test_monthly_index_is_periods(self):
        text = HEADER + '202001,    1.00\n202002,    2.00\n'
        frame = parse(text, monthly=True)
        self.assertEqual(frame.index.freqstr, 'M')
        self.assertEqual(str(frame.index[0]), '2020-01')


class RealizedTests(unittest.TestCase):
    """这是整个策略唯一可能出未来函数的地方，必须钉死。"""

    def setUp(self):
        # 先 252 天每天 +1%（留足 126 天的预热），之后每天 +3%：两段方差差 9 倍，边界好验证
        self.daily = daily_series([0.01] * 252 + [0.03] * 126)
        self.months = self.daily.index.to_period('M').unique()

    def test_variance_uses_only_data_through_the_previous_month(self):
        sigma = realized(self.daily, self.months, days=126)
        switch = self.daily.index[252].to_period('M')          # 收益跳到 3% 的那个月
        # 跳变当月用的还是上月末的估计，里面一个 3% 都没有
        self.assertAlmostEqual(sigma[switch], np.sqrt(0.01 ** 2 * MONTH), places=10)

    def test_shifting_the_whole_series_later_shifts_the_estimate(self):
        sigma = realized(self.daily, self.months, days=126)
        self.assertTrue(sigma.notna().any())
        self.assertTrue(sigma.dropna().index[0] > self.months[0])   # 前面的月份没有足够历史

    def test_no_lookahead_future_returns_cannot_change_a_past_estimate(self):
        sigma = realized(self.daily, self.months, days=126)
        tampered = self.daily.copy()
        tampered.iloc[-30:] = 9.99                              # 只改最后 30 天
        after = realized(tampered, self.months, days=126)
        last = tampered.index[-30].to_period('M')
        early = sigma.index < last
        pd.testing.assert_series_equal(sigma[early], after[early])

    def test_constant_returns_give_the_textbook_variance(self):
        flat = daily_series([0.02] * 300)
        sigma = realized(flat, flat.index.to_period('M').unique(), days=126)
        self.assertAlmostEqual(sigma.dropna().iloc[0], np.sqrt(0.02 ** 2 * MONTH), places=12)


class ManageTests(unittest.TestCase):
    def setUp(self):
        self.months = pd.period_range('2020-01', periods=4, freq='M')
        self.wml = pd.Series([0.10, -0.20, 0.05, -0.05], index=self.months)
        self.sigma = pd.Series([0.05, 0.10, 0.05, 0.20], index=self.months)

    def test_weight_is_target_over_forecast(self):
        weight, _ = manage(self.wml, self.sigma, target=0.12)
        self.assertAlmostEqual(weight.iloc[0], (0.12 / np.sqrt(12)) / 0.05)
        # 波动翻倍，仓位减半
        self.assertAlmostEqual(weight.iloc[1], weight.iloc[0] / 2)

    def test_managed_return_is_weight_times_return(self):
        weight, managed = manage(self.wml, self.sigma)
        pd.testing.assert_series_equal(managed, (weight * self.wml).rename('managed'))

    def test_cap_limits_leverage_only_at_the_top(self):
        weight, _ = manage(self.wml, self.sigma, target=0.12, cap=1.0)
        self.assertLessEqual(weight.max(), 1.0)
        self.assertAlmostEqual(weight.iloc[3], (0.12 / np.sqrt(12)) / 0.20)   # 低杠杆那档不受影响

    def test_target_scales_returns_but_not_the_sharpe(self):
        _, low = manage(self.wml, self.sigma, target=0.06)
        _, high = manage(self.wml, self.sigma, target=0.24)
        self.assertAlmostEqual(low.mean() / low.std(ddof=1), high.mean() / high.std(ddof=1))
        self.assertAlmostEqual((high / low).dropna().unique()[0], 4.0)


class MomentTests(unittest.TestCase):
    def test_drawdown_of_a_known_path(self):
        # +100% 之后 −50%，净值 1 → 2 → 1，最大回撤 −50%
        self.assertAlmostEqual(drawdown(pd.Series([1.0, -0.5])), -0.5)

    def test_moments_of_a_flat_series(self):
        result = moments(pd.Series([0.01] * 24, index=pd.period_range('2020-01', periods=24, freq='M')))
        self.assertAlmostEqual(result['年化收益'], 0.12)
        self.assertAlmostEqual(result['年化波动'], 0.0)
        self.assertEqual(result['月数'], 24)

    def test_sharpe_subtracts_the_risk_free_rate_when_given(self):
        months = pd.period_range('2020-01', periods=12, freq='M')
        returns = pd.Series(np.linspace(-0.02, 0.06, 12), index=months)
        rf = pd.Series(0.005, index=months)
        plain, net = moments(returns), moments(returns, rf)
        self.assertGreater(plain['夏普'], net['夏普'])
        self.assertAlmostEqual(plain['年化收益'], net['年化收益'])   # 收益不受影响，只有夏普变


class PredictabilityTests(unittest.TestCase):
    def test_uses_non_overlapping_monthly_windows(self):
        # 每个自然月内部恒定、月和月之间不同：不重叠口径下每月方差各自独立
        daily = pd.Series(0.0, index=pd.bdate_range('2020-01-01', '2020-06-30'))
        for i, month in enumerate(daily.index.to_period('M').unique()):
            daily[daily.index.to_period('M') == month] = 0.01 * (i + 1)
        variance = monthly_variance(daily)
        self.assertEqual(len(variance), 6)
        self.assertAlmostEqual(variance.iloc[0], 0.01 ** 2 * 252)
        self.assertAlmostEqual(variance.iloc[5], 0.06 ** 2 * 252)

    def test_white_noise_variance_is_not_predictable(self):
        rng = np.random.default_rng(0)
        daily = pd.Series(rng.normal(0, 0.01, 2600), index=pd.bdate_range('2010-01-01', periods=2600))
        self.assertLess(predictability(daily), 0.25)

    def test_persistent_variance_is_predictable(self):
        rng = np.random.default_rng(1)
        level = pd.Series(np.repeat(np.exp(np.linspace(-6, -3, 130)), 20)[:2600])
        daily = pd.Series(rng.normal(0, 1, 2600) * np.sqrt(level.to_numpy()),
                          index=pd.bdate_range('2010-01-01', periods=2600))
        self.assertGreater(predictability(daily), 0.5)


class CostTests(unittest.TestCase):
    """换名字和调杠杆的交易量要分开算：被换掉的名字卖旧买新，留下的只按杠杆差额调整。"""

    def setUp(self):
        self.months = pd.period_range('2020-01', periods=3, freq='M')

    def test_constant_leverage_only_pays_for_replaced_names(self):
        weight = pd.Series([1.0, 1.0, 1.0], index=self.months)
        # 一条腿换 10%：卖旧 0.1 + 买新 0.1 = 0.2；两条腿 0.4。10bp → 4bp
        paid = costs(weight, replaced=(0.1, 0.1), cost_bps=10.0)
        self.assertAlmostEqual(paid.iloc[1], 0.4 * 10 / 1e4)

    def test_no_replacement_only_pays_for_the_leverage_change(self):
        weight = pd.Series([1.0, 1.5, 1.5], index=self.months)
        paid = costs(weight, replaced=(0.0, 0.0), cost_bps=10.0)
        self.assertAlmostEqual(paid.iloc[1], 0.5 * 2 * 10 / 1e4)   # |Δw| × 两条腿
        self.assertAlmostEqual(paid.iloc[2], 0.0)

    def test_both_effects_add_up(self):
        weight = pd.Series([1.0, 2.0, 2.0], index=self.months)
        paid = costs(weight, replaced=(0.5, 0.5), cost_bps=10.0)
        # 每条腿：(1+2)×0.5 换名字 + |2−1|×0.5 调杠杆 = 2.0；两条腿 4.0
        self.assertAlmostEqual(paid.iloc[1], 4.0 * 10 / 1e4)

    def test_cost_scales_linearly_with_the_rate(self):
        weight = pd.Series([1.0, 1.4, 0.9], index=self.months)
        self.assertAlmostEqual(costs(weight, cost_bps=20.0).sum(),
                               costs(weight, cost_bps=10.0).sum() * 2)

    def test_higher_turnover_costs_more(self):
        weight = pd.Series([1.0, 1.0, 1.0], index=self.months)
        self.assertGreater(costs(weight, replaced=(0.4, 0.4)).sum(),
                           costs(weight, replaced=(0.1, 0.1)).sum())


class AlphaTests(unittest.TestCase):
    def test_pure_beta_has_no_alpha(self):
        months = pd.period_range('2000-01', periods=120, freq='M')
        market = pd.Series(np.random.default_rng(3).normal(0.005, 0.04, 120), index=months)
        result = alpha(market * 0.5, market)
        self.assertAlmostEqual(result['beta'], 0.5)
        self.assertAlmostEqual(result['年化alpha'], 0.0, places=10)

    def test_constant_edge_shows_up_as_alpha(self):
        months = pd.period_range('2000-01', periods=120, freq='M')
        market = pd.Series(np.random.default_rng(4).normal(0.005, 0.04, 120), index=months)
        result = alpha(market * 0.5 + 0.01, market)
        self.assertAlmostEqual(result['年化alpha'], 0.12)
        self.assertEqual(result['月数'], 120)


class ScheduleTests(unittest.TestCase):
    """杠杆更新频率：区间内权重必须保持不变，且只用上一个区间结束时能看到的数据。"""

    def setUp(self):
        self.daily = daily_series([0.01] * 200 + [0.03] * 200)

    def test_monthly_weight_is_constant_within_a_month(self):
        weight = daily_weights(self.daily, 'M').dropna()
        counts = weight.groupby(weight.index.to_period('M')).nunique()
        self.assertTrue((counts == 1).all())

    def test_daily_weight_moves_every_day(self):
        monthly = daily_weights(self.daily, 'M').dropna()
        every = daily_weights(self.daily, 'D').dropna()
        self.assertGreater(every.nunique(), monthly.nunique())

    def test_weekly_sits_between_monthly_and_daily(self):
        counts = [daily_weights(self.daily, s).dropna().nunique() for s in ('M', 'W', 'D')]
        self.assertEqual(counts, sorted(counts))

    def test_no_lookahead_a_later_change_cannot_move_an_earlier_weight(self):
        weight = daily_weights(self.daily, 'W')
        tampered = self.daily.copy()
        tampered.iloc[-20:] = 9.99
        after = daily_weights(tampered, 'W')
        early = weight.index < tampered.index[-20]
        pd.testing.assert_series_equal(weight[early], after[early])

    def test_costs_only_reduce_returns(self):
        # 波动率要连续变化，日度更新才真的比月度多换手（分段常数的话两者换手一样）
        rng = np.random.default_rng(7)
        wobbly = daily_series(rng.normal(0, 1, 800) * np.linspace(0.005, 0.03, 800))
        free, _ = schedules(wobbly, options=('M', 'D'), cost_bps=0.0)
        paid, _ = schedules(wobbly, options=('M', 'D'), cost_bps=20.0)
        self.assertTrue((paid['年化收益'] <= free['年化收益']).all())
        self.assertGreater(free.loc['每天', '年化换手'], free.loc['每月', '年化换手'])


if __name__ == '__main__':
    unittest.main()
