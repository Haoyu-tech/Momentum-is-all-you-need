"""复现 Barroso & Santa-Clara (2015)《Momentum has its moments》, JFE 116(1):111-120。

**论文的想法**：动量的夏普比市场、价值、规模都高，但崩起来也最惨（1932 年 8 月 −74%、
2009 年 4 月 −46%）。作者发现动量**自己的波动率**比市场波动率好预测得多，所以按它过去
六个月的已实现波动率反向缩放仓位，就能把崩盘基本消掉，夏普差不多翻倍。

**策略**（论文第 2 节）：

    σ̂²_t = 21 · (1/126) · Σ_{j=0}^{125} r²_{WML, d_{t-1}-j}      # 月度方差，用上月最后一个
                                                                  # 交易日往前 126 个日收益
    w_t   = σ_target / σ̂_t                                        # σ_target = 12% 年化
    r*_t  = w_t · r_{WML,t}                                       # 风险管理后的月收益

126 个日收益**截止到上月最后一个交易日**，用来缩放本月的收益，所以没有未来函数。
缩放不设上限，和原文一致；杠杆的分布会单独报出来，另外也报一个加了上限的稳健性检验。

**数据**：Kenneth French 数据库的动量因子（月度 + 日度）和三因子，CRSP 口径，1926 年至今。
论文样本是 1927-01 至 2011-12。本脚本除了复现这一段，还会跑**论文发表后的真正样本外**
（2012-01 起），这是原文不可能做到的检验。

输出到 data/mom/：barroso_monthly.csv（逐月）、barroso_summary.csv（各段统计），
图在 why/docs/mom_barroso.png。
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.metrics import COST_BPS

WHY = Path(__file__).resolve().parent.parent
TURNOVER = (0.269, 0.311)  # 多头腿、空头腿每月被替换的比例，实测值见 costs() 的说明
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FILES = {'deciles_daily': '10_Portfolios_Prior_12_2_Daily_CSV.zip',
         'deciles_monthly': '10_Portfolios_Prior_12_2_CSV.zip',
         'mom_daily': 'F-F_Momentum_Factor_daily_CSV.zip',
         'mom_monthly': 'F-F_Momentum_Factor_CSV.zip',
         'factors_daily': 'F-F_Research_Data_Factors_daily_CSV.zip',
         'factors_monthly': 'F-F_Research_Data_Factors_CSV.zip'}
DAYS, MONTH = 126, 21      # 六个月的交易日；一个月的交易日（论文用来把日方差换算成月方差）
TARGET = 0.12              # 目标波动率，12% 年化
PAPER_START, PAPER_END = '1927-03', '2011-12'   # 论文样本（Table 1 的表头）
MISSING = (-99.99, -999)   # French 文件里的缺失码


def fetch(name, cache):
    """下载并缓存 French 的 zip，返回解压出来的 CSV 文本。"""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / FILES[name]
    if not path.exists() or path.stat().st_size == 0:
        request = urllib.request.Request(FRENCH + FILES[name], headers={'User-Agent': 'Mozilla/5.0'})
        path.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    archive = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    return archive.read(archive.namelist()[0]).decode('latin-1')


def parse(text, monthly):
    """French 的 CSV 前面有一段说明文字，中间可能还有第二张表（年度），只取第一张。

    日期列：日度是 YYYYMMDD，月度是 YYYYMM。收益是百分数，这里转成小数。
    """
    rows, started = [], False
    for line in text.splitlines():
        head = line.split(',')[0].strip()
        if not started:
            if head.isdigit() and len(head) == (6 if monthly else 8):
                started = True
            else:
                continue
        if not (head.isdigit() and len(head) == (6 if monthly else 8)):
            break                       # 第一张表结束（空行，或下面的年度表）
        rows.append([p.strip() for p in line.split(',')])
    frame = pd.DataFrame(rows).set_index(0)
    frame.index = (pd.PeriodIndex(frame.index, freq='M') if monthly
                   else pd.to_datetime(frame.index, format='%Y%m%d'))
    frame = frame.apply(pd.to_numeric, errors='coerce')
    return frame.replace(list(MISSING), np.nan) / 100


def columns(text, monthly):
    """从说明文字后面那一行表头拿列名。"""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if parts[0] == '' and len(parts) > 1 and all(parts[1:]):
            return parts[1:]
    raise ValueError('没找到表头行')


def series(name, cache, monthly):
    """读一张 French 的表，列名填好。文件里可能有多张表（市值加权/等权/年度），只取第一张。"""
    text = fetch(name, cache)
    frame = parse(text, monthly=monthly)
    frame.columns = columns(text, monthly=monthly)
    return frame


def load(cache, source='deciles'):
    """返回日度动量、月度动量、月度无风险利率和市场超额收益，全部是小数收益。

    source='deciles'：论文的口径。脚注 4 写明赢家组是**前 10%**、输家组是后 10%，
    WML = 前 10% − 后 10%，用 French 的「10 Portfolios Formed on Prior (2-12) Returns」
    市值加权收益。这个序列的波动远高于 Mom 因子（论文 Table 1 报 27.53% 年化）。

    source='factor'：French 的 Mom 因子，前 30% 减后 30%，分散得多，波动只有一半左右。
    一开始我们用的是它，夏普 0.487 而不是论文的 0.53。
    """
    factors = series('factors_monthly', cache, monthly=True)
    if source == 'factor':
        daily = series('mom_daily', cache, monthly=False).iloc[:, 0]
        monthly = series('mom_monthly', cache, monthly=True).iloc[:, 0]
    else:
        wide_daily, wide_monthly = series('deciles_daily', cache, False), series('deciles_monthly', cache, True)
        daily = wide_daily['Hi PRIOR'] - wide_daily['Lo PRIOR']
        monthly = wide_monthly['Hi PRIOR'] - wide_monthly['Lo PRIOR']
    return (daily.rename('wml').dropna(), monthly.rename('wml').dropna(),
            factors['RF'].rename('rf'), factors['Mkt-RF'].rename('mkt'))


def load_daily_market(cache):
    """日度市场超额收益，用来和动量的方差可预测性作对比。"""
    text = fetch('factors_daily', cache)
    frame = parse(text, monthly=False)
    frame.columns = columns(text, monthly=False)
    return frame['Mkt-RF'].rename('mkt').dropna()


def realized(daily, months, days=DAYS, scale=MONTH):
    """每个月的已实现方差：用**截止上月最后一个交易日**的 days 个日收益，换算成月度。

    这是全文唯一可能出未来函数的地方。滚动窗口在日频上算完之后，取每个月最后一天的值，
    再整体后移一个月——所以 t 月用的全部是 t−1 月结束时就能看到的数据。
    """
    variance = daily.pow(2).rolling(days).mean() * scale
    month_end = variance.groupby(daily.index.to_period('M')).last()
    month_end = month_end.reindex(months)
    return np.sqrt(month_end.shift(1)).rename('sigma')


def manage(wml, sigma, target=TARGET, cap=None):
    """按目标波动率缩放：w_t = σ_target / σ̂_t，收益 r*_t = w_t · r_t。

    σ_target 是年化，σ̂ 是月度，所以要先除以 √12 对齐。cap 给定时限制最大杠杆。
    """
    weight = (target / np.sqrt(12)) / sigma
    if cap is not None:
        weight = weight.clip(upper=cap)
    return weight.rename('weight'), (weight * wml).rename('managed')


def drawdown(returns):
    """累计净值的最大回撤（小数）。"""
    curve = (1 + returns).cumprod()
    return float((curve / curve.cummax() - 1).min())


def moments(returns, rf=None):
    """论文表 1 的那组统计量，外加最大回撤和最差单月。"""
    returns = returns.dropna()
    if returns.empty:
        return {}
    excess = returns - rf.reindex(returns.index) if rf is not None else returns
    deviation = returns.std(ddof=1)
    return dict(月数=len(returns), 年化收益=float(returns.mean() * 12),
                年化波动=float(deviation * np.sqrt(12)),
                夏普=float(excess.mean() / deviation * np.sqrt(12)) if deviation > 0 else 0.0,
                偏度=float(returns.skew()), 峰度=float(returns.kurtosis()),
                最差单月=float(returns.min()), 最大回撤=drawdown(returns))


def monthly_variance(daily):
    """每个自然月内的已实现方差（年化）。

    必须用**不重叠**的窗口：滚动 126 天的话相邻两个月共用 105 个交易日，
    AR(1) 的 R² 会被这种机械重叠推到 0.94，那是重叠不是可预测性。
    """
    return daily.pow(2).groupby(daily.index.to_period('M')).mean().mul(252)


def predictability(daily, split=None):
    """论文的核心机制：动量自己的方差比市场方差好预测得多。

    对数已实现方差做 AR(1)。split 给定时报**样本外** R²——用扩展窗口逐月重估参数，
    和论文的口径一致（论文报动量 57.82%，比市场高 19.01 个百分点）。
    """
    series = np.log(monthly_variance(daily).replace(0, np.nan)).dropna()
    lagged, current = series.shift(1).dropna(), series.iloc[1:]
    if split is None:
        slope, intercept = np.polyfit(lagged, current, 1)
        fitted = intercept + slope * lagged
    else:
        start = current.index.get_indexer([pd.Period(split, freq='M')], method='bfill')[0]
        fitted = pd.Series(index=current.index[start:], dtype=float)
        for i in range(start, len(current)):
            slope, intercept = np.polyfit(lagged.iloc[:i], current.iloc[:i], 1)
            fitted.iloc[i - start] = intercept + slope * lagged.iloc[i]
        current = current.iloc[start:]
    return float(1 - ((current - fitted) ** 2).sum() / ((current - current.mean()) ** 2).sum())


def table(pieces, windows):
    """对每个时间段、每条收益序列算一遍统计量。"""
    rows = []
    for name, window in windows.items():
        for label, series in pieces.items():
            rows.append(dict(区间=name, 策略=label, **moments(series[window])))
    return pd.DataFrame(rows).set_index(['区间', '策略'])


def plot(monthly, split, path):
    """左：累计净值（对数轴），样本外加阴影；右：回撤。崩盘有没有被消掉，看右图。"""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from ..charts import AXIS, INK, INK2, MUTED, NEUTRAL, SERIES, plt, style, tidy
    style()
    stamps = monthly.index.to_timestamp()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(wspace=0.17))
    fig.suptitle('按自身波动率缩放，动量的崩盘基本消失', x=0.042, ha='left', y=0.97,
                 fontsize=15, color=INK, weight='bold')
    fig.text(0.042, 0.888, 'Barroso & Santa-Clara (2015, JFE) 的复现。用过去 126 个交易日的已实现波动率'
             '把动量仓位缩放到 12% 年化目标波动。\n'
             f'阴影是论文样本之后的真正样本外（{split} 起），这是原文做不到的检验。' + '\n'
             'WML = 前 10% 减后 10%（论文脚注 4 的口径），Kenneth French 数据，月度，不含交易成本。',
             fontsize=9.5, color=INK2, va='top', linespacing=1.5)
    shade = stamps[monthly.index >= split]
    series = [('动量（原始）', 'wml', SERIES[1]), ('动量（风险管理）', 'managed', SERIES[0]),
              ('市场超额', 'mkt', SERIES[2])]

    for axis in (left, right):
        tidy(axis)
        if len(shade):
            axis.axvspan(shade[0], stamps[-1], color=NEUTRAL, zorder=0)
    for label, column, colour in series:
        growth = (1 + monthly[column].fillna(0)).cumprod()
        left.plot(stamps, growth, color=colour, linewidth=1.6, label=label)
        curve = (1 + monthly[column].fillna(0)).cumprod()
        right.plot(stamps, (curve / curve.cummax() - 1) * 100, color=colour, linewidth=1.4, label=label)
    left.set_yscale('log')
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
    left.yaxis.set_minor_formatter(NullFormatter())
    left.set_title('1 美元的累计增长（对数轴，不含成本）', loc='left', fontsize=11.5, color=INK, pad=10)
    left.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc='upper left')
    right.set_title('回撤（%）', loc='left', fontsize=11.5, color=INK, pad=10)
    right.axhline(0, color=AXIS, linewidth=1)
    if len(shade):
        for axis in (left, right):
            axis.text(shade[0], axis.get_ylim()[1], '  样本外', fontsize=9, color=MUTED, va='top')
    fig.subplots_adjust(top=0.74, bottom=0.09, left=0.055, right=0.98)
    fig.savefig(path, dpi=160)
    print(f'图写入 {path}')


def daily_weights(daily, schedule='M', days=DAYS, target=TARGET, cap=None):
    """多久更新一次杠杆：'M' 每月、'W' 每周、'D' 每天。

    波动率估计用**截止昨天**的 days 个日收益，所以每个区间第一天用的是上个区间
    结束时就能看到的数据——和论文的月度口径一致，只是换了更新频率。
    """
    sigma = np.sqrt(daily.pow(2).rolling(days).mean()).shift(1)
    weight = (target / np.sqrt(252)) / sigma
    if cap is not None:
        weight = weight.clip(upper=cap)
    if schedule == 'D':
        return weight.rename('weight')
    marks = daily.index.to_period(schedule)
    return weight.where(~pd.Series(marks, index=daily.index).duplicated()).ffill().rename('weight')


def schedules(daily, options=('M', 'W', 'D'), cost_bps=0.0, **kwargs):
    """比较不同的杠杆更新频率。

    成本：WML 是 1 美元多头 + 1 美元空头，所以杠杆变动 Δw 要交易 2|Δw| 的名义金额。
    这只算**调杠杆**的成本，不含 WML 组合自己每月换股的成本（那部分三种频率都一样）。
    """
    rows, series = [], {}
    for schedule in options:
        weight = daily_weights(daily, schedule, **kwargs)
        gross = (weight.diff().abs() * 2).fillna(weight.abs() * 2)
        net = (weight * daily - gross * cost_bps / 1e4).dropna()
        monthly = (1 + net).groupby(net.index.to_period('M')).prod() - 1
        rows.append(dict(更新频率={'M': '每月', 'W': '每周', 'D': '每天'}[schedule],
                         年调杠杆次数=float(weight.diff().ne(0).groupby(weight.index.year).sum().mean()),
                         年化换手=float(gross.groupby(gross.index.year).sum().mean()),
                         **moments(monthly)))
        series[schedule] = monthly
    return pd.DataFrame(rows).set_index('更新频率'), series



def costs(weight, replaced=TURNOVER, cost_bps=COST_BPS):
    """每月的交易成本，占名义本金的比例。

    WML 是因子收益序列，看不到持仓，所以换手率只能外部给。默认值是用 why/data 的
    美股日线**实测**出来的：11-1 动量按十分位分组、等权、月度调仓、日均成交额 500 万
    美元以上，多头腿每月换掉 26.9% 的名字、空头腿 31.1%（13 次调仓，2024-07 起）。

    一条腿的交易量分两块：被换掉的名字要卖旧买新，交易 (w_{t-1}+w_t)·f；留下来的名字
    只需按杠杆变化调整，交易 |w_t − w_{t-1}|·(1−f)。两条腿相加就是总名义交易量。
    """
    before, now = weight.shift(1).fillna(0.0), weight
    leg = lambda f: (before + now) * f + (now - before).abs() * (1 - f)
    traded = sum(leg(f) for f in replaced)
    return (traded * cost_bps / 1e4).rename('cost')


def alpha(managed, market, rf=None):
    """对市场回归，返回年化 alpha、t 值、beta。managed 和 market 都是月度超额收益。"""
    frame = pd.concat([managed, market], axis=1).dropna()
    y, x = frame.iloc[:, 0], frame.iloc[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    error = residual.std(ddof=1) / np.sqrt(len(residual))
    return dict(年化alpha=intercept * 12, t值=intercept / error, beta=slope, 月数=len(y))


def wanted(path):
    """命令行传空字符串表示不要这张图。Path('') 会变成 Path('.')，不能直接当真值用。"""
    return path is not None and str(path) not in ('.', '')


def main():
    parser = argparse.ArgumentParser(description='Replicate Barroso & Santa-Clara (2015) risk-managed momentum')
    parser.add_argument('--cache', type=Path, default=WHY / 'data' / 'mom' / 'french')
    parser.add_argument('--output', type=Path, default=WHY / 'data' / 'mom')
    parser.add_argument('--days', type=int, default=DAYS, help='已实现方差用多少个交易日')
    parser.add_argument('--target', type=float, default=TARGET, help='目标波动率（年化）')
    parser.add_argument('--cap', type=float, default=None, help='杠杆上限，默认不设（与原文一致）')
    parser.add_argument('--split', default='2012-01', help='样本外从哪个月开始')
    parser.add_argument('--source', default='deciles', choices=['deciles', 'factor'],
                        help='deciles 是论文口径（前 10% 减后 10%）；factor 是 French 的 Mom 因子（前 30%）')
    parser.add_argument('--plot', type=Path, default=WHY / 'docs' / 'mom_barroso.png')
    parser.add_argument('--schedules', action='store_true',
                        help='比较杠杆更新频率：每月 / 每周 / 每天')
    parser.add_argument('--cost-bps', type=float, default=10.0, help='调杠杆的单边成本（bp）')
    args = parser.parse_args()

    daily, wml, rf, mkt = load(args.cache, args.source)
    sigma = realized(daily, wml.index, days=args.days)
    weight, managed = manage(wml, sigma, args.target, args.cap)
    monthly = pd.concat([wml, mkt, rf, sigma, weight, managed], axis=1).loc[wml.index]
    args.output.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output / 'barroso_monthly.csv', float_format='%.6f')

    windows = {f'论文样本 {PAPER_START} 至 {PAPER_END}': slice(PAPER_START, PAPER_END),
               f'样本外 {args.split} 至今': slice(args.split, None),
               '全样本': slice(None, None)}
    # 三条序列都对齐到风险管理能算出来的月份，否则多出来的前几个月会让对比失真
    live = managed.dropna().index
    pieces = {'动量（原始）': wml.reindex(live), '动量（风险管理）': managed.reindex(live),
              '市场超额': mkt.reindex(live)}
    summary = table(pieces, windows)
    summary.to_csv(args.output / 'barroso_summary.csv', float_format='%.6f')
    pd.set_option('display.width', 220)
    print(summary.round(4).to_string())
    print()
    daily_mkt = load_daily_market(args.cache)
    print(f"数据口径：{'前 10% 减后 10%（论文脚注 4）' if args.source == 'deciles' else 'Mom 因子（前 30%）'}")
    print('方差可预测性（对数已实现月方差的 AR(1)，不重叠窗口）：')
    for label, series in [('动量', daily), ('市场', daily_mkt)]:
        print(f'  {label}  样本内 R² {predictability(series):.4f}   '
              f'样本外 R² {predictability(series, args.split):.4f}')
    print(f'杠杆 w_t：中位 {weight.median():.2f}  5% 分位 {weight.quantile(.05):.2f}  '
          f'95% 分位 {weight.quantile(.95):.2f}  最大 {weight.max():.2f}')
    print()
    print(f'扣交易成本后（换手实测 多头腿 {TURNOVER[0]:.1%} / 空头腿 {TURNOVER[1]:.1%} 每月）')
    expense = costs(weight, cost_bps=args.cost_bps)
    print(f'  成本本身：中位 {expense.median() * 1e4:.1f} bp/月   年化 {expense.mean() * 12:.2%}   '
          f'名义换手年化 {(expense / (args.cost_bps / 1e4)).mean() * 12:.1f} 倍')
    rows = []
    for bps in sorted({0.0, 5.0, args.cost_bps, 20.0}):
        after = (weight * wml - costs(weight, cost_bps=bps)).reindex(live)
        for name, window in windows.items():
            piece = after[window].dropna()
            rows.append(dict(成本bp=bps, 区间=name, 年化收益=piece.mean() * 12,
                             夏普=piece.mean() / piece.std(ddof=1) * np.sqrt(12),
                             **alpha(piece, mkt.reindex(live)[window])))
    after_costs = pd.DataFrame(rows).set_index(['区间', '成本bp'])
    after_costs.to_csv(args.output / 'barroso_costs.csv', float_format='%.6f')
    print(after_costs.round(4).to_string())

    if args.schedules:
        pace, series = schedules(daily, cost_bps=args.cost_bps, days=args.days,
                                 target=args.target, cap=args.cap)
        print(f'\n杠杆更新频率的比较（调杠杆成本 {args.cost_bps:g} bp 单边，全样本）')
        print(pace.round(3).to_string())
        wide = pd.DataFrame(series).dropna()
        gap = wide['D'] - wide['M']
        error = gap.std(ddof=1) / np.sqrt(len(gap))
        print(f"每天 vs 每月：年化差 {gap.mean() * 12:+.4f}  t 值 {gap.mean() / error:.2f}  "
              f"两者相关系数 {wide['D'].corr(wide['M']):.4f}")
        pace.to_csv(args.output / 'barroso_schedules.csv', float_format='%.6f')
    if wanted(args.plot):
        plot(monthly.reindex(live), args.split, args.plot)
    print(f"\n写入 {args.output / 'barroso_monthly.csv'} 和 barroso_summary.csv")


if __name__ == '__main__':
    main()
