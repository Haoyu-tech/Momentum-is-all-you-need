"""Export newly computed aggregate research outputs with English labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LABELS={
    '前 10 只':'Top 10 stocks (VW)', '前 10% 市值加权':'Top decile (VW)',
    '前 10% 等权':'Top decile (EW)', '前 10% 多空':'Winner-minus-loser (VW)',
    '前 10% 市值加权 + 波动目标 19%':'Top decile (VW), 19% vol target',
    '前 10% 等权 + 波动目标 19%':'Top decile (EW), 19% vol target',
    '标普 500 等权（基准）':'Historical S&P 500 universe (EW)',
    '动量（原始）':'Unmanaged momentum', '动量（风险管理）':'Risk-managed momentum',
    '市场超额':'Market excess return', '全样本':'Full sample',
    '论文样本 1927-03 至 2011-12':'Paper window: 1927-03 to 2011-12',
    '样本外 2012-01 至今':'Extension: 2012-01 onward'}
COLUMNS={'0':'month','策略':'strategy','区间':'period','年化收益':'annualized_mean_return',
    '年化波动':'annualized_volatility','夏普':'sharpe','最大回撤':'max_drawdown',
    '偏度':'skewness','终值':'terminal_wealth','t值':'mean_return_t_stat','月数':'months',
    '峰度':'excess_kurtosis','最差单月':'worst_month','调仓次数':'rebalances',
    '紧约束中位':'median_bottleneck_capacity','紧约束下四分位':'q25_bottleneck_capacity',
    '总量中位':'median_aggregate_capacity','紧约束中位_2025年美元':'median_bottleneck_2025_usd',
    '持仓中位ADV':'median_holdings_adv','换手中位':'median_turnover','杠杆中位':'median_leverage',
    '调仓日':'rebalance_date','持股数':'stocks_changed','杠杆':'leverage','换手':'turnover',
    '紧约束容量':'bottleneck_capacity','总量容量':'aggregate_capacity','最紧的票':'bottleneck_permno',
    '持仓最小ADV':'min_holdings_adv','紧约束容量_2025':'bottleneck_capacity_2025_usd',
    '总量容量_2025':'aggregate_capacity_2025_usd'}
FILES=['final_summary.csv','final_daily.csv','barroso_summary.csv','barroso_monthly.csv',
       'capacity_summary.csv','capacity_series.csv','benchmark_value.csv']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'why/data/mom')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in FILES:
        source=args.source/name
        if not source.is_file():
            print(f'Skipped unavailable aggregate: {name}')
            continue
        frame=pd.read_csv(source).rename(columns={**COLUMNS,**LABELS})
        for col in frame.columns:
            if pd.api.types.is_string_dtype(frame[col].dtype):frame[col]=frame[col].replace(LABELS)
        frame=frame.drop(columns=['bottleneck_permno'],errors='ignore')
        destination=args.output/name
        if name in ['final_daily.csv','benchmark_value.csv']:
            destination=destination.with_suffix('.csv.gz')
            frame.to_csv(destination,index=False,compression={'method':'gzip','mtime':0})
        else:frame.to_csv(destination,index=False)
        records.append({'file':destination.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(frame)})
        print(f'Exported {destination.name}: {len(frame):,} rows')
    if not records:parser.error('No selected aggregate research outputs were found')
    (args.output/'export_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Files not regenerated remain at their previous snapshot. Inspect export_manifest.json before comparison.')

if __name__=='__main__':main()
