"""Render English publication figures from the included aggregate result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ASSETS = ROOT/'results', ROOT/'assets'
COLORS = ['#2563eb', '#10b981', '#f97316', '#a855f7', '#64748b']
EW = 'Top decile (EW)'
VW = 'Top decile (VW)'
SCALED_EW = 'Top decile (EW), 19% vol target'
SCALED_VW = 'Top decile (VW), 19% vol target'
BENCHMARK = 'Historical S&P 500 universe (EW)'

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.5,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.spines.left':False, 'axes.edgecolor':'#cbd5e1',
        'text.color':'#0f172a', 'axes.labelcolor':'#475569',
        'xtick.color':'#64748b', 'ytick.color':'#64748b',
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'svg.fonttype':'none'})

def tidy(ax):
    ax.grid(axis='y', color='#e2e8f0', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

def save(fig, name, note):
    fig.text(.06,.018,note,fontsize=8,color='#64748b')
    fig.tight_layout(rect=(0,.055,1,.93))
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS/f'{name}.png',dpi=130,facecolor='white')
    fig.savefig(ASSETS/f'{name}.svg',facecolor='white')
    # Indexed PNG export keeps the README assets compact without altering data.
    with Image.open(ASSETS/f'{name}.png') as im:
        im.convert('RGB').quantize(colors=256).save(ASSETS/f'{name}.png',optimize=True)
    plt.close(fig)

def main():
    style()
    daily=pd.read_csv(RESULTS/'final_daily.csv.gz',index_col=0,parse_dates=True)
    table=pd.read_csv(RESULTS/'final_summary.csv').set_index('strategy')
    order=[SCALED_EW,SCALED_VW,EW,VW,BENCHMARK]
    compact={SCALED_EW:'EW + volatility target',SCALED_VW:'VW + volatility target',
             EW:'Equal-weight decile',VW:'Value-weight decile',BENCHMARK:'Universe benchmark (EW)'}
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
    fig.suptitle('Equity momentum: long-run growth and drawdowns',x=.06,ha='left',fontweight='bold',fontsize=17)
    for name,color in zip(order,COLORS):
        growth=(1+daily[name]).cumprod()
        axes[0].plot(growth.index,growth,label=compact[name],color=color,lw=1.2)
        axes[1].plot(growth.index,100*(growth/growth.cummax()-1),color=color,lw=.9)
    axes[0].set_yscale('log')
    axes[0].set_title('Growth of $1 · logarithmic scale',loc='left')
    axes[1].set_title('Drawdown from the prior peak (%)',loc='left')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    for ax in axes: tidy(ax)
    save(fig,'01_performance',f'{daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d} | Stored portfolio returns | 10 bps one-way model cost | No new stock-level backtest')

    fig,(left,right)=plt.subplots(1,2,figsize=(12.5,4.7))
    fig.suptitle('Portfolio construction changes the risk profile',x=.06,ha='left',fontweight='bold',fontsize=17)
    short=['Top 10 (VW)','Top decile (VW)','Top decile (EW)','WML (VW)','VW + target','EW + target','Universe (EW)']
    left.barh(short,table['sharpe'],color=['#94a3b8','#f97316','#a855f7','#e11d48','#10b981','#2563eb','#64748b'])
    left.invert_yaxis();left.set_xlabel('Annualized mean / volatility (risk-free = 0)')
    left.set_title('Reported Sharpe ratio',loc='left')
    right.barh(short,-100*table['max_drawdown'],color='#94a3b8')
    right.invert_yaxis();right.set_xlabel('Peak-to-trough loss (%)')
    right.set_title('Maximum drawdown magnitude',loc='left')
    for ax in [left,right]:tidy(ax)
    save(fig,'02_risk_comparison','The 19% target is an in-sample research setting. Financing and stock-borrow costs are not modeled.')

    eras=[('1958–1974',1958,1974),('1975–1999',1975,1999),('2000–2009',2000,2009),('2010–2025',2010,2025)]
    rows=[]
    for label,low,high in eras:
        part=daily.loc[(daily.index.year>=low)&(daily.index.year<=high)]
        extra=part[SCALED_EW]-part[BENCHMARK]
        rows.append({'period':label,'strategy':part[SCALED_EW].mean()*252,
            'benchmark':part[BENCHMARK].mean()*252,'excess':extra.mean()*252,
            'iid_t':extra.mean()/extra.std()*np.sqrt(len(extra))})
    era=pd.DataFrame(rows)
    era.to_csv(RESULTS/'period_comparison.csv',index=False,float_format='%.8f')
    fig,ax=plt.subplots(figsize=(11.5,4.7))
    fig.suptitle('Historical excess returns are concentrated before 2000',x=.06,ha='left',fontweight='bold',fontsize=17)
    bars=ax.bar(era['period'],100*era['excess'],color=['#2563eb' if v>0 else '#e11d48' for v in era['excess']])
    for bar,t in zip(bars,era['iid_t']):
        if bar.get_height()>0:
            ax.annotate(f't = {t:.2f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
        else:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()/2,f't = {t:.2f}',
                    ha='center',va='center',fontsize=10,color='white')
    ax.axhline(0,color='#94a3b8',lw=.8);ax.set_ylabel('Annualized mean excess return (%)');tidy(ax)
    save(fig,'03_periods','EW top decile + 19% volatility target minus the EW historical-universe benchmark | t-statistics use an IID standard error, not HAC')

    monthly=pd.read_csv(RESULTS/'barroso_monthly.csv',index_col=0)
    monthly.index=pd.PeriodIndex(monthly.index,freq='M').to_timestamp('M')
    paper=monthly.loc['1927-03':'2011-12']
    fig,axes=plt.subplots(1,2,figsize=(12.5,5))
    fig.suptitle('Risk-managed momentum: a paper-window reproduction',x=.06,ha='left',fontweight='bold',fontsize=17)
    for key,label,color in [('wml','Unmanaged WML','#f97316'),('managed','Risk-managed WML','#2563eb')]:
        series=paper[key].dropna();growth=(1+series).cumprod()
        axes[0].plot(growth.index,growth,label=label,color=color,lw=1.1)
        axes[1].plot(growth.index,100*(growth/growth.cummax()-1),color=color,lw=.9)
    axes[0].set_yscale('log');axes[0].legend(frameon=False)
    axes[0].set_title('Compounded monthly return index',loc='left')
    axes[1].set_title('Drawdown (%)',loc='left')
    for ax in axes:tidy(ax)
    save(fig,'04_paper_replication','Barroso & Santa-Clara (2015) inspired implementation | French decile WML, 126-day realized variance, 12% target | Updated data vintage')

    cap=pd.read_csv(RESULTS/'capacity_series.csv',index_col=0,parse_dates=True)
    cap_table=pd.read_csv(RESULTS/'capacity_summary.csv')
    fig,(left,right)=plt.subplots(1,2,figsize=(12.5,4.8))
    fig.suptitle('Trading capacity is a separate constraint',x=.06,ha='left',fontweight='bold',fontsize=17)
    left.plot(cap.index,cap['bottleneck_capacity'],color='#2563eb',lw=1,label='Nominal USD')
    left.plot(cap.index,cap['bottleneck_capacity_2025_usd'],color='#10b981',lw=1,label='Approx. 2025 USD')
    left.set_yscale('log');left.legend(frameon=False)
    left.set_title('Bottleneck capacity at each rebalance',loc='left')
    left.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'${x/1e6:g}m' if x>=1e6 else f'${x/1e3:g}k'))
    right.bar(cap_table['period'],cap_table['median_bottleneck_capacity']/1e6,color='#2563eb')
    right.set_ylabel('Nominal USD, millions');right.set_title('Median capacity by era',loc='left')
    for ax in [left,right]:tidy(ax)
    save(fig,'05_capacity','EW portfolio | 10% of trailing median dollar volume | One-day execution | Screening model; no market-impact curve or fill guarantee')

    errors=[]
    for name in daily:
        series=daily[name];growth=(1+series).cumprod()
        computed={'annualized_mean_return':series.mean()*252,
            'annualized_volatility':series.std()*np.sqrt(252),
            'sharpe':series.mean()/series.std()*np.sqrt(252),
            'max_drawdown':(growth/growth.cummax()-1).min()}
        for key,value in computed.items():
            if abs(value-table.loc[name,key])>2e-6:errors.append(f'{name}: {key}')
    record={'rows':len(daily),'start':str(daily.index.min().date()),'end':str(daily.index.max().date()),
            'stored_table_checks':errors,'figures':5}
    (RESULTS/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if errors:raise SystemExit('Stored metrics do not match the bundled portfolio returns')

if __name__=='__main__':main()
