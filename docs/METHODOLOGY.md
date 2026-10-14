# Methodology and interpretation

## Signal and trading timeline

For a total-return index or adjusted-price panel, the stock signal is:

$$m_{i,t}=\frac{P_{i,t-21}}{P_{i,t-252}}-1.$$

This implementation uses trading-day approximations to a past-year signal that skips the latest month. It requires at least 80% observed prices in the lookback window. Monthly signals are formed at the last trading day, execution is assumed at the following trading day's close, and portfolio returns begin the day after execution. The code's synthetic tests check this alignment.

## Historical universe and full-history signal

The CRSP route determines membership from dated index constituent records, including firms removed from the index. Stock signals use each covered stock's full return history, including available pre-entry observations, before masking by current membership.

This addresses the current-member snapshot problem. It does not establish that every other input is point-in-time: the sector map uses the last available in-index SIC classification per stock. Within-sector experiments therefore have a classification-history limitation.

## Portfolio construction

- Top 10 and top-decile long-only portfolios are compared with winner-minus-loser portfolios.
- Equal weights and formation-date capitalization weights are held between rebalances.
- Reported transaction charges use weight/constituent changes at rebalance boundaries. They do not fully model drift-aware turnover, execution impact, financing or stock-borrow costs.
- Missing holding-period returns are filled with zero by the portfolio utilities. Delisting and missing-data handling should be audited against the raw export before treating the series as investable.
- The comparison benchmark is a monthly rebalanced, equal-weight portfolio of the historical S&P 500 universe, not the published capitalization-weighted S&P 500 index. The benchmark does not deduct an equivalent transaction-cost charge.

## Volatility scaling

The paper-oriented workflow estimates volatility from the preceding 126 daily WML returns and applies a 12% annual target to monthly returns. The CRSP portfolio extension applies a similarly lagged rolling volatility estimate to the constructed strategy, updates exposure monthly and caps leverage at 3.

The 19% target in the final comparison was chosen from an in-sample target ladder. This setting is a research comparison, not an independently selected out-of-sample optimum. The synthetic tests confirm that future observations cannot alter earlier volatility and beta estimates; they do not validate every empirical-data assumption.

## Metric definitions

| Metric | Definition in the final result export |
|---|---|
| Annualized return | Mean daily net return × 252; an arithmetic annualized mean, not CAGR |
| Annualized volatility | Daily sample standard deviation × √252 |
| Sharpe | Annualized mean divided by annualized volatility, with risk-free rate set to zero |
| Drawdown | Peak-to-trough loss of the compounded daily wealth series |
| Terminal wealth | Product of `(1 + daily return)`, starting at 1 |
| Mean-return t-statistic | Daily mean / standard deviation × √N; IID, not HAC |

The French reproduction has separate monthly-return and risk-free conventions. Do not directly equate its Sharpe with the long-only CRSP comparison without aligning those conventions.

## Capacity model

Capacity is screened from lagged trailing 21-day median dollar volume, a 10% participation assumption and one-day execution. For each weight change, the bottleneck calculation is approximately:

$$A_t = \min_i \frac{0.10\,ADV_{i,t}}{L_t |\Delta w_{i,t}|}.$$

The model omits market-impact curves, bid–ask dynamics and fill uncertainty. Weight changes use consecutive target weights, and leverage-change turnover is not fully represented. Dropping names with unavailable volume can loosen the estimated constraint. The inflation adjustment uses a sparse, interpolated CPI lookup and is approximate.

The aggregate-volume bound is more permissive than the stock-specific bottleneck and should not be presented as guaranteed tradable assets under management.

## Interpretation

The recorded scaled equal-weight decile has a higher full-sample Sharpe than the unscaled versions, but most benchmark-relative excess return is concentrated before 2000. The 2000-onward comparison should be examined separately; full-sample performance is not evidence of continuing excess returns.

Fixed 10 bps costs across decades are optimistic for early historical periods. The source also provides an illustrative era-based cost ladder; neither schedule substitutes for a calibrated stock-level transaction model. Research configurations were explored on the same data, so multiple-comparison and selection effects remain.

See [references](REFERENCES.md) for the literature and the implementation-to-paper mapping.
