# Momentum research modules

[Project overview](../../README.md) · [Methods](../../docs/METHODOLOGY.md) · [Data setup](../../docs/DATA.md)

| Module | Purpose |
|---|---|
| `momentum.py` | 252-day lookback with the last 21 trading days skipped; monthly/weekly selection |
| `universe.py` | Cached Wikipedia snapshot for the exploratory short-sample workflow |
| `hedge.py` | SPY/QQQ, sector, loser-leg and volatility-ETP comparisons |
| `barroso.py` | French decile WML and risk-management reproduction |
| `crsp.py` | Historical membership and full-history CRSP loaders |
| `breadth.py` | Top-N/decile, equal/value weights and within-sector selection |
| `overlay.py` | Rolling beta hedges, fixed hedges and volatility targets |
| `final.py` | Long-run construction comparison |
| `capacity.py` | Trailing dollar-volume capacity screening |
| `tests/` | Synthetic regression tests for signal timing, trading and risk scaling |

Run modules from the **repository root** using `python -m why.mom.<module>`. The `why` package layout preserves the original relative imports and shared backtest utilities.

The research source preserves the author's original comments and output-field names. Public-facing result exports, figures and documentation are English. CRSP archive paths are configurable through `MOM_CRSP_MEMBERS_ZIP` and `MOM_CRSP_STOCK_ZIP`.
