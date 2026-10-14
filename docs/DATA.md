# Data and reproduction

## Three distinct workflows

| Workflow | Required inputs | Included here |
|---|---|---|
| Figure regeneration | Aggregate portfolio CSVs under `results/` | Yes |
| French WML reproduction | Public daily/monthly French files, downloaded by `barroso.py` | Code and aggregate recorded output |
| CRSP stock-level backtest | Authorized membership and daily-stock archives | Code only; raw inputs and stock-level caches are external |

The bundled long-run portfolio return file spans **1958-03-04 to 2025-09-30**. The original local notes sometimes describe raw-data availability through December 2025; that is not the endpoint of the portfolio series exported here.

## Preview the results without market-data credentials

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py
python -B -m unittest discover -s why/mom/tests -t .
```

The figure script consumes stored aggregate outputs. It does not fetch stock-level market data or rerun the portfolio backtest. PNG files are used in the README; the script also exports SVG versions for sharing.

## Prepare CRSP inputs

Obtain the matching CRSP Version 2 S&P 500 constituent and daily-stock exports through your authorized data source. The membership archive is expected to include:

`PERMNO`, `Ticker`, `SICCD`, `MbrStartDt`, `MbrEndDt`, `DlyCalDt`, `DlyRet`, `DlyCap`, `DlyClose`.

The daily-stock archive requires:

`PERMNO`, `DlyCalDt`, `DlyRet`, `DlyCap`, `Ticker`, `SICCD`.

Capacity analysis additionally requires `DlyVol` and `DlyClose` in the membership archive. Archives are expected to contain a readable CSV as their first entry; adjust the loader if your export has a different layout.

```bash
python scripts/prepare_crsp.py --members /path/to/constituents.zip --stocks /path/to/daily_stock.zip --with-volume
python -m why.mom.final --plot ""
python -m why.mom.capacity --plot ""
python scripts/export_results.py
python scripts/plot_results.py
```

The preparation script writes local caches to `why/data/mom/`. Those caches and raw archives are excluded from version control. Large stock-level extraction and matrix construction need sufficient disk space and RAM.

Alternatively set `MOM_CRSP_MEMBERS_ZIP` and `MOM_CRSP_STOCK_ZIP`; the preparation script uses these defaults. Environment variables must be set before importing the modules.

## French data workflow

```bash
python -m why.mom.barroso --help
python -m why.mom.barroso
```

The downloader accesses the Kenneth French Data Library. Today's download can differ from the historical vintage used to generate the included outputs. Record retrieval date, sample endpoint and archive hashes before comparing results. The paper-window comparison is a reproduction attempt on the available vintage, not a guarantee of an exact archival match.

## Exploratory short-sample workflow

The original `momentum.py` and `hedge.py` use a separate daily-price panel and benchmark panel. They are included for completeness but are not the headline historical-membership experiment. Required daily-panel columns are `date`, `instrument_id`, `symbol`, `close`, `adj_close_pit` and `ret`; benchmark inputs provide `date`, `symbol` and `ret`.

The Wikipedia membership snapshot omits companies removed before that snapshot. Do not present this exploratory universe as historically complete. Set `--data` and `--output` explicitly and inspect `--help` before using it.

## Provenance and exports

[provenance.json](../results/provenance.json) records source-file hashes and export schemas. English tables were generated from the local stored research outputs; signal and backtest calculations were preserved. The only source-path adaptation replaces machine-specific CRSP input paths with environment-variable/repository-relative defaults.

New stock-level runs write original research outputs under `why/data/mom/`. Those raw output labels may be Chinese. Use the English export helper before rerendering the public figures. The included `results/` snapshot remains distinct from newly computed outputs until explicitly exported.
