# LNR stock-price forecasting

A stock-specific version of the S&P 500 multivariate forecasting implementation
for Linamar Corporation (`LNR.TO`). It forecasts adjusted-close cumulative log
returns at 1, 5, and 21 business-day horizons:

- `lnr_logret_1b`: next-session return
- `lnr_logret_5b`: forward one-week return
- `lnr_logret_21b`: forward one-month return

The target and optional covariates are loaded from Yahoo Finance. Covariates are
lagged by one business day before registration, preserving the same cutoff-aware
forecasting contract used by the S&P 500 implementation. The default panel uses
VIX, WTI, NASDAQ, gold, and the dollar index; it does not require a FRED API key.

## Quick start

From the repository root:

```bash
source .venv/bin/activate
PYTHONPATH=aieng-forecasting:implementations \
  python implementations/LNR_Forecasting/run_lnr_forecast.py
```

Use `--refresh` to fetch fresh Yahoo Finance data. The default run includes all
Yahoo covariates; use `--no-covariates` for a fast target-only smoke check.

## Layout

```text
LNR_Forecasting/
├── 01_lnr_multivariate_backtest.ipynb # notebook smoke test and baseline backtest
├── data.py                 # LNR targets and leak-safe Yahoo covariates
├── analysis.py             # compact return diagnostics
├── run_lnr_forecast.py     # command-line smoke runner
└── specs/lnr_smoke.yaml    # short backtest design
```

This package reuses predictors from `aieng.forecasting.methods.numerical`.
The S&P 500 notebook can be adapted to this package by replacing its data
imports and target IDs with the exports from `LNR_Forecasting.data`.

The forecasts are statistical return forecasts, not investment advice. LNR is
an industrial stock, so WTI is treated as one market covariate rather than as a
proxy for the company's revenue or fundamentals.
