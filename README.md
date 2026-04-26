# 10-Month Moving Average Dashboard

Interactive Streamlit dashboard for learning a monthly moving-average timing rule on SPY and QQQ with Stooq daily data.

For low-latency deployment, the app uses bundled CSV files in `data/stooq/`. This avoids runtime Stooq downloads, API keys, and upload steps.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Data

Preferred bundled ETF files:

- `data/stooq/spy_us_d.csv`
- `data/stooq/qqq_us_d.csv`

Index-proxy fallback files are also bundled:

- `data/stooq/^spx_d.csv`
- `data/stooq/^ndx_d.csv`

The loader prefers exact ETF exports (`spy_us_d.csv`, `spy.us_d.csv`, `spy_d.csv`, `spy.csv`, and the QQQ equivalents) before falling back to SPX/NDX aliases.

## Rule

The strategy holds the selected market when the previous monthly close is above its simple moving average. Otherwise it holds cash. Defaults are a 10-month moving average, 2% annual cash yield, 1.25x leveraged view, and a window floor of 1950 where data exists. The main dashboard keeps the beginner path simple: choose SPY or QQQ and review the rule. Advanced assumptions are available in the collapsed settings.

The leveraged strategy applies the selected leverage to monthly ticker returns while invested. If a leveraged monthly loss reaches or exceeds 100%, the strategy is treated as liquidated and remains at zero.

## Tests

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```
