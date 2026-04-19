# US Stock Monitor

Repo-ready starter package for your US stock monitoring project.

## Structure

```text
us-stock-monitor/
├─ app/
│  └─ us_stock_monitor_app.py
├─ backtest/
│  └─ us_stock_template_backtest.py
├─ data/
│  └─ fundamentals_snapshot_template.csv
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## Included
- `app/us_stock_monitor_app.py`
  - Screen stocks on a chosen date
  - Save passed watchlists as `US stock dd-MMM-yyyy.csv`
  - Monitor saved watchlists daily
  - BUY / SELL / HOLD hooks are placeholders for later rules
- `backtest/us_stock_template_backtest.py`
  - Date-based screening and backtest framework
- `data/fundamentals_snapshot_template.csv`
  - Sample fundamentals snapshot template

## Install

```bash
pip install -r requirements.txt
```

## Run Streamlit app

```bash
streamlit run app/us_stock_monitor_app.py
```

## Run backtest script

```bash
python backtest/us_stock_template_backtest.py \
  --tickers CLS JBL SANM NVDA AAPL TSLA MELI HIMS \
  --start 2022-01-01 \
  --end 2025-12-31 \
  --asof 2024-12-31 \
  --rebalance monthly \
  --holding-period 63 \
  --output-dir output
```

## Notes
- For strict point-in-time historical fundamentals, you should later replace the sample CSV workflow with a proper SEC filing-based data pipeline.
- BUY / SELL / HOLD zone rules are intentionally not hardcoded yet.
