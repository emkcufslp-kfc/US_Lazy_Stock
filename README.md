# US Stock Monitor

Production-style Streamlit dashboard for US stock screening and watchlist monitoring.

## Repository Structure

```text
US Lazy stock/
  app/
    us_stock_monitor_app.py
    saved_watchlists/
  backtest/
    us_stock_template_backtest.py
  data/
    full_screen_latest.csv
    fundamentals_snapshot_template.csv
  docs/
    SPEC.md
    ACCEPTANCE.md
    RISKS.md
    TODO.md
    CHANGELOG.md
    README_us_stock_monitor_app.md
  README.md
  requirements.txt
```

## Key Features
- Run stock screening on a selected date.
- Save passed stocks as `US stock dd-MMM-yyyy.csv`.
- Monitor saved watchlists daily.
- Offline sample mode for deterministic, no-network usage.
- BUY / SELL / HOLD hooks exist for future strategy rules.

## Install

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app/us_stock_monitor_app.py
```

## Recommended First Run (Offline)
1. Enable `Use offline sample data (no live API)` in the sidebar (enabled by default).
2. Click `Run screen`.
3. Review output and optionally save passed results.

## Optional Live Mode
- Turn off `Use offline sample data (no live API)`.
- Provide custom tickers or let the app build an auto-universe from free sources.
- Optionally upload fundamentals CSV (`data/fundamentals_snapshot_template.csv` as template).

## Validation

```bash
python -m py_compile app/us_stock_monitor_app.py
python scripts/validate_reference_sources.py
```

## Backtest Script

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
- For strict point-in-time fundamentals, replace free-source workflows with filing-grade pipelines.
- BUY / SELL / HOLD zone logic is currently placeholder-only.
- Additional GitHub datasets are kept as **reference-only fallbacks** (not primary truth).
- Fallback source verification report is generated at `docs/reference_validation_report.json`.
