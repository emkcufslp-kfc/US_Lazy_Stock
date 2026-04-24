# Product Spec

## Purpose
Build a production-style Streamlit dashboard for US stock screening and watchlist monitoring with an offline-safe default path.

## Core User Flows
1. Select screening and monitoring dates.
2. Run a stock screen and inspect pass/fail details.
3. Save passed stocks as a dated watchlist CSV.
4. Load a saved watchlist and run monitor actions (BUY/SELL/HOLD/WATCH placeholders).

## Functional Requirements
- Provide an offline sample mode that does not require live APIs.
- Allow optional live data screening using Yahoo/SEC/free sources.
- Support optional fundamentals snapshot CSV upload.
- Save and list watchlist files under `app/saved_watchlists/`.
- Keep buy/sell/hold rule hooks as explicit extension points.

## Non-Functional Requirements
- Safe-by-default behavior for constrained/no-network environments.
- Reversible file outputs (CSV only inside repo paths).
- Clear UI labels for data source mode and caveats.

## Data Sources
- Offline: `data/full_screen_latest.csv` and `data/fundamentals_snapshot_template.csv`.
- Optional online: Yahoo Finance, SEC company facts, S&P/NASDAQ free endpoints.
- Secondary validation: Financial Modeling Prep free/basic API when `FMP_API_KEY` is configured in Streamlit secrets or the environment.
