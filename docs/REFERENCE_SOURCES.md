# Reference-Only GitHub Sources

These repositories are intentionally treated as **reference-only** backups when primary free feeds (for example Yahoo/SEC/free exchange endpoints) are unavailable.

## Verified sources
- datasets/s-and-p-500-companies
  - `https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv`
- datasets/s-and-p-500-companies-financials
  - `https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/main/data/constituents-financials.csv`
- rreichel3/US-Stock-Symbols
  - `https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt`
  - `https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt`
  - `https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt`
- Ate329/top-us-stock-tickers
  - `https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/all.csv`
  - `https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv`
- SteelCerberus/us-market-data
  - `https://raw.githubusercontent.com/SteelCerberus/us-market-data/main/data/us_market_data.csv`
- datasets/s-and-p-500
  - `https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv`

## Validation policy
- Validate HTTP status, required schema columns, row counts, and ticker format.
- Run cross-source consistency checks (for example S&P 500 ticker overlap).
- Do not use a source as fallback unless it passes validation.

## How to validate
```bash
python scripts/validate_reference_sources.py
```

Validation output:
- `docs/reference_validation_report.json`
