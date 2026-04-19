
# US Stock Monitor App

This version is built for your exact workflow:

1. Input a date, for example `01 Jan 2026`
2. Screen a list of US stocks using your fundamental and technical rules
3. Save the passed list as `US stock dd-MMM-yyyy.csv`
4. Monitor that saved list daily
5. Dashboard shows `BUY / SELL / HOLD / WATCH`
6. BUY / SELL / HOLD zone rules can be added later

## Current screening rules

### Fundamental
- At least 3 of last 4 EPS YoY values >= 20%
- At least 3 of last 4 Revenue YoY values >= 20%
- 3Y average ROE >= 15%

### Technical
- Close > MA50, MA150, MA200
- MA50 > MA150 > MA200
- Price within 25% of 52-week high
- Price at least 25% above 52-week low

## Important limitation
Historical point-in-time fundamentals are not cleanly available for free from Yahoo.
So:
- technical screening works immediately
- strict historical fundamental screening needs a fundamentals snapshot CSV

## Install
```bash
pip install streamlit yfinance pandas numpy matplotlib
```

## Run
```bash
streamlit run us_stock_monitor_app.py
```

## Files
Saved watchlists are stored in:
```text
saved_watchlists/
```

Example saved file:
```text
US stock 01-Jan-2026.csv
```

## Future rule hook points
Add your future action rules into:
- `buy_zone_rule()`
- `sell_zone_rule()`
- `hold_zone_rule()`
