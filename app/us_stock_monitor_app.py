
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
WATCHLIST_DIR = APP_DIR / "saved_watchlists"
WATCHLIST_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class ScreenConfig:
    min_eps_yoy: float = 20.0
    min_revenue_yoy: float = 20.0
    min_roe_avg: float = 15.0
    high_52w_within_pct: float = 25.0
    low_52w_above_pct: float = 25.0
    require_fundamentals: bool = False


@st.cache_data(show_spinner=False)
def download_price_history(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=pd.Timestamp(end) + pd.Timedelta(days=5),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    data: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            tmp = raw[t].copy()
            tmp.columns = [str(c).title() for c in tmp.columns]
            tmp.index = pd.to_datetime(tmp.index)
            tmp = tmp.sort_index().dropna(how="all")
            if not tmp.empty:
                data[t] = tmp
    else:
        tmp = raw.copy()
        tmp.columns = [str(c).title() for c in tmp.columns]
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.sort_index().dropna(how="all")
        if tickers and not tmp.empty:
            data[tickers[0]] = tmp

    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["MA50"] = close.rolling(50, min_periods=50).mean()
    out["MA150"] = close.rolling(150, min_periods=150).mean()
    out["MA200"] = close.rolling(200, min_periods=200).mean()
    out["HIGH_52W"] = close.rolling(252, min_periods=252).max()
    out["LOW_52W"] = close.rolling(252, min_periods=252).min()
    out["RET_63D"] = close.pct_change(63)
    return out


def load_fundamentals_snapshot(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    req = {"as_of_date", "ticker"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Fundamentals CSV missing columns: {sorted(missing)}")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df.sort_values(["ticker", "as_of_date"])


def get_fund_row(ticker: str, as_of_date: pd.Timestamp, fdf: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if fdf is None:
        return None
    temp = fdf[(fdf["ticker"] == ticker) & (fdf["as_of_date"] <= as_of_date)]
    if temp.empty:
        return None
    return temp.iloc[-1]


def eval_fundamental(
    ticker: str,
    as_of_date: pd.Timestamp,
    fdf: Optional[pd.DataFrame],
    cfg: ScreenConfig,
) -> Tuple[bool, Dict[str, object]]:
    row = get_fund_row(ticker, as_of_date, fdf)
    if row is None:
        if cfg.require_fundamentals:
            return False, {"fundamentals_used": False, "note": "No fundamentals row"}
        return True, {"fundamentals_used": False, "note": "Fundamentals skipped"}

    eps_cols = [c for c in ["eps_q1_yoy", "eps_q2_yoy", "eps_q3_yoy", "eps_q4_yoy"] if c in row.index]
    rev_cols = [c for c in ["revenue_q1_yoy", "revenue_q2_yoy", "revenue_q3_yoy", "revenue_q4_yoy"] if c in row.index]
    roe_cols = [c for c in ["roe_y1", "roe_y2", "roe_y3"] if c in row.index]

    eps_vals = pd.to_numeric(row[eps_cols], errors="coerce") if eps_cols else pd.Series(dtype=float)
    rev_vals = pd.to_numeric(row[rev_cols], errors="coerce") if rev_cols else pd.Series(dtype=float)
    roe_vals = pd.to_numeric(row[roe_cols], errors="coerce") if roe_cols else pd.Series(dtype=float)

    eps_pass = (eps_vals >= cfg.min_eps_yoy).sum() >= 3 if len(eps_vals) >= 3 else False
    rev_pass = (rev_vals >= cfg.min_revenue_yoy).sum() >= 3 if len(rev_vals) >= 3 else False
    roe_avg = float(roe_vals.mean()) if len(roe_vals) >= 1 else np.nan
    roe_pass = roe_avg >= cfg.min_roe_avg if len(roe_vals) >= 1 else False

    return bool(eps_pass and rev_pass and roe_pass), {
        "fundamentals_used": True,
        "eps_pass": bool(eps_pass),
        "rev_pass": bool(rev_pass),
        "roe_pass": bool(roe_pass),
        "roe_avg": roe_avg,
        "eps_vals": list(eps_vals.values),
        "rev_vals": list(rev_vals.values),
    }


def eval_technical(df: pd.DataFrame, as_of_date: pd.Timestamp, cfg: ScreenConfig) -> Tuple[bool, Dict[str, object]]:
    temp = df[df.index <= as_of_date]
    if temp.empty:
        return False, {"note": "No price data up to selected date"}

    row = temp.iloc[-1]
    fields = ["Close", "MA50", "MA150", "MA200", "HIGH_52W", "LOW_52W"]
    if any(pd.isna(row.get(f)) for f in fields):
        return False, {"note": "Not enough lookback for MA / 52-week range"}

    close = float(row["Close"])
    ma50 = float(row["MA50"])
    ma150 = float(row["MA150"])
    ma200 = float(row["MA200"])
    high_52w = float(row["HIGH_52W"])
    low_52w = float(row["LOW_52W"])

    above_mas = close > ma50 and close > ma150 and close > ma200
    ma_stack = ma50 > ma150 > ma200
    within_high = (close <= high_52w) and (close >= high_52w * (1 - cfg.high_52w_within_pct / 100.0))
    above_low = close >= low_52w * (1 + cfg.low_52w_above_pct / 100.0)

    out = {
        "close": close,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "above_mas": bool(above_mas),
        "ma_stack": bool(ma_stack),
        "within_high": bool(within_high),
        "above_low": bool(above_low),
        "ret_63d": float(row.get("RET_63D", np.nan)),
    }
    return bool(above_mas and ma_stack and within_high and above_low), out


def buy_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> bool:
    return False


def sell_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> bool:
    return False


def hold_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> bool:
    return False


def derive_action_status(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> str:
    if sell_zone_rule(ticker, df, now_date):
        return "SELL"
    if buy_zone_rule(ticker, df, now_date):
        return "BUY"
    if hold_zone_rule(ticker, df, now_date):
        return "HOLD"
    return "WATCH"


def watchlist_filename(as_of_date: pd.Timestamp) -> str:
    return f"US stock {as_of_date.strftime('%d-%b-%Y')}.csv"


def save_watchlist(screen_df: pd.DataFrame, as_of_date: pd.Timestamp) -> Path:
    path = WATCHLIST_DIR / watchlist_filename(as_of_date)
    screen_df.to_csv(path, index=False)
    return path


def list_saved_watchlists() -> List[Path]:
    return sorted(WATCHLIST_DIR.glob("US stock *.csv"))


def load_watchlist(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def screen_on_date(
    tickers: List[str],
    as_of_date: pd.Timestamp,
    start_download: str,
    end_download: str,
    cfg: ScreenConfig,
    fdf: Optional[pd.DataFrame],
) -> pd.DataFrame:
    price_data = download_price_history(tickers, start_download, end_download)
    price_data = {t: add_indicators(df) for t, df in price_data.items()}

    rows = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            rows.append({"ticker": ticker, "pass": False, "note": "No price data"})
            continue

        tech_pass, tech = eval_technical(df, as_of_date, cfg)
        fund_pass, fund = eval_fundamental(ticker, as_of_date, fdf, cfg)
        final_pass = tech_pass and fund_pass

        score = 0
        score += 1 if tech.get("above_mas") else 0
        score += 1 if tech.get("ma_stack") else 0
        score += 1 if tech.get("within_high") else 0
        score += 1 if tech.get("above_low") else 0
        score += 1 if fund.get("eps_pass") else 0
        score += 1 if fund.get("rev_pass") else 0
        score += 1 if fund.get("roe_pass") else 0

        rows.append({
            "ticker": ticker,
            "pass": final_pass,
            "score": score,
            "technical_pass": tech_pass,
            "fundamental_pass": fund_pass,
            "close": tech.get("close"),
            "ma50": tech.get("ma50"),
            "ma150": tech.get("ma150"),
            "ma200": tech.get("ma200"),
            "high_52w": tech.get("high_52w"),
            "low_52w": tech.get("low_52w"),
            "ret_63d": tech.get("ret_63d"),
            "eps_pass": fund.get("eps_pass"),
            "rev_pass": fund.get("rev_pass"),
            "roe_pass": fund.get("roe_pass"),
            "fundamentals_used": fund.get("fundamentals_used"),
            "note": fund.get("note") or tech.get("note"),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["pass", "score", "ret_63d"], ascending=[False, False, False]).reset_index(drop=True)


def monitor_watchlist(watchlist_df: pd.DataFrame, monitor_date: pd.Timestamp) -> pd.DataFrame:
    tickers = watchlist_df["ticker"].astype(str).str.upper().tolist()
    start = (monitor_date - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    end = (monitor_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    price_data = download_price_history(tickers, start, end)
    price_data = {t: add_indicators(df) for t, df in price_data.items()}

    rows = []
    for _, row in watchlist_df.iterrows():
        ticker = str(row["ticker"]).upper()
        df = price_data.get(ticker)
        if df is None or df.empty:
            rows.append({"ticker": ticker, "action": "WATCH", "current_price": np.nan, "monitor_note": "No current price data"})
            continue

        temp = df[df.index <= monitor_date]
        if temp.empty:
            rows.append({"ticker": ticker, "action": "WATCH", "current_price": np.nan, "monitor_note": "No data up to monitor date"})
            continue

        latest = temp.iloc[-1]
        rows.append({
            "ticker": ticker,
            "action": derive_action_status(ticker, df, monitor_date),
            "current_price": float(latest["Close"]),
            "ret_63d": float(latest.get("RET_63D", np.nan)),
            "ma50": float(latest.get("MA50", np.nan)),
            "ma150": float(latest.get("MA150", np.nan)),
            "ma200": float(latest.get("MA200", np.nan)),
            "monitor_note": "Rules pending",
        })

    return watchlist_df.merge(pd.DataFrame(rows), on="ticker", how="left")


st.set_page_config(page_title="US Stock Monitor", layout="wide")
st.title("US Stock Monitor")
st.caption("Screen on a chosen date, save the selected list, then monitor it daily until buy/sell/hold rules are added.")

with st.sidebar:
    st.header("Input")
    tickers_text = st.text_area("US stock universe", value="CLS,JBL,SANM,NVDA,AAPL,TSLA,MELI,HIMS")
    screen_date = st.date_input("Screen date", value=date(2026, 1, 1))
    monitor_date = st.date_input("Monitor date", value=date.today())
    fundamentals_file = st.file_uploader("Optional fundamentals snapshot CSV", type=["csv"])
    require_fundamentals = st.checkbox("Require fundamentals for pass", value=False)

    st.header("Rules")
    min_eps_yoy = st.number_input("Min EPS YoY (%)", value=20.0, step=1.0)
    min_revenue_yoy = st.number_input("Min Revenue YoY (%)", value=20.0, step=1.0)
    min_roe_avg = st.number_input("Min 3Y average ROE (%)", value=15.0, step=1.0)
    high_52w_within_pct = st.number_input("Within 52W high (%)", value=25.0, step=1.0)
    low_52w_above_pct = st.number_input("Above 52W low (%)", value=25.0, step=1.0)

tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
as_of_ts = pd.Timestamp(screen_date)
monitor_ts = pd.Timestamp(monitor_date)

cfg = ScreenConfig(
    min_eps_yoy=float(min_eps_yoy),
    min_revenue_yoy=float(min_revenue_yoy),
    min_roe_avg=float(min_roe_avg),
    high_52w_within_pct=float(high_52w_within_pct),
    low_52w_above_pct=float(low_52w_above_pct),
    require_fundamentals=bool(require_fundamentals),
)

try:
    fdf = load_fundamentals_snapshot(fundamentals_file)
except Exception as exc:
    st.error(f"Failed to read fundamentals CSV: {exc}")
    fdf = None

tab1, tab2, tab3 = st.tabs(["1. Screen & Save", "2. Monitor Watchlist", "3. Saved Files"])

with tab1:
    st.subheader("Screen stocks on selected date")
    left, right = st.columns(2)

    with left:
        if st.button("Run screen", use_container_width=True):
            with st.spinner("Screening..."):
                start_download = (as_of_ts - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
                end_download = (as_of_ts + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                st.session_state["screen_df"] = screen_on_date(
                    tickers=tickers,
                    as_of_date=as_of_ts,
                    start_download=start_download,
                    end_download=end_download,
                    cfg=cfg,
                    fdf=fdf,
                )
                st.success("Screen completed.")

    with right:
        if st.button("Save passed list", use_container_width=True):
            screen_df = st.session_state.get("screen_df")
            if screen_df is None or screen_df.empty:
                st.warning("Run the screen first.")
            else:
                passed = screen_df[screen_df["pass"]].copy()
                if passed.empty:
                    st.warning("No passed stocks to save.")
                else:
                    path = save_watchlist(passed, as_of_ts)
                    st.success(f"Saved: {path.name}")

    screen_df = st.session_state.get("screen_df")
    if screen_df is not None and not screen_df.empty:
        st.dataframe(screen_df, use_container_width=True)
        passed = screen_df[screen_df["pass"]]
        st.markdown(f"**Passed stocks:** {len(passed)}")
        if not passed.empty:
            st.dataframe(passed[["ticker", "score", "technical_pass", "fundamental_pass", "close", "ret_63d"]], use_container_width=True)
    else:
        st.info("No screen result yet.")

with tab2:
    st.subheader("Monitor a saved watchlist")
    files = list_saved_watchlists()
    if not files:
        st.info("No saved watchlist files yet.")
    else:
        options = {p.name: p for p in files}
        selected_name = st.selectbox("Select saved watchlist", list(options.keys()))
        selected_path = options[selected_name]

        if st.button("Run monitor", use_container_width=True):
            st.session_state["monitor_df"] = monitor_watchlist(load_watchlist(selected_path), monitor_ts)
            st.session_state["monitor_selected_path"] = selected_path

        monitor_df = st.session_state.get("monitor_df")
        monitor_selected_path = st.session_state.get("monitor_selected_path")
        if monitor_df is not None and monitor_selected_path == selected_path:
            st.dataframe(monitor_df, use_container_width=True)

            buy_count = int((monitor_df["action"] == "BUY").sum())
            sell_count = int((monitor_df["action"] == "SELL").sum())
            hold_count = int((monitor_df["action"] == "HOLD").sum())
            watch_count = int((monitor_df["action"] == "WATCH").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BUY", buy_count)
            c2.metric("SELL", sell_count)
            c3.metric("HOLD", hold_count)
            c4.metric("WATCH", watch_count)

            if buy_count > 0:
                st.warning("Action alert: at least one stock is in BUY zone.")
            elif sell_count > 0:
                st.error("Action alert: at least one stock is in SELL zone.")
            else:
                st.info("No BUY/SELL alerts yet. Current rules are still placeholders.")
        else:
            st.info("Select a saved file and run monitor.")

with tab3:
    st.subheader("Saved watchlists")
    files = list_saved_watchlists()
    if not files:
        st.info("No saved files.")
    else:
        rows = []
        for p in files:
            try:
                df = pd.read_csv(p)
                rows.append({"file": p.name, "rows": len(df), "path": str(p)})
            except Exception:
                rows.append({"file": p.name, "rows": np.nan, "path": str(p)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.markdown(
    """
**Current behavior**
- Screens using your fundamental + technical template
- Saves passed list as `US stock dd-MMM-yyyy.csv`
- Monitors saved watchlists daily
- BUY / SELL / HOLD rules are placeholder-only for now

**Later**
- Once you provide buy/sell/hold zone rules, they can be added to:
  - `buy_zone_rule()`
  - `sell_zone_rule()`
  - `hold_zone_rule()`
"""
)
