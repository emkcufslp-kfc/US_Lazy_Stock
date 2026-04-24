
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
WATCHLIST_DIR = APP_DIR / "saved_watchlists"
try:
    WATCHLIST_DIR.mkdir(exist_ok=True, parents=True)
except Exception:
    pass  # read-only filesystem on some cloud hosts
SAMPLE_SCREEN_PATH = APP_DIR.parent / "data" / "full_screen_latest.csv"
AUTO_UNIVERSE_TOP_N = 500


@dataclass
class ScreenConfig:
    min_eps_yoy: float = 20.0
    min_revenue_yoy: float = 20.0
    min_roe_avg: float = 15.0
    high_52w_within_pct: float = 25.0
    low_52w_above_pct: float = 25.0
    require_fundamentals: bool = True
    require_close_above_mas: bool = True
    require_ma_stack: bool = True
    roe_stability_max_std: float = 40.0
    use_tradingview_setup: bool = True
    min_market_cap: float = 10_000_000_000.0
    fundamental_mode: str = "tv_ttm"
    apply_technical_filter: bool = False
    min_fundamental_rules_pass: int = 3


@dataclass
class WVFConfig:
    lookback_std_high: int = 22
    bollinger_length: int = 20
    bollinger_mult: float = 2.0
    lookback_percentile: int = 50
    high_percentile: float = 0.85
    low_percentile: float = 1.01


@st.cache_data(show_spinner=False)
def download_price_history(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=pd.Timestamp(end) + pd.Timedelta(days=5),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
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


def add_wvf_indicator(df: pd.DataFrame, wvf_cfg: WVFConfig) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns or "Low" not in out.columns:
        return out
    highest_close = out["Close"].rolling(wvf_cfg.lookback_std_high, min_periods=wvf_cfg.lookback_std_high).max()
    wvf = ((highest_close - out["Low"]) / highest_close) * 100.0
    sdev = wvf_cfg.bollinger_mult * wvf.rolling(wvf_cfg.bollinger_length, min_periods=wvf_cfg.bollinger_length).std()
    midline = wvf.rolling(wvf_cfg.bollinger_length, min_periods=wvf_cfg.bollinger_length).mean()
    upper_band = midline + sdev
    range_high = wvf.rolling(wvf_cfg.lookback_percentile, min_periods=wvf_cfg.lookback_percentile).max() * wvf_cfg.high_percentile
    range_low = wvf.rolling(wvf_cfg.lookback_percentile, min_periods=wvf_cfg.lookback_percentile).min() * wvf_cfg.low_percentile
    green = (wvf >= upper_band) | (wvf >= range_high)

    out["WVF"] = wvf
    out["WVF_UPPER_BAND"] = upper_band
    out["WVF_RANGE_HIGH"] = range_high
    out["WVF_RANGE_LOW"] = range_low
    out["WVF_GREEN"] = green.fillna(False)
    return out


def _normalize_ticker_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    return s[s.str.match(r"^[A-Z][A-Z0-9\\-]*$")]


def _fetch_reference_csv(url: str, required_cols: List[str], timeout_sec: int = 20) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "USLazyStock/1.0"})
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if not set(required_cols).issubset(set(df.columns)):
            return None
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_reference_txt_lines(url: str, timeout_sec: int = 20) -> List[str]:
    try:
        resp = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "USLazyStock/1.0"})
        resp.raise_for_status()
        lines = [x.strip().upper().replace(".", "-") for x in resp.text.splitlines() if x.strip()]
        lines = [x for x in lines if pd.Series([x]).str.match(r"^[A-Z][A-Z0-9\\-]*$").iloc[0]]
        return lines
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_sp500_profiles() -> pd.DataFrame:
    def _normalize(
        df: pd.DataFrame,
        symbol_col: str,
        name_col: str,
        sector_col: str,
        business_col: Optional[str] = None,
    ) -> pd.DataFrame:
        business_series = df[business_col] if business_col and business_col in df.columns else df[sector_col]
        out = pd.DataFrame({
            "ticker": df[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
            "company_name": df[name_col].astype(str).str.strip(),
            "sector": df[sector_col].astype(str).str.strip(),
            "business_nature": business_series.astype(str).str.strip(),
        })
        out = out.replace({"": np.nan}).dropna(subset=["ticker"])
        out["company_name"] = out["company_name"].fillna("N/A")
        out["sector"] = out["sector"].fillna("N/A")
        out["business_nature"] = out["business_nature"].fillna(out["sector"]).fillna("N/A")
        return out.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    errors: List[str] = []

    # Source 1: Wikipedia (with browser-like headers to reduce 403 risk)
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if tables and {"Symbol", "Security", "GICS Sector"}.issubset(set(tables[0].columns)):
            return _normalize(tables[0], "Symbol", "Security", "GICS Sector", "GICS Sub-Industry")
        errors.append("Wikipedia table missing required columns.")
    except Exception as exc:
        errors.append(f"Wikipedia: {exc}")

    # Source 2: DataHub CSV mirror
    try:
        csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(csv_url)
        if {"Symbol", "Name", "Sector"}.issubset(set(df.columns)):
            return _normalize(df, "Symbol", "Name", "Sector", "Industry")
        errors.append("DataHub CSV missing required columns.")
    except Exception as exc:
        errors.append(f"DataHub: {exc}")

    # Source 3 (reference-only fallback): GitHub datasets mirror
    gh_constituents = _fetch_reference_csv(
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        required_cols=["Symbol", "Security", "GICS Sector"],
    )
    if gh_constituents is not None:
        return _normalize(gh_constituents, "Symbol", "Security", "GICS Sector", "GICS Sub-Industry")
    errors.append("GitHub datasets/s-and-p-500-companies unavailable or invalid.")

    # Source 4 (reference-only fallback): GitHub S&P 500 financials mirror
    gh_financials = _fetch_reference_csv(
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/main/data/constituents-financials.csv",
        required_cols=["Symbol", "Name", "Sector"],
    )
    if gh_financials is not None:
        return _normalize(gh_financials, "Symbol", "Name", "Sector", "Sector")
    errors.append("GitHub datasets/s-and-p-500-companies-financials unavailable or invalid.")

    # Source 5 (reference-only fallback): community daily S&P 500 list
    gh_sp500 = _fetch_reference_csv(
        "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv",
        required_cols=["symbol", "name", "industry"],
    )
    if gh_sp500 is not None:
        return _normalize(gh_sp500, "symbol", "name", "industry", "industry")
    errors.append("GitHub Ate329/top-us-stock-tickers unavailable or invalid.")

    # Source 6: Small local fallback so app remains usable offline/restricted
    fallback = [
        ("AAPL", "Apple Inc.", "Information Technology", "Consumer Electronics"),
        ("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
        ("NVDA", "NVIDIA Corporation", "Information Technology", "Semiconductors"),
        ("AMZN", "Amazon.com, Inc.", "Consumer Discretionary", "Broadline Retail"),
        ("META", "Meta Platforms, Inc.", "Communication Services", "Interactive Media & Services"),
        ("GOOGL", "Alphabet Inc.", "Communication Services", "Interactive Media & Services"),
        ("TSLA", "Tesla, Inc.", "Consumer Discretionary", "Automobile Manufacturers"),
        ("AVGO", "Broadcom Inc.", "Information Technology", "Semiconductors"),
        ("JPM", "JPMorgan Chase & Co.", "Financials", "Diversified Banks"),
        ("XOM", "Exxon Mobil Corporation", "Energy", "Integrated Oil & Gas"),
        ("JBL", "Jabil Inc.", "Information Technology", "Electronic Manufacturing Services"),
        ("CLS", "Celestica Inc.", "Information Technology", "Electronic Manufacturing Services"),
    ]
    if fallback:
        return pd.DataFrame(fallback, columns=["ticker", "company_name", "sector", "business_nature"])

    raise ValueError("Unable to load S&P 500 constituents. " + " | ".join(errors))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_sp500_tickers() -> List[str]:
    return load_sp500_profiles()["ticker"].tolist()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_nasdaq_profiles() -> pd.DataFrame:
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep="|")
        if "Symbol" not in df.columns:
            return pd.DataFrame(columns=["ticker", "company_name", "sector", "business_nature"])
        df = df[df["Symbol"].astype(str).str.upper() != "FILE CREATION TIME:"].copy()
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() == "N"]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() == "N"]
        df["ticker"] = df["Symbol"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
        df = df[df["ticker"].str.match(r"^[A-Z][A-Z0-9\\-]*$")]
        name_col = "Security Name" if "Security Name" in df.columns else "ticker"
        out = pd.DataFrame({
            "ticker": df["ticker"],
            "company_name": df[name_col].astype(str).str.strip(),
            "sector": "N/A",
            "business_nature": "N/A",
        }).drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
        return out
    except Exception:
        pass

    # Reference-only fallback A: community daily list with sector metadata
    gh_all = _fetch_reference_csv(
        "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/all.csv",
        required_cols=["symbol", "name", "industry"],
    )
    if gh_all is not None:
        out = pd.DataFrame({
            "ticker": _normalize_ticker_series(gh_all["symbol"]),
            "company_name": gh_all["name"].astype(str).str.strip(),
            "sector": gh_all["industry"].astype(str).str.strip().replace("", "N/A"),
            "business_nature": gh_all["industry"].astype(str).str.strip().replace("", "N/A"),
        })
        out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
        if not out.empty:
            return out

    # Reference-only fallback B: symbol-only lists from US-Stock-Symbols
    urls = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt",
    ]
    tickers: List[str] = []
    for u in urls:
        tickers.extend(_fetch_reference_txt_lines(u))
    tickers = sorted(set(tickers))
    if tickers:
        return pd.DataFrame({
            "ticker": tickers,
            "company_name": ["N/A"] * len(tickers),
            "sector": ["N/A"] * len(tickers),
            "business_nature": ["N/A"] * len(tickers),
        })

    return pd.DataFrame(columns=["ticker", "company_name", "sector", "business_nature"])


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_combined_universe_profiles() -> pd.DataFrame:
    sp = load_sp500_profiles().copy()
    sp["source_rank"] = 0
    nq = load_nasdaq_profiles().copy()
    nq["source_rank"] = 1
    out = pd.concat([sp, nq], ignore_index=True)
    out["company_name"] = out["company_name"].fillna("N/A")
    out["sector"] = out["sector"].fillna("N/A")
    out["business_nature"] = out["business_nature"].fillna("N/A")
    out = out.sort_values(["source_rank", "ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    return out[["ticker", "company_name", "sector", "business_nature"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_community_all_profiles() -> pd.DataFrame:
    """Load sector/industry metadata for all US stocks from community CSV (Ate329/top-us-stock-tickers all.csv).
    This covers Nasdaq + NYSE + AMEX stocks with industry labels, filling gaps left by the NASDAQ FTP source."""
    df = _fetch_reference_csv(
        "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/all.csv",
        required_cols=["symbol", "name", "industry"],
    )
    if df is None:
        return pd.DataFrame(columns=["ticker", "company_name", "sector", "business_nature"])
    out = pd.DataFrame({
        "ticker": _normalize_ticker_series(df["symbol"]),
        "company_name": df["name"].astype(str).str.strip(),
        "sector": df["industry"].astype(str).str.strip().replace({"": "N/A", "nan": "N/A"}),
        "business_nature": df["industry"].astype(str).str.strip().replace({"": "N/A", "nan": "N/A"}),
    })
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return out


def build_profile_map(tickers: List[str]) -> Dict[str, Dict[str, str]]:
    # Layer 1: Combined universe (S&P 500 with sectors + Nasdaq with company names)
    prof = load_combined_universe_profiles()
    lookup = {
        str(r["ticker"]): {
            "company_name": str(r["company_name"]),
            "sector": str(r["sector"]),
            "business_nature": str(r.get("business_nature", "N/A")),
        }
        for _, r in prof.iterrows()
    }
    out: Dict[str, Dict[str, str]] = {}
    for t in tickers:
        item = lookup.get(t, {"company_name": "N/A", "sector": "N/A", "business_nature": "N/A"})
        out[t] = dict(item)  # ensure mutable copy

    # Layer 2: Enrich missing sector/industry from community CSV (covers all US exchanges with industry labels)
    needs_sector = [t for t in tickers if out[t].get("sector") == "N/A" or out[t].get("company_name") == "N/A"]
    if needs_sector:
        comm = load_community_all_profiles()
        comm_lookup = {str(r["ticker"]): r for _, r in comm.iterrows()}
        for t in needs_sector:
            row = comm_lookup.get(t)
            if row is not None:
                if out[t].get("company_name") == "N/A":
                    name = str(row.get("company_name", "")).strip()
                    if name and name != "nan":
                        out[t]["company_name"] = name
                if out[t].get("sector") == "N/A":
                    sec = str(row.get("sector", "")).strip()
                    if sec and sec not in ("nan", "N/A"):
                        out[t]["sector"] = sec
                if out[t].get("business_nature") == "N/A":
                    bn = str(row.get("business_nature", "")).strip()
                    if bn and bn not in ("nan", "N/A"):
                        out[t]["business_nature"] = bn

    # Layer 3: Fall back to yfinance ONLY for tickers still missing company_name
    # (avoids hundreds of API calls just for missing sector data)
    for t in tickers:
        if out[t].get("company_name") == "N/A":
            live = load_live_fundamentals_yf(t)
            if out[t].get("company_name") == "N/A":
                out[t]["company_name"] = str(live.get("company_name", "N/A"))
            if out[t].get("sector") == "N/A":
                out[t]["sector"] = str(live.get("sector", "N/A"))
            if out[t].get("business_nature") == "N/A":
                out[t]["business_nature"] = str(live.get("business_nature", "N/A"))
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_live_fundamentals_yf(ticker: str) -> Dict[str, object]:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    company_name = info.get("longName") or info.get("shortName") or "N/A"
    sector = info.get("sector") or "N/A"
    business_nature = info.get("industry") or sector or "N/A"
    eps_yoy_value = info.get("earningsQuarterlyGrowth")
    rev_yoy_value = info.get("revenueGrowth")
    roe_avg = info.get("returnOnEquity")
    revenue_value = info.get("totalRevenue")
    to_pct = lambda v: float(v) * 100.0 if v is not None and pd.notna(v) else np.nan
    return {
        "company_name": company_name,
        "sector": sector,
        "business_nature": business_nature,
        "eps_yoy_value": to_pct(eps_yoy_value),
        "rev_yoy_value": to_pct(rev_yoy_value),
        "roe_avg": to_pct(roe_avg),
        "revenue_value": float(revenue_value) if revenue_value is not None and pd.notna(revenue_value) else np.nan,
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_market_cap_yf(ticker: str) -> float:
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        if fi is not None:
            mc = fi.get("market_cap")
            if mc is not None and pd.notna(mc):
                return float(mc)
        info = tk.get_info()
        mc2 = info.get("marketCap")
        if mc2 is not None and pd.notna(mc2):
            return float(mc2)
    except Exception:
        pass
    return np.nan


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_tradingview_like_fundamentals_yf(ticker: str) -> Dict[str, object]:
    tk = yf.Ticker(ticker)
    qis = tk.quarterly_income_stmt
    qbs = tk.quarterly_balance_sheet

    if qis is None or qis.empty:
        return {}

    qis = qis.copy()
    qis.columns = pd.to_datetime(qis.columns)
    qis = qis.sort_index(axis=1)

    eps_row = "Diluted EPS" if "Diluted EPS" in qis.index else ("Basic EPS" if "Basic EPS" in qis.index else None)
    rev_row = "Total Revenue" if "Total Revenue" in qis.index else None

    eps_q = pd.to_numeric(qis.loc[eps_row], errors="coerce").dropna() if eps_row else pd.Series(dtype=float)
    rev_q = pd.to_numeric(qis.loc[rev_row], errors="coerce").dropna() if rev_row else pd.Series(dtype=float)

    eps_qoq = (eps_q.pct_change() * 100.0).dropna()
    rev_qoq = (rev_q.pct_change() * 100.0).dropna()

    roe_ttm_series = pd.Series(dtype=float)
    if qbs is not None and not qbs.empty and "Net Income" in qis.index and "Stockholders Equity" in qbs.index:
        qbs2 = qbs.copy()
        qbs2.columns = pd.to_datetime(qbs2.columns)
        qbs2 = qbs2.sort_index(axis=1)
        ni_q = pd.to_numeric(qis.loc["Net Income"], errors="coerce").dropna()
        eq_q = pd.to_numeric(qbs2.loc["Stockholders Equity"], errors="coerce").dropna()
        df = pd.DataFrame({"ni": ni_q, "eq": eq_q}).dropna().sort_index()
        vals = []
        idxs = []
        for i in range(3, len(df)):
            w = df.iloc[i - 3 : i + 1]
            ttm_ni = float(w["ni"].sum())
            avg_eq = float(w["eq"].mean())
            if avg_eq == 0:
                continue
            vals.append((ttm_ni / avg_eq) * 100.0)
            idxs.append(w.index[-1])
        if vals:
            roe_ttm_series = pd.Series(vals, index=idxs, dtype=float).sort_index()

    return {
        "eps_qoq_series": eps_qoq.tail(4).tolist(),
        "rev_qoq_series": rev_qoq.tail(4).tolist(),
        "roe_ttm_series": roe_ttm_series.tail(4).tolist(),
        "eps_qoq_value": float(eps_qoq.iloc[-1]) if not eps_qoq.empty else np.nan,
        "rev_qoq_value": float(rev_qoq.iloc[-1]) if not rev_qoq.empty else np.nan,
        "roe_fq_value": float(roe_ttm_series.iloc[-1]) if not roe_ttm_series.empty else np.nan,
        "revenue_value": float(rev_q.iloc[-1]) if not rev_q.empty else np.nan,
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_tv_match_fundamentals_yf(ticker: str, as_of_date_str: str) -> Dict[str, object]:
    """TradingView-equivalent fundamental metrics for a given as-of date.

    Metrics computed:
      - eps_ttm_yoy  : Diluted EPS TTM YoY %  = (sum last 4Q EPS / sum prior 4Q EPS - 1) × 100
      - rev_ttm_yoy  : Total Revenue TTM YoY % = (sum last 4Q rev / sum prior 4Q rev - 1) × 100
      - roe_ttm      : ROE TTM %               = (sum last 4Q net income / avg last 4Q equity) × 100

    Quarterly data is filtered to as_of_date so historical dates are supported.
    Falls back to single-quarter YoY when only 5 quarters are available.
    """
    as_of = pd.Timestamp(as_of_date_str)
    result: Dict[str, object] = {
        "eps_ttm_yoy": np.nan,
        "rev_ttm_yoy": np.nan,
        "roe_ttm": np.nan,
        "revenue_value": np.nan,
    }
    try:
        tk = yf.Ticker(ticker)
        qis = tk.quarterly_income_stmt
    except Exception:
        return result
    if qis is None or qis.empty:
        return result

    qis = qis.copy()
    qis.columns = pd.to_datetime(qis.columns)
    qis = qis.sort_index(axis=1)
    qis = qis.loc[:, qis.columns <= as_of]          # ← historical date filter

    eps_row = "Diluted EPS" if "Diluted EPS" in qis.index else ("Basic EPS" if "Basic EPS" in qis.index else None)
    rev_row = "Total Revenue" if "Total Revenue" in qis.index else None

    eps_q = pd.to_numeric(qis.loc[eps_row], errors="coerce").dropna() if eps_row else pd.Series(dtype=float)
    rev_q = pd.to_numeric(qis.loc[rev_row], errors="coerce").dropna() if rev_row else pd.Series(dtype=float)

    # TTM EPS YoY: (sum of last 4 quarters) / (sum of prior 4 quarters) − 1
    if len(eps_q) >= 8:
        c, p = float(eps_q.iloc[-4:].sum()), float(eps_q.iloc[-8:-4].sum())
        if p != 0 and pd.notna(p):
            result["eps_ttm_yoy"] = (c / p - 1.0) * 100.0
    elif len(eps_q) >= 5:
        # Fallback: most-recent quarter vs same quarter one year prior
        if eps_q.iloc[-5] != 0 and pd.notna(eps_q.iloc[-5]):
            result["eps_ttm_yoy"] = (float(eps_q.iloc[-1]) / float(eps_q.iloc[-5]) - 1.0) * 100.0

    # TTM Revenue YoY
    if len(rev_q) >= 8:
        c, p = float(rev_q.iloc[-4:].sum()), float(rev_q.iloc[-8:-4].sum())
        if p != 0 and pd.notna(p):
            result["rev_ttm_yoy"] = (c / p - 1.0) * 100.0
    elif len(rev_q) >= 5:
        if rev_q.iloc[-5] != 0 and pd.notna(rev_q.iloc[-5]):
            result["rev_ttm_yoy"] = (float(rev_q.iloc[-1]) / float(rev_q.iloc[-5]) - 1.0) * 100.0

    result["revenue_value"] = float(rev_q.iloc[-4:].sum()) if len(rev_q) >= 4 else (float(rev_q.iloc[-1]) if not rev_q.empty else np.nan)

    # TTM ROE: (sum last 4Q net income) / (avg last 4Q equity) × 100
    try:
        qbs = tk.quarterly_balance_sheet
        if qbs is not None and not qbs.empty and "Net Income" in qis.index and "Stockholders Equity" in qbs.index:
            qbs2 = qbs.copy()
            qbs2.columns = pd.to_datetime(qbs2.columns)
            qbs2 = qbs2.sort_index(axis=1)
            qbs2 = qbs2.loc[:, qbs2.columns <= as_of]
            ni_q = pd.to_numeric(qis.loc["Net Income"], errors="coerce").dropna()
            eq_q = pd.to_numeric(qbs2.loc["Stockholders Equity"], errors="coerce").dropna()
            df_roe = pd.DataFrame({"ni": ni_q, "eq": eq_q}).dropna().sort_index()
            if len(df_roe) >= 4:
                ttm_ni = float(df_roe["ni"].iloc[-4:].sum())
                avg_eq = float(df_roe["eq"].iloc[-4:].mean())
                if avg_eq != 0 and pd.notna(avg_eq):
                    result["roe_ttm"] = (ttm_ni / avg_eq) * 100.0
    except Exception:
        pass

    # ── Fallback to yfinance pre-computed info fields ──────────────────────
    # yfinance only returns the last 4 quarters of raw data, which is not
    # enough to compute TTM YoY (needs 8Q) or even single-Q YoY (needs 5Q).
    # yfinance's info API exposes the same pre-computed values TradingView uses:
    #   earningsQuarterlyGrowth → EPS most-recent-quarter YoY (≈ TV TTM EPS YoY)
    #   revenueGrowth           → Revenue TTM YoY
    #   returnOnEquity          → ROE TTM
    # These are used as the primary fill when raw-data computation yields NaN.
    if pd.isna(result["eps_ttm_yoy"]) or pd.isna(result["rev_ttm_yoy"]) or pd.isna(result["roe_ttm"]):
        try:
            info = tk.get_info()
            if pd.isna(result["eps_ttm_yoy"]):
                v = info.get("earningsQuarterlyGrowth")
                if v is not None and pd.notna(v):
                    result["eps_ttm_yoy"] = float(v) * 100.0
            if pd.isna(result["rev_ttm_yoy"]):
                v = info.get("revenueGrowth")
                if v is not None and pd.notna(v):
                    result["rev_ttm_yoy"] = float(v) * 100.0
            if pd.isna(result["roe_ttm"]):
                v = info.get("returnOnEquity")
                if v is not None and pd.notna(v):
                    result["roe_ttm"] = float(v) * 100.0
            if pd.isna(result["revenue_value"]):
                v = info.get("totalRevenue")
                if v is not None and pd.notna(v):
                    result["revenue_value"] = float(v)
        except Exception:
            pass

    return result


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_gemini_like_fundamentals_yf(ticker: str) -> Dict[str, object]:
    tk = yf.Ticker(ticker)
    qis = tk.quarterly_income_stmt
    if qis is None or qis.empty:
        return {}

    qis = qis.copy()
    qis.columns = pd.to_datetime(qis.columns)
    qis = qis.sort_index(axis=1)

    eps_row = "Diluted EPS" if "Diluted EPS" in qis.index else ("Basic EPS" if "Basic EPS" in qis.index else None)
    rev_row = "Total Revenue" if "Total Revenue" in qis.index else None

    eps_q = pd.to_numeric(qis.loc[eps_row], errors="coerce").dropna() if eps_row else pd.Series(dtype=float)
    rev_q = pd.to_numeric(qis.loc[rev_row], errors="coerce").dropna() if rev_row else pd.Series(dtype=float)

    eps_yoy = (eps_q / eps_q.shift(4) - 1.0) * 100.0 if len(eps_q) >= 5 else pd.Series(dtype=float)
    rev_yoy = (rev_q / rev_q.shift(4) - 1.0) * 100.0 if len(rev_q) >= 5 else pd.Series(dtype=float)
    eps_yoy = eps_yoy.dropna()
    rev_yoy = rev_yoy.dropna()

    # ROE 3Y: annual Net Income / Stockholders Equity
    ai = tk.income_stmt
    ab = tk.balance_sheet
    roe_year_vals = pd.Series(dtype=float)
    if ai is not None and not ai.empty and ab is not None and not ab.empty and "Net Income" in ai.index and "Stockholders Equity" in ab.index:
        ai2 = ai.copy()
        ab2 = ab.copy()
        ai2.columns = pd.to_datetime(ai2.columns)
        ab2.columns = pd.to_datetime(ab2.columns)
        ni = pd.to_numeric(ai2.loc["Net Income"], errors="coerce")
        eq = pd.to_numeric(ab2.loc["Stockholders Equity"], errors="coerce")
        tmp = pd.DataFrame({"ni": ni, "eq": eq}).dropna()
        tmp = tmp[tmp["eq"] != 0].sort_index()
        if not tmp.empty:
            roe_year_vals = (tmp["ni"] / tmp["eq"]) * 100.0

    eps_list = eps_yoy.tail(4).tolist()
    rev_list = rev_yoy.tail(4).tolist()
    roe_list = roe_year_vals.tail(3).tolist()
    revenue_value = float(rev_q.iloc[-1]) if not rev_q.empty else np.nan

    # Backfill from SEC when Yahoo quarterly history is too short for 3~4 YoY checks.
    sec = load_sec_fundamentals_for_rules(ticker, pd.Timestamp("today").normalize().strftime("%Y-%m-%d"))
    if len(eps_list) < 3:
        sec_eps = sec.get("eps_yoy_list", []) if isinstance(sec, dict) else []
        if sec_eps:
            eps_list = list(pd.Series(sec_eps, dtype=float).tail(4).values)
    if len(rev_list) < 3:
        sec_rev = sec.get("rev_yoy_list", []) if isinstance(sec, dict) else []
        if sec_rev:
            rev_list = list(pd.Series(sec_rev, dtype=float).tail(4).values)
    if len(roe_list) < 3:
        sec_roe = sec.get("roe_year_values", []) if isinstance(sec, dict) else []
        if sec_roe:
            roe_list = list(pd.Series(sec_roe, dtype=float).tail(3).values)
    if pd.isna(revenue_value):
        sec_rev_fy = sec.get("rev_fy_list", []) if isinstance(sec, dict) else []
        if sec_rev_fy:
            revenue_value = float(pd.Series(sec_rev_fy, dtype=float).dropna().iloc[-1])

    eps_s = pd.Series(eps_list, dtype=float)
    rev_s = pd.Series(rev_list, dtype=float)
    roe_s = pd.Series(roe_list, dtype=float)

    return {
        "eps_yoy_series": eps_list,
        "rev_yoy_series": rev_list,
        "roe_year_series": roe_list,
        "eps_yoy_value": float(eps_s.iloc[-1]) if not eps_s.empty else np.nan,
        "rev_yoy_value": float(rev_s.iloc[-1]) if not rev_s.empty else np.nan,
        "roe_avg": float(roe_s.mean()) if not roe_s.empty else np.nan,
        "revenue_value": revenue_value,
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_sec_ticker_cik_map() -> Dict[str, int]:
    out: Dict[str, int] = {}

    # Primary source: official SEC ticker map
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "USLazyStock/1.0 (free-audit)"}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        for _, item in data.items():
            t = str(item.get("ticker", "")).upper().strip()
            cik = item.get("cik_str")
            if t and cik:
                out[t] = int(cik)
        if out:
            return out
    except Exception:
        pass

    # Fallback source: free public mirror (sec-cik-mapper)
    try:
        mirror = (
            "https://raw.githubusercontent.com/jadchaar/sec-cik-mapper/main/"
            "mappings/stocks/ticker_to_cik.json"
        )
        resp = requests.get(mirror, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        for t, cik in data.items():
            t2 = str(t).upper().strip()
            try:
                out[t2] = int(cik)
            except Exception:
                continue
    except Exception:
        pass

    return out


def _sec_pick_latest_yoy(facts: dict, tags: List[str], as_of_date: pd.Timestamp) -> float:
    yoy = _sec_collect_recent_quarter_yoy(facts, tags, as_of_date)
    return float(yoy[-1]) if yoy else np.nan


def _sec_iter_taxonomies(facts: dict) -> List[dict]:
    out = []
    if isinstance(facts.get("us-gaap"), dict):
        out.append(facts["us-gaap"])
    if isinstance(facts.get("ifrs-full"), dict):
        out.append(facts["ifrs-full"])
    return out


def _sec_collect_recent_quarter_yoy(
    facts: dict,
    tags: List[str],
    as_of_date: pd.Timestamp,
    max_points: int = 4,
) -> List[float]:
    def _pick_single_quarter_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d["filed"] = pd.to_datetime(d.get("filed"), errors="coerce")
        d["frame"] = d.get("frame", "").astype(str)
        d["has_q_frame"] = d["frame"].str.match(r"^CY\\d{4}Q[1-4]$")
        d["cal_year"] = d["end"].dt.year
        # Prefer frame year when present.
        frame_year = pd.to_numeric(d["frame"].str.extract(r"^CY(\\d{4})Q[1-4]")[0], errors="coerce")
        d.loc[frame_year.notna(), "cal_year"] = frame_year[frame_year.notna()].astype(int)
        if "start" in d.columns:
            d["start"] = pd.to_datetime(d.get("start"), errors="coerce")
            d["dur_days"] = (d["end"] - d["start"]).dt.days
        else:
            d["dur_days"] = np.nan

        # Keep one row per quarter endpoint:
        # prefer explicit quarterly frame, then shorter duration (~90d), then latest filing.
        picked = []
        for _, g in d.groupby(["end", "fp"], dropna=False):
            g2 = g.copy()
            g2["dur_rank"] = g2["dur_days"].fillna(9999)
            g2 = g2.sort_values(["has_q_frame", "dur_rank", "filed"], ascending=[False, True, False])
            picked.append(g2.iloc[0])
        out = pd.DataFrame(picked).sort_values("end")
        return out

    for tax in _sec_iter_taxonomies(facts):
        for tag in tags:
            units = ((tax.get(tag) or {}).get("units") or {})
            series = units.get("USD/shares") or units.get("USD")
            if not series:
                continue
            df = pd.DataFrame(series)
            if df.empty or "end" not in df.columns or "val" not in df.columns:
                continue
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            df["fy"] = pd.to_numeric(df.get("fy"), errors="coerce")
            df["fp"] = df.get("fp", "").astype(str)
            df = df.dropna(subset=["end", "val", "fy"])
            df = df[(df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])) & (df["end"] <= as_of_date)]
            if df.empty:
                continue
            df = _pick_single_quarter_rows(df)
            yoy_vals: List[float] = []
            for _, latest in df.iterrows():
                prior = df[(df["cal_year"] == latest["cal_year"] - 1) & (df["fp"] == latest["fp"])]
                if prior.empty:
                    continue
                pv = float(prior.sort_values("end").iloc[-1]["val"])
                lv = float(latest["val"])
                if pv == 0:
                    continue
                yoy_vals.append((lv / pv - 1.0) * 100.0)
            if yoy_vals:
                return yoy_vals[-max_points:]
    return []


def _sec_collect_recent_fy_values(
    facts: dict,
    tags: List[str],
    as_of_date: pd.Timestamp,
    max_points: int = 3,
) -> List[float]:
    for tax in _sec_iter_taxonomies(facts):
        for tag in tags:
            units = ((tax.get(tag) or {}).get("units") or {})
            series = units.get("USD")
            if not series:
                continue
            df = pd.DataFrame(series)
            if df.empty or "end" not in df.columns or "val" not in df.columns:
                continue
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            df["fy"] = pd.to_numeric(df.get("fy"), errors="coerce")
            df["fp"] = df.get("fp", "").astype(str)
            df = df.dropna(subset=["end", "val", "fy"])
            df = df[(df["fp"] == "FY") & (df["end"] <= as_of_date)].sort_values("fy")
            if df.empty:
                continue
            vals = df["val"].astype(float).tolist()
            if vals:
                return vals[-max_points:]
    return []


def _sec_pick_roe_avg(facts: dict, as_of_date: pd.Timestamp) -> float:
    roe_vals = _sec_collect_roe_year_values(facts, as_of_date, max_points=3)
    if not roe_vals:
        return np.nan
    return float(pd.Series(roe_vals, dtype=float).mean())


def _sec_collect_roe_year_values(facts: dict, as_of_date: pd.Timestamp, max_points: int = 3) -> List[float]:
    for tax in _sec_iter_taxonomies(facts):
        ni_tag_candidates = ["NetIncomeLoss", "ProfitLoss"]
        eq_tag_candidates = ["StockholdersEquity", "Equity"]
        ni_units = None
        eq_units = None
        for t in ni_tag_candidates:
            ni_units = ((tax.get(t) or {}).get("units") or {}).get("USD")
            if ni_units:
                break
        for t in eq_tag_candidates:
            eq_units = ((tax.get(t) or {}).get("units") or {}).get("USD")
            if eq_units:
                break
        if not ni_units or not eq_units:
            continue
        ni = pd.DataFrame(ni_units)
        eq = pd.DataFrame(eq_units)
        if ni.empty or eq.empty:
            continue
        for d in [ni, eq]:
            d["end"] = pd.to_datetime(d["end"], errors="coerce")
            d["val"] = pd.to_numeric(d["val"], errors="coerce")
            d["fy"] = pd.to_numeric(d.get("fy"), errors="coerce")
        ni = ni.dropna(subset=["end", "val", "fy"])
        eq = eq.dropna(subset=["end", "val", "fy"])
        ni = ni[(ni["fp"] == "FY") & (ni["end"] <= as_of_date)]
        eq = eq[(eq["fp"] == "FY") & (eq["end"] <= as_of_date)]
        if ni.empty or eq.empty:
            continue
        merged = ni[["fy", "val"]].rename(columns={"val": "net_income"}).merge(
            eq[["fy", "val"]].rename(columns={"val": "equity"}), on="fy", how="inner"
        )
        merged = merged[(merged["equity"] != 0)].sort_values("fy").tail(3)
        if merged.empty:
            continue
        roe = ((merged["net_income"] / merged["equity"]) * 100.0).astype(float).tolist()
        if roe:
            return roe[-max_points:]
    return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_sec_fundamentals_for_rules(ticker: str, as_of_date: str) -> Dict[str, object]:
    try:
        cik_map = load_sec_ticker_cik_map()
        cik = cik_map.get(ticker.upper())
        if not cik:
            return {}
        cik10 = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        headers = {"User-Agent": "USLazyStock/1.0 (free-fundamentals)"}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        facts = resp.json().get("facts", {})
        as_of = pd.Timestamp(as_of_date)
        eps_yoy_list = _sec_collect_recent_quarter_yoy(
            facts, ["EarningsPerShareDiluted", "EarningsPerShareBasic", "DilutedEarningsLossPerShare", "BasicEarningsLossPerShare"], as_of
        )
        rev_yoy_list = _sec_collect_recent_quarter_yoy(
            facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenue"], as_of
        )
        rev_fy_list = _sec_collect_recent_fy_values(
            facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenue"], as_of
        )
        roe_year_values = _sec_collect_roe_year_values(facts, as_of, max_points=3)
        roe_avg = _sec_pick_roe_avg(facts, as_of)
        return {
            "eps_yoy_list": eps_yoy_list,
            "rev_yoy_list": rev_yoy_list,
            "rev_fy_list": rev_fy_list,
            "roe_year_values": roe_year_values,
            "roe_avg": roe_avg,
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_sec_fundamentals_audit(ticker: str, as_of_date: str) -> Dict[str, float]:
    try:
        cik_map = load_sec_ticker_cik_map()
        cik = cik_map.get(ticker.upper())
        if not cik:
            return {"eps_yoy_sec": np.nan, "rev_yoy_sec": np.nan, "roe_avg_sec": np.nan}
        cik10 = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        headers = {"User-Agent": "USLazyStock/1.0 (free-audit)"}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        facts = resp.json().get("facts", {})
        as_of = pd.Timestamp(as_of_date)
        eps_yoy_sec = _sec_pick_latest_yoy(facts, ["EarningsPerShareDiluted", "EarningsPerShareBasic"], as_of)
        rev_yoy_sec = _sec_pick_latest_yoy(
            facts,
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
            as_of,
        )
        roe_avg_sec = _sec_pick_roe_avg(facts, as_of)
        return {"eps_yoy_sec": eps_yoy_sec, "rev_yoy_sec": rev_yoy_sec, "roe_avg_sec": roe_avg_sec}
    except Exception:
        return {"eps_yoy_sec": np.nan, "rev_yoy_sec": np.nan, "roe_avg_sec": np.nan}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def top_volume_universe(as_of_date: str, top_n: int = AUTO_UNIVERSE_TOP_N, lookback_days: int = 60) -> Tuple[List[str], pd.DataFrame]:
    profiles = load_combined_universe_profiles()
    tickers = profiles["ticker"].tolist()
    start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=max(lookback_days * 2, 150))).strftime("%Y-%m-%d")
    end = (pd.Timestamp(as_of_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError("Failed to download volume data for universe selection.")

    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            tdf = raw[t].copy()
            tdf.columns = [str(c).title() for c in tdf.columns]
            vols = pd.to_numeric(tdf.get("Volume"), errors="coerce").dropna()
            if vols.empty:
                continue
            recent = vols.tail(lookback_days)
            if recent.empty:
                continue
            rows.append({"ticker": t, "avg_volume": float(recent.mean())})
    else:
        single = tickers[0] if tickers else ""
        tdf = raw.copy()
        tdf.columns = [str(c).title() for c in tdf.columns]
        vols = pd.to_numeric(tdf.get("Volume"), errors="coerce").dropna().tail(lookback_days)
        if not vols.empty and single:
            rows.append({"ticker": single, "avg_volume": float(vols.mean())})

    vol_df = pd.DataFrame(rows)
    if vol_df.empty:
        raise ValueError("No valid volume series found for universe selection.")
    vol_df = vol_df.sort_values("avg_volume", ascending=False).head(top_n).reset_index(drop=True)
    vol_df = vol_df.merge(profiles, on="ticker", how="left")
    vol_df["company_name"] = vol_df["company_name"].fillna("N/A")
    vol_df["sector"] = vol_df["sector"].fillna("N/A")
    vol_df["business_nature"] = vol_df["business_nature"].fillna(vol_df["sector"]).fillna("N/A")
    return vol_df["ticker"].tolist(), vol_df


def apply_always_include(base_tickers: List[str], include_tickers: List[str], top_n: int) -> List[str]:
    include_clean = [t.strip().upper() for t in include_tickers if t and t.strip()]
    include_clean = list(dict.fromkeys(include_clean))
    merged = include_clean + [t for t in base_tickers if t not in include_clean]
    return merged[:top_n]


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
    # ── TradingView match mode (default) ────────────────────────────────────
    # Replicates TV screener exactly:
    #   EPS Diluted TTM YoY > min_eps_yoy
    #   Revenue TTM YoY     > min_revenue_yoy
    #   ROE TTM             > min_roe_avg
    #   Market Cap          > min_market_cap
    # ALL 3 rules must pass (min_fundamental_rules_pass=3) to match TV AND logic.
    if cfg.fundamental_mode == "tv_ttm":
        tv = load_tv_match_fundamentals_yf(ticker, as_of_date.strftime("%Y-%m-%d"))
        eps_ttm_yoy = tv.get("eps_ttm_yoy", np.nan)
        rev_ttm_yoy = tv.get("rev_ttm_yoy", np.nan)
        roe_ttm     = tv.get("roe_ttm", np.nan)

        eps_pass = bool(pd.notna(eps_ttm_yoy) and float(eps_ttm_yoy) >= cfg.min_eps_yoy)
        rev_pass = bool(pd.notna(rev_ttm_yoy) and float(rev_ttm_yoy) >= cfg.min_revenue_yoy)
        roe_pass = bool(pd.notna(roe_ttm)     and float(roe_ttm)     >= cfg.min_roe_avg)

        market_cap_value = load_market_cap_yf(ticker)
        market_cap_pass  = bool(pd.notna(market_cap_value) and market_cap_value >= cfg.min_market_cap)
        strict_depth_ok  = pd.notna(market_cap_value)

        fundamental_rules_hit = int(eps_pass) + int(rev_pass) + int(roe_pass)
        fund_final_pass = bool(
            fundamental_rules_hit >= int(cfg.min_fundamental_rules_pass)
            and market_cap_pass
            and strict_depth_ok
        )
        if not cfg.require_fundamentals:
            fund_final_pass = True

        return fund_final_pass, {
            "fundamentals_used": True,
            "fundamentals_source": "tradingview_ttm_match",
            "eps_pass": eps_pass,
            "rev_pass": rev_pass,
            "roe_pass": roe_pass,
            "market_cap_pass": market_cap_pass,
            "roe_avg": roe_ttm,
            "eps_yoy_value": eps_ttm_yoy,
            "rev_yoy_value": rev_ttm_yoy,
            "revenue_value": tv.get("revenue_value", np.nan),
            "market_cap_value": market_cap_value,
            "eps_ge_min_qtrs": 1 if eps_pass else 0,
            "rev_ge_min_qtrs": 1 if rev_pass else 0,
            "eps_consec_qtrs": 1 if eps_pass else 0,
            "rev_consec_qtrs": 1 if rev_pass else 0,
            "annual_revenue_uptrend": np.nan,
            "roe_std": np.nan,
            "roe_stable": True,
            "roe_each_ge_min": roe_pass,
            "roe_quality": "better(>17)" if pd.notna(roe_ttm) and float(roe_ttm) > 17 else "ok(>15)",
            "eps_vals": [eps_ttm_yoy],
            "rev_vals": [rev_ttm_yoy],
            "fundamental_rules_hit": fundamental_rules_hit,
            "note": f"TradingView match: TTM EPS YoY + TTM Revenue YoY + TTM ROE (at least {cfg.min_fundamental_rules_pass} must pass)",
        }

    if cfg.fundamental_mode == "gemini_yoy":
        gm = load_gemini_like_fundamentals_yf(ticker)
        eps_yoy = pd.Series(gm.get("eps_yoy_series", []), dtype=float)
        rev_yoy = pd.Series(gm.get("rev_yoy_series", []), dtype=float)
        roe_year = pd.Series(gm.get("roe_year_series", []), dtype=float)

        def _max_consecutive_ge(series: pd.Series, threshold: float) -> int:
            run = 0
            best = 0
            for v in series.dropna().tolist():
                run = run + 1 if float(v) >= threshold else 0
                if run > best:
                    best = run
            return best

        eps_ge_min_qtrs = int((eps_yoy >= cfg.min_eps_yoy).sum()) if not eps_yoy.empty else 0
        rev_ge_min_qtrs = int((rev_yoy >= cfg.min_revenue_yoy).sum()) if not rev_yoy.empty else 0
        eps_consec_qtrs = _max_consecutive_ge(eps_yoy, cfg.min_eps_yoy)
        rev_consec_qtrs = _max_consecutive_ge(rev_yoy, cfg.min_revenue_yoy)

        # Relaxed growth thresholds:
        # Prioritize latest quarter momentum, otherwise fall back to broader continuity checks.
        latest_eps_yoy = float(eps_yoy.iloc[-1]) if not eps_yoy.empty else np.nan
        latest_rev_yoy = float(rev_yoy.iloc[-1]) if not rev_yoy.empty else np.nan
        eps_pass = bool(
            (pd.notna(latest_eps_yoy) and latest_eps_yoy >= cfg.min_eps_yoy)
            or (len(eps_yoy) >= 3 and eps_ge_min_qtrs >= 2 and eps_consec_qtrs >= 1)
        )
        rev_pass = bool(
            (pd.notna(latest_rev_yoy) and latest_rev_yoy >= cfg.min_revenue_yoy)
            or (len(rev_yoy) >= 3 and rev_ge_min_qtrs >= 2 and rev_consec_qtrs >= 1)
        )

        roe_avg = float(roe_year.tail(3).mean()) if len(roe_year) >= 1 else np.nan
        roe_year_tail = roe_year.tail(3)
        roe_ge_min_years = int((roe_year_tail >= cfg.min_roe_avg).sum()) if len(roe_year_tail) >= 1 else 0
        roe_each_ge_min = bool(len(roe_year_tail) >= 3 and roe_ge_min_years >= 1)
        roe_std = float(roe_year.tail(3).std()) if len(roe_year) >= 2 else np.nan
        roe_stable = bool(pd.notna(roe_std) and roe_std <= cfg.roe_stability_max_std)
        roe_pass = bool(pd.notna(roe_avg) and roe_avg >= cfg.min_roe_avg and roe_each_ge_min and roe_stable)

        market_cap_value = load_market_cap_yf(ticker)
        market_cap_pass = bool(pd.notna(market_cap_value) and market_cap_value >= cfg.min_market_cap)

        strict_depth_ok = pd.notna(market_cap_value)
        fundamental_rules_hit = int(bool(eps_pass)) + int(bool(rev_pass)) + int(bool(roe_pass))
        fund_final_pass = bool(fundamental_rules_hit >= int(cfg.min_fundamental_rules_pass) and market_cap_pass and strict_depth_ok)
        if not cfg.require_fundamentals:
            fund_final_pass = True

        return fund_final_pass, {
            "fundamentals_used": True,
            "fundamentals_source": "gemini_yoy_yf",
            "eps_pass": bool(eps_pass),
            "rev_pass": bool(rev_pass),
            "roe_pass": bool(roe_pass),
            "market_cap_pass": bool(market_cap_pass),
            "roe_avg": roe_avg,
            "eps_yoy_value": gm.get("eps_yoy_value", np.nan),
            "rev_yoy_value": gm.get("rev_yoy_value", np.nan),
            "revenue_value": gm.get("revenue_value", np.nan),
            "market_cap_value": market_cap_value,
            "eps_ge_min_qtrs": eps_ge_min_qtrs,
            "rev_ge_min_qtrs": rev_ge_min_qtrs,
            "eps_consec_qtrs": eps_consec_qtrs,
            "rev_consec_qtrs": rev_consec_qtrs,
            "annual_revenue_uptrend": np.nan,
            "roe_std": roe_std,
            "roe_stable": roe_stable,
            "roe_each_ge_min": roe_each_ge_min,
            "roe_quality": "better(>17)" if pd.notna(roe_avg) and roe_avg > 17 else "ok(>15)",
            "eps_vals": eps_yoy.tolist(),
            "rev_vals": rev_yoy.tolist(),
            "fundamental_rules_hit": fundamental_rules_hit,
            "note": "Gemini match setup (relaxed): YoY EPS/Revenue continuity + 3Y ROE + market cap filter",
        }

    tv = load_tradingview_like_fundamentals_yf(ticker)
    eps_qoq = pd.Series(tv.get("eps_qoq_series", []), dtype=float)
    rev_qoq = pd.Series(tv.get("rev_qoq_series", []), dtype=float)
    roe_ttm = pd.Series(tv.get("roe_ttm_series", []), dtype=float)

    eps_ge_min_qtrs = int((eps_qoq >= cfg.min_eps_yoy).sum()) if not eps_qoq.empty else 0
    rev_ge_min_qtrs = int((rev_qoq >= cfg.min_revenue_yoy).sum()) if not rev_qoq.empty else 0
    eps_consec_qtrs = 0
    rev_consec_qtrs = 0
    run = 0
    for v in eps_qoq.tolist():
        run = run + 1 if v >= cfg.min_eps_yoy else 0
        eps_consec_qtrs = max(eps_consec_qtrs, run)
    run = 0
    for v in rev_qoq.tolist():
        run = run + 1 if v >= cfg.min_revenue_yoy else 0
        rev_consec_qtrs = max(rev_consec_qtrs, run)

    # User criteria (relaxed):
    # - EPS growth > min threshold for at least 2 consecutive quarters
    # - Revenue growth > min threshold for at least 2 consecutive quarters, with >=2 hits in recent 3~4 quarters
    latest_eps_qoq = float(eps_qoq.iloc[-1]) if not eps_qoq.empty else np.nan
    latest_rev_qoq = float(rev_qoq.iloc[-1]) if not rev_qoq.empty else np.nan
    eps_pass = bool(
        (pd.notna(latest_eps_qoq) and latest_eps_qoq >= cfg.min_eps_yoy)
        or (len(eps_qoq) >= 2 and eps_consec_qtrs >= 2 and eps_ge_min_qtrs >= 2)
    )
    rev_pass = bool(
        (pd.notna(latest_rev_qoq) and latest_rev_qoq >= cfg.min_revenue_yoy)
        or (len(rev_qoq) >= 3 and rev_consec_qtrs >= 2 and rev_ge_min_qtrs >= 2)
    )

    roe_avg = float(roe_ttm.mean()) if not roe_ttm.empty else np.nan
    roe_std = float(roe_ttm.std()) if len(roe_ttm) >= 2 else np.nan
    roe_ttm_tail = roe_ttm.tail(3)
    roe_ge_min_years = int((roe_ttm_tail >= cfg.min_roe_avg).sum()) if len(roe_ttm_tail) >= 1 else 0
    roe_each_ge_min = bool(len(roe_ttm_tail) >= 3 and roe_ge_min_years >= 1)
    roe_stable = bool(pd.notna(roe_std) and roe_std <= cfg.roe_stability_max_std)
    roe_pass = bool(pd.notna(roe_avg) and roe_avg >= cfg.min_roe_avg and roe_each_ge_min and roe_stable)

    market_cap_value = load_market_cap_yf(ticker)
    market_cap_pass = bool(pd.notna(market_cap_value) and market_cap_value >= cfg.min_market_cap)

    strict_depth_ok = pd.notna(market_cap_value)
    fundamental_rules_hit = int(bool(eps_pass)) + int(bool(rev_pass)) + int(bool(roe_pass))
    fund_final_pass = bool(fundamental_rules_hit >= int(cfg.min_fundamental_rules_pass) and market_cap_pass and strict_depth_ok)
    if not cfg.require_fundamentals:
        fund_final_pass = True

    return fund_final_pass, {
        "fundamentals_used": True,
        "fundamentals_source": "tradingview_yf_qoq",
        "eps_pass": bool(eps_pass),
        "rev_pass": bool(rev_pass),
        "roe_pass": bool(roe_pass),
        "market_cap_pass": bool(market_cap_pass),
        "roe_avg": roe_avg,
        "eps_yoy_value": tv.get("eps_qoq_value", np.nan),
        "rev_yoy_value": tv.get("rev_qoq_value", np.nan),
        "revenue_value": tv.get("revenue_value", np.nan),
        "market_cap_value": market_cap_value,
        "eps_ge_min_qtrs": eps_ge_min_qtrs,
        "rev_ge_min_qtrs": rev_ge_min_qtrs,
        "eps_consec_qtrs": eps_consec_qtrs,
        "rev_consec_qtrs": rev_consec_qtrs,
        "annual_revenue_uptrend": np.nan,
        "roe_std": roe_std,
        "roe_stable": roe_stable,
        "roe_each_ge_min": roe_each_ge_min,
        "roe_quality": "better(>17)" if pd.notna(roe_avg) and roe_avg > 17 else "ok(>15)",
        "eps_vals": eps_qoq.tolist(),
        "rev_vals": rev_qoq.tolist(),
        "fundamental_rules_hit": fundamental_rules_hit,
        "note": "TradingView setup (relaxed): EPS/Revenue QoQ continuity + ROE + Market Cap > min",
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def stooq_close_on_or_before(ticker: str, as_of_date: str) -> float:
    as_of_ts = pd.Timestamp(as_of_date)
    candidates = [f"{ticker.lower()}.us", ticker.lower()]
    for symbol in candidates:
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            if df.empty or "Date" not in df.columns or "Close" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
            temp = df[df["Date"] <= as_of_ts]
            if temp.empty:
                continue
            return float(temp.iloc[-1]["Close"])
        except Exception:
            continue
    return np.nan


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

    ma_rules_pass = True
    if cfg.require_close_above_mas:
        ma_rules_pass = ma_rules_pass and above_mas
    if cfg.require_ma_stack:
        ma_rules_pass = ma_rules_pass and ma_stack

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
        "ma_rules_pass": bool(ma_rules_pass),
        "ret_63d": float(row.get("RET_63D", np.nan)),
    }
    return bool(ma_rules_pass and within_high and above_low), out


def buy_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> Tuple[bool, str]:
    temp = df[df.index <= now_date]
    if temp.empty:
        return False, ""

    row = temp.iloc[-1]
    close = float(row.get("Close", 0))
    ma50 = float(row.get("MA50", 0))
    ma150 = float(row.get("MA150", 0))
    ma200 = float(row.get("MA200", 0))
    high_52w = float(row.get("HIGH_52W", 0))

    # 1. Trend Breakout: MA Stack + Close near 52W High (requires longer history)
    if len(temp) >= 200 and ma50 > ma150 > ma200 and close > ma50:
        if high_52w > 0 and close >= high_52w * 0.95:
            return True, "Bullish crossover + near 52W High breakout."

    # 2. Bottoming Signal: WVF Green in last 3 days with trend guard
    recent_3 = temp.tail(3)
    if "WVF_GREEN" in recent_3.columns and bool(recent_3["WVF_GREEN"].any()):
        has_ma200 = pd.notna(ma200) and ma200 > 0
        if has_ma200 and close > ma200:
            return True, "WVF bottom signal detected while above primary trend (MA200)."
        if (not has_ma200) and pd.notna(ma50) and ma50 > 0 and close > ma50:
            return True, "WVF bottom signal detected (MA200 unavailable; using MA50 trend guard)."

    return False, ""


def sell_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> Tuple[bool, str]:
    temp = df[df.index <= now_date]
    if temp.empty:
        return False, ""

    row = temp.iloc[-1]
    close = float(row.get("Close", 0))
    ma50 = float(row.get("MA50", 0))
    ma150 = float(row.get("MA150", 0))
    ma200 = float(row.get("MA200", 0))

    if ma150 > 0 and close < ma150:
        if ma200 > 0 and close < ma200:
            return True, "Price broke below MA200 (Major Trend Exit)."
        return True, "Price broke below MA150 (Intermediate Trend Exit)."

    return False, ""


def hold_zone_rule(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> Tuple[bool, str]:
    temp = df[df.index <= now_date]
    if temp.empty:
        return False, ""

    row = temp.iloc[-1]
    close = float(row.get("Close", 0))
    ma200 = float(row.get("MA200", 0))

    if ma200 > 0 and close > ma200:
        return True, "Holding above primary trend (MA200)."

    return False, ""


def derive_action_status(ticker: str, df: pd.DataFrame, now_date: pd.Timestamp) -> Tuple[str, str]:
    is_sell, sell_note = sell_zone_rule(ticker, df, now_date)
    if is_sell:
        return "SELL", sell_note

    is_buy, buy_note = buy_zone_rule(ticker, df, now_date)
    if is_buy:
        return "BUY", buy_note

    is_hold, hold_note = hold_zone_rule(ticker, df, now_date)
    if is_hold:
        return "HOLD", hold_note

    return "WATCH", "Awaiting trend setup or signal."


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


@st.cache_data(show_spinner=False)
def load_sample_screen_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_SCREEN_PATH)
    if "pass" in df.columns:
        raw = df["pass"]
        if raw.dtype == bool:
            df["pass"] = raw
        else:
            df["pass"] = raw.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df


def screen_on_date(
    tickers: List[str],
    as_of_date: pd.Timestamp,
    start_download: str,
    end_download: str,
    cfg: ScreenConfig,
    fdf: Optional[pd.DataFrame],
    profile_map: Optional[Dict[str, Dict[str, str]]] = None,
    enable_audit: bool = False,
    audit_limit: int = 100,
    enable_live_fundamentals: bool = True,
    enable_sec_audit: bool = False,
    sec_audit_limit: int = 50,
) -> pd.DataFrame:
    price_data = download_price_history(tickers, start_download, end_download)
    price_data = {t: add_indicators(df) for t, df in price_data.items()}

    rows = []
    for idx, ticker in enumerate(tickers):
        profile = (profile_map or {}).get(ticker, {"company_name": "N/A", "sector": "N/A", "business_nature": "N/A"})
        df = price_data.get(ticker)
        if df is None or df.empty:
            rows.append({
                "ticker": ticker,
                "company_name": profile.get("company_name", "N/A"),
                "sector": profile.get("sector", "N/A"),
                "business_nature": profile.get("business_nature", "N/A"),
                "pass": False,
                "note": "No price data",
            })
            continue

        tech_pass, tech = eval_technical(df, as_of_date, cfg)
        fund_pass, fund = eval_fundamental(ticker, as_of_date, fdf, cfg)
        if enable_live_fundamentals and (pd.isna(fund.get("eps_yoy_value")) or pd.isna(fund.get("rev_yoy_value")) or pd.isna(fund.get("roe_avg"))):
            live = load_live_fundamentals_yf(ticker)
            fund["eps_yoy_value"] = live.get("eps_yoy_value", fund.get("eps_yoy_value"))
            fund["rev_yoy_value"] = live.get("rev_yoy_value", fund.get("rev_yoy_value"))
            fund["roe_avg"] = live.get("roe_avg", fund.get("roe_avg"))
            fund["revenue_value"] = live.get("revenue_value", fund.get("revenue_value"))
            if profile.get("company_name") == "N/A":
                profile["company_name"] = str(live.get("company_name", "N/A"))
            if profile.get("sector") == "N/A":
                profile["sector"] = str(live.get("sector", "N/A"))
            if profile.get("business_nature") == "N/A":
                profile["business_nature"] = str(live.get("business_nature", "N/A"))
        final_pass = (tech_pass if cfg.apply_technical_filter else True) and fund_pass

        score = 0
        score += 1 if tech.get("above_mas") else 0
        score += 1 if tech.get("ma_stack") else 0
        score += 1 if tech.get("within_high") else 0
        score += 1 if tech.get("above_low") else 0
        score += 1 if fund.get("eps_pass") else 0
        score += 1 if fund.get("rev_pass") else 0
        score += 1 if fund.get("roe_pass") else 0
        score += 1 if fund.get("market_cap_pass") else 0
        audit_close = np.nan
        audit_diff_pct = np.nan
        if enable_audit and idx < int(audit_limit):
            audit_close = stooq_close_on_or_before(ticker, as_of_date.strftime("%Y-%m-%d"))
            close_val = tech.get("close")
            if pd.notna(audit_close) and close_val is not None and pd.notna(close_val) and float(close_val) != 0:
                audit_diff_pct = (float(audit_close) / float(close_val) - 1.0) * 100.0
        sec_eps_yoy = np.nan
        sec_rev_yoy = np.nan
        sec_roe_avg = np.nan
        if enable_sec_audit and idx < int(sec_audit_limit):
            sec = load_sec_fundamentals_audit(ticker, as_of_date.strftime("%Y-%m-%d"))
            sec_eps_yoy = sec.get("eps_yoy_sec", np.nan)
            sec_rev_yoy = sec.get("rev_yoy_sec", np.nan)
            sec_roe_avg = sec.get("roe_avg_sec", np.nan)

        rows.append({
            "ticker": ticker,
            "company_name": profile.get("company_name", "N/A"),
            "sector": profile.get("sector", "N/A"),
            "business_nature": profile.get("business_nature", "N/A"),
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
            "eps_yoy_value": fund.get("eps_yoy_value", np.nan),
            "rev_yoy_value": fund.get("rev_yoy_value", np.nan),
            "revenue_value": fund.get("revenue_value", np.nan),
            "roe_3y_avg_value": fund.get("roe_avg", np.nan),
            "fundamental_rules_hit": fund.get("fundamental_rules_hit", np.nan),
            "eps_ge_min_qtrs": fund.get("eps_ge_min_qtrs", np.nan),
            "rev_ge_min_qtrs": fund.get("rev_ge_min_qtrs", np.nan),
            "eps_consec_qtrs": fund.get("eps_consec_qtrs", np.nan),
            "rev_consec_qtrs": fund.get("rev_consec_qtrs", np.nan),
            "annual_revenue_uptrend": fund.get("annual_revenue_uptrend", np.nan),
            "eps_pass": fund.get("eps_pass", np.nan),
            "rev_pass": fund.get("rev_pass", np.nan),
            "roe_pass": fund.get("roe_pass", np.nan),
            "market_cap_pass": fund.get("market_cap_pass", np.nan),
            "market_cap_value": fund.get("market_cap_value", np.nan),
            "roe_std": fund.get("roe_std", np.nan),
            "roe_stable": fund.get("roe_stable", np.nan),
            "roe_each_ge_min": fund.get("roe_each_ge_min", np.nan),
            "roe_quality": fund.get("roe_quality", "N/A"),
            "audit_close_stooq": audit_close,
            "audit_close_diff_pct": audit_diff_pct,
            "audit_eps_yoy_sec": sec_eps_yoy,
            "audit_rev_yoy_sec": sec_rev_yoy,
            "audit_roe_avg_sec": sec_roe_avg,
            "fundamentals_used": fund.get("fundamentals_used"),
            "fundamentals_source": fund.get("fundamentals_source"),
            "note": fund.get("note") or tech.get("note"),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["pass", "score", "ret_63d"], ascending=[False, False, False]).reset_index(drop=True)


def monitor_watchlist(
    watchlist_df: pd.DataFrame,
    monitor_date: pd.Timestamp,
    enable_wvf_alert: bool = False,
    wvf_cfg: Optional[WVFConfig] = None,
) -> pd.DataFrame:
    tickers = watchlist_df["ticker"].astype(str).str.upper().tolist()
    start = (monitor_date - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    end = (monitor_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    price_data = download_price_history(tickers, start, end)
    price_data = {t: add_indicators(df) for t, df in price_data.items()}
    if enable_wvf_alert:
        cfg = wvf_cfg or WVFConfig()
        price_data = {t: add_wvf_indicator(df, cfg) for t, df in price_data.items()}

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
        recent_3 = temp.tail(3)
        recent_green = pd.Series(dtype=bool)
        wvf_last_green_date = pd.NaT
        wvf_last_green_value = np.nan
        wvf_last_green_upper_band = np.nan
        wvf_last_green_range_high = np.nan
        wvf_last_green_trigger = "N/A"
        if enable_wvf_alert:
            recent_green = recent_3.get("WVF_GREEN", pd.Series(dtype=bool)).fillna(False).astype(bool)
            green_rows = recent_3[recent_green]
            if not green_rows.empty:
                last_green = green_rows.iloc[-1]
                last_green_idx = green_rows.index[-1]
                last_green_wvf = float(last_green.get("WVF", np.nan))
                last_green_upper = float(last_green.get("WVF_UPPER_BAND", np.nan))
                last_green_range_high = float(last_green.get("WVF_RANGE_HIGH", np.nan))
                cond_upper = pd.notna(last_green_wvf) and pd.notna(last_green_upper) and last_green_wvf >= last_green_upper
                cond_range = pd.notna(last_green_wvf) and pd.notna(last_green_range_high) and last_green_wvf >= last_green_range_high
                if cond_upper and cond_range:
                    wvf_last_green_trigger = "upper_band & range_high"
                elif cond_upper:
                    wvf_last_green_trigger = "upper_band"
                elif cond_range:
                    wvf_last_green_trigger = "range_high"
                else:
                    wvf_last_green_trigger = "green_rule"
                wvf_last_green_date = pd.Timestamp(last_green_idx)
                wvf_last_green_value = last_green_wvf
                wvf_last_green_upper_band = last_green_upper
                wvf_last_green_range_high = last_green_range_high
        wvf_green_last_3d = bool(recent_green.any()) if enable_wvf_alert else False
        action_status, action_note = derive_action_status(ticker, df, monitor_date)
        rows.append({
            "ticker": ticker,
            "action": action_status,
            "current_price": float(latest["Close"]),
            "ret_63d": float(latest.get("RET_63D", np.nan)),
            "ma50": float(latest.get("MA50", np.nan)),
            "ma150": float(latest.get("MA150", np.nan)),
            "ma200": float(latest.get("MA200", np.nan)),
            "wvf_value": float(latest.get("WVF", np.nan)) if enable_wvf_alert else np.nan,
            "wvf_upper_band": float(latest.get("WVF_UPPER_BAND", np.nan)) if enable_wvf_alert else np.nan,
            "wvf_range_high": float(latest.get("WVF_RANGE_HIGH", np.nan)) if enable_wvf_alert else np.nan,
            "wvf_green_alert": bool(latest.get("WVF_GREEN", False)) if enable_wvf_alert else False,
            "wvf_green_last_3d": wvf_green_last_3d,
            "wvf_last_green_date": wvf_last_green_date,
            "wvf_last_green_value": wvf_last_green_value,
            "wvf_last_green_upper_band": wvf_last_green_upper_band,
            "wvf_last_green_range_high": wvf_last_green_range_high,
            "wvf_last_green_trigger": wvf_last_green_trigger,
            "monitor_note": action_note,
        })

    return watchlist_df.merge(pd.DataFrame(rows), on="ticker", how="left")


st.set_page_config(
    page_title="US-STOCK · Terminal",
    page_icon="▌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (Terminal / Bloomberg-inspired · Enhanced Visual Edition) ---
st.markdown("""
<style>
    /* ─────────────  FONTS  ───────────── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@200;400;500;700;800&display=swap');

    :root {
        --bg:           #060910;
        --bg-elev:      #141C2A;
        --bg-hover:     #243246;
        --bg-panel:     #0A0D14;
        --bg-glass:     rgba(10,13,20,0.72);
        --border:       #2A3A52;
        --border-strong:#3F5676;
        --text:         #F4F8FF;
        --text-muted:   #C7D3E6;
        --text-dim:     #92A3BF;
        --amber:        #FFB800;
        --amber-dim:    #A37700;
        --amber-glow:   rgba(255,184,0,0.35);
        --cyan:         #22D3EE;
        --cyan-dim:     #0E7C8F;
        --cyan-glow:    rgba(34,211,238,0.3);
        --green:        #10B981;
        --green-glow:   rgba(16,185,129,0.3);
        --red:          #EF4444;
        --purple:       #A78BFA;
        --mono:         'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    /* ─────────────  ANIMATIONS  ───────────── */
    @keyframes scan        { 0%{transform:translateX(-100%)} 100%{transform:translateX(200%)} }
    @keyframes blink       { 50%{opacity:0} }
    @keyframes aurora      { 0%,100%{opacity:0.6;transform:scale(1) translateY(0)} 50%{opacity:1;transform:scale(1.08) translateY(-12px)} }
    @keyframes aurora2     { 0%,100%{opacity:0.4;transform:scale(1) translateX(0)} 50%{opacity:0.7;transform:scale(1.05) translateX(14px)} }
    @keyframes glow-pulse  { 0%,100%{opacity:0.7} 50%{opacity:1} }
    @keyframes shimmer     { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
    @keyframes border-flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes float-up    { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }

    /* ─── Global reset ─── */
    html, body, [class*="css"] {
        font-family: var(--mono);
        font-feature-settings: "tnum","ss01","cv11";
    }

    /* ─── Aurora background ─── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        position: relative;
    }
    .stApp::after, [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse 80% 55% at 20% -5%,  rgba(255,184,0,0.07)  0%, transparent 55%),
            radial-gradient(ellipse 65% 50% at 85% 105%, rgba(34,211,238,0.06) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 60% 50%,  rgba(167,139,250,0.03) 0%, transparent 50%);
        animation: aurora 14s ease-in-out infinite;
    }

    /* ─── Dot-grid + scanline overlay ─── */
    .stApp::before, [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 9999;
        mix-blend-mode: overlay;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 4px);
        background-size: 28px 28px, 100% 4px;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }

    /* ─────────────  TERMINAL HEADER BAR  ───────────── */
    .term-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 22px;
        margin: 0 0 18px 0;
        background: linear-gradient(180deg, rgba(20,26,38,0.95) 0%, rgba(6,9,16,0.98) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,184,0,0.18);
        border-top: 2px solid var(--amber);
        position: relative;
        overflow: hidden;
        box-shadow:
            0 0 0 1px rgba(255,184,0,0.06) inset,
            0 4px 32px rgba(0,0,0,0.6),
            0 0 60px rgba(255,184,0,0.04);
    }
    /* sweeping amber scan line */
    .term-header::after {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 40%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,184,0,0.07), transparent);
        animation: scan 5s linear infinite;
    }
    /* top-edge glow bar */
    .term-header::before {
        content: "";
        position: absolute;
        top: -1px; left: 10%; right: 10%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--amber), transparent);
        filter: blur(2px);
        animation: glow-pulse 3s ease-in-out infinite;
    }

    .term-brand { display: flex; align-items: baseline; gap: 14px; }
    .term-brand .brand {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--amber);
        letter-spacing: 0.1em;
        text-shadow: 0 0 18px var(--amber-glow), 0 0 40px rgba(255,184,0,0.15);
    }
    .term-brand .brand .cursor {
        display: inline-block;
        width: 0.55em; height: 1em;
        background: var(--amber);
        box-shadow: 0 0 10px var(--amber), 0 0 20px var(--amber-glow);
        vertical-align: -2px; margin-left: 4px;
        animation: blink 1.05s step-end infinite;
    }
    .term-brand .slash { color: var(--text-dim); font-weight: 400; }
    .term-brand .tagline {
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .term-header-right {
        display: flex; gap: 10px; align-items: center;
        font-family: var(--mono);
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 11px;
        border: 1px solid var(--border-strong);
        color: var(--text-muted);
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
        transition: all 0.2s;
    }
    .status-chip:hover { border-color: var(--amber); color: var(--text); }
    .status-chip .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 8px var(--green), 0 0 14px var(--green-glow);
        animation: glow-pulse 1.8s ease-in-out infinite;
    }
    .status-chip.live {
        color: var(--green);
        border-color: rgba(16,185,129,0.35);
        background: rgba(16,185,129,0.06);
        box-shadow: 0 0 12px rgba(16,185,129,0.08);
    }
    .status-chip.hist { color: var(--cyan); border-color: rgba(34,211,238,0.35); }
    .status-chip.hist .dot { background: var(--cyan); box-shadow: 0 0 8px var(--cyan), 0 0 14px var(--cyan-glow); }
    .status-chip .kv-key { color: var(--text-dim); }
    .status-chip .kv-val { color: var(--text); font-weight: 500; }

    /* ─────────────  PROCEDURE · WORKFLOW  ───────────── */
    .procedure {
        border: 1px solid var(--border);
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.02) inset;
    }
    .procedure-head {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 12px;
        font-family: var(--mono); font-size: 0.72rem;
    }
    .procedure-head .pt {
        color: var(--amber);
        text-shadow: 0 0 10px var(--amber-glow);
    }
    .procedure-head .title { color: var(--text); font-weight: 700; letter-spacing: 0.14em; }
    .procedure-head .hint  { color: var(--text-dim); letter-spacing: 0.05em; margin-left: auto; }
    .procedure-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px;
    }
    .step {
        padding: 14px 16px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.015);
        backdrop-filter: blur(8px);
        font-family: var(--mono);
        transition: all 0.25s;
        display: flex; flex-direction: column; gap: 4px;
        position: relative; overflow: hidden;
    }
    .step::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
        pointer-events: none;
    }
    .step .step-num {
        font-size: 0.66rem; font-weight: 500;
        color: var(--text-dim);
        letter-spacing: 0.18em; text-transform: uppercase;
    }
    .step .step-title { font-size: 0.98rem; font-weight: 700; color: var(--text); letter-spacing: 0.02em; }
    .step .step-sub   { font-size: 0.72rem; color: var(--text-dim); line-height: 1.5; letter-spacing: 0.02em; }

    .step.done {
        border-color: rgba(16,185,129,0.4);
        background: linear-gradient(180deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%);
        box-shadow: 0 0 16px rgba(16,185,129,0.08), inset 0 0 20px rgba(16,185,129,0.03);
    }
    .step.done .step-num { color: var(--green); text-shadow: 0 0 8px var(--green-glow); }
    .step.done .step-num::before { content: "✓ "; color: var(--green); }

    .step.active {
        border-color: var(--amber);
        background: linear-gradient(180deg, rgba(255,184,0,0.1) 0%, rgba(255,184,0,0.03) 100%);
        box-shadow:
            inset 3px 0 0 var(--amber),
            0 0 28px rgba(255,184,0,0.12),
            0 0 0 1px rgba(255,184,0,0.08) inset;
    }
    .step.active .step-num {
        color: var(--amber);
        text-shadow: 0 0 10px var(--amber-glow);
        animation: blink 1.4s ease-in-out infinite;
    }
    .step.active .step-num::before { content: "▶ "; color: var(--amber); }
    .step.active .step-title {
        color: var(--amber);
        text-shadow: 0 0 14px var(--amber-glow);
    }
    .step.pending { opacity: 0.45; }
    .step.pending .step-title { color: var(--text-muted); }

    /* ─────────────  TICKER RULE STRIP  ───────────── */
    .ticker-strip {
        border: 1px solid var(--border);
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        padding: 10px 18px;
        margin-bottom: 18px;
        display: flex; flex-wrap: wrap; gap: 22px; align-items: center;
        font-family: var(--mono); font-size: 0.78rem; color: var(--text-muted);
        box-shadow: 0 1px 16px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.015) inset;
        position: relative; overflow: hidden;
    }
    .ticker-strip::before {
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        opacity: 0.3;
    }
    .ticker-strip .pt  { color: var(--amber); margin-right: 6px; text-shadow: 0 0 8px var(--amber-glow); }
    .ticker-strip .k   { color: var(--text-dim); letter-spacing: 0.08em; text-transform: uppercase; margin-right: 6px; }
    .ticker-strip .v   { color: var(--text); font-weight: 500; }
    .ticker-strip .warn{ color: var(--amber); text-shadow: 0 0 8px var(--amber-glow); }
    .ticker-strip .div { color: var(--border-strong); margin: 0 2px; }

    /* ─────────────  KPI STRIP  ───────────── */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        border: 1px solid rgba(255,184,0,0.15);
        background: var(--bg-glass);
        backdrop-filter: blur(14px);
        margin: 10px 0 20px 0;
        box-shadow:
            0 0 0 1px rgba(255,184,0,0.05) inset,
            0 4px 24px rgba(0,0,0,0.4),
            0 0 40px rgba(255,184,0,0.04);
        position: relative; overflow: hidden;
    }
    .kpi-strip::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, var(--amber), var(--cyan), transparent);
        background-size: 200% 100%;
        animation: border-flow 4s linear infinite;
        opacity: 0.6;
    }
    .kpi-cell {
        padding: 20px 22px;
        border-right: 1px solid var(--border);
        position: relative;
        transition: background 0.25s;
    }
    .kpi-cell:hover { background: rgba(255,184,0,0.04); }
    .kpi-cell:last-child { border-right: none; }
    .kpi-cell::before {
        content: "";
        position: absolute;
        top: 12px; left: 0;
        width: 3px; height: 20px;
        background: linear-gradient(180deg, var(--amber), var(--cyan));
        opacity: 0;
        transition: opacity 0.2s;
        box-shadow: 0 0 8px var(--amber-glow);
    }
    .kpi-cell:hover::before { opacity: 1; }
    .kpi-label {
        font-family: var(--mono); font-size: 0.68rem; font-weight: 500;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-dim); margin-bottom: 8px;
        display: flex; align-items: center; gap: 8px;
    }
    .kpi-label::before { content: "■"; color: var(--amber); font-size: 0.6rem; text-shadow: 0 0 6px var(--amber-glow); }
    .kpi-value {
        font-family: var(--mono); font-size: 2rem; font-weight: 700;
        color: var(--text); letter-spacing: -0.02em; line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .kpi-value .unit {
        font-size: 0.75rem; color: var(--text-dim); font-weight: 400;
        margin-left: 4px; letter-spacing: 0.05em;
    }
    .kpi-value.amber {
        color: var(--amber);
        text-shadow: 0 0 16px var(--amber-glow), 0 0 32px rgba(255,184,0,0.1);
    }
    .kpi-value.cyan {
        color: var(--cyan);
        text-shadow: 0 0 16px var(--cyan-glow), 0 0 32px rgba(34,211,238,0.1);
    }
    .kpi-sub { margin-top: 5px; font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim); }

    /* ─────────────  RULE PANEL  ───────────── */
    .rule-box {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-left: 2px solid var(--cyan);
        padding: 16px 20px;
        margin-bottom: 16px;
        font-family: var(--mono); font-size: 0.8rem;
        color: var(--text-muted); line-height: 1.75;
        box-shadow: 0 0 20px rgba(34,211,238,0.05), 0 2px 20px rgba(0,0,0,0.3);
        position: relative;
    }
    .rule-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0; width: 2px;
        background: linear-gradient(180deg, var(--cyan), transparent);
        box-shadow: 0 0 8px var(--cyan-glow);
    }
    .rule-box .lbl {
        color: var(--cyan);
        font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        font-size: 0.72rem; display: block; margin-bottom: 6px;
        text-shadow: 0 0 8px var(--cyan-glow);
    }
    .rule-box code, .rule-box b, .rule-box strong {
        color: var(--text);
        background: rgba(255,184,0,0.1);
        padding: 1px 6px; border-radius: 2px;
        border: 1px solid rgba(255,184,0,0.15);
    }
    .rule-box .rule-line { display: flex; gap: 10px; align-items: flex-start; }
    .rule-box .rule-line .ix { color: var(--amber); min-width: 1.5em; text-shadow: 0 0 6px var(--amber-glow); }

    /* ─────────────  SECTION HEADER  ───────────── */
    .section-header {
        font-family: var(--mono); font-size: 0.78rem; font-weight: 700;
        color: var(--text); letter-spacing: 0.18em; text-transform: uppercase;
        margin: 28px 0 14px 0;
        display: flex; align-items: center; gap: 10px;
    }
    .section-header::before {
        content: ">_";
        color: var(--amber); font-weight: 800;
        text-shadow: 0 0 12px var(--amber-glow), 0 0 24px rgba(255,184,0,0.2);
    }
    .section-header::after {
        content: "";
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(255,184,0,0.3), var(--border), transparent);
        margin-left: 6px;
    }
    .section-header .count {
        color: var(--text-dim); font-weight: 400;
        font-size: 0.72rem; letter-spacing: 0.1em;
    }

    /* ─────────────  SIDEBAR  ───────────── */
    [data-testid="stSidebar"] {
        background: rgba(10,15,24,0.96);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 30px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] > div { padding-top: 12px; }
    [data-testid="stSidebar"] h3 {
        font-family: var(--mono); font-size: 0.75rem !important; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
        color: var(--amber); margin-bottom: 16px !important;
        text-shadow: 0 0 10px var(--amber-glow);
    }
    [data-testid="stSidebar"] label {
        font-family: var(--mono); font-size: 0.78rem !important;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--text) !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
    }

    /* ─────────────  WIDGETS  ───────────── */
    .stButton > button {
        font-family: var(--mono) !important; font-size: 0.78rem !important;
        font-weight: 600 !important; letter-spacing: 0.12em !important;
        text-transform: uppercase !important; border-radius: 2px !important;
        border: 1px solid var(--border-strong) !important;
        background: rgba(255,255,255,0.03) !important;
        color: var(--text) !important; transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
        background: rgba(255,184,0,0.07) !important;
        box-shadow: 0 0 16px rgba(255,184,0,0.2), 0 0 0 1px rgba(255,184,0,0.3) inset !important;
        text-shadow: 0 0 10px var(--amber-glow);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FFB800, #FFA000) !important;
        color: #060910 !important; border-color: var(--amber) !important;
        box-shadow: 0 0 20px rgba(255,184,0,0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #FFCC33, #FFB800) !important;
        color: #060910 !important;
        box-shadow: 0 0 32px rgba(255,184,0,0.45), 0 4px 16px rgba(255,184,0,0.2) !important;
    }
    .stDownloadButton > button {
        font-family: var(--mono) !important; font-size: 0.78rem !important;
        letter-spacing: 0.1em !important; text-transform: uppercase !important;
        border-radius: 2px !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput input,
    .stTextArea textarea {
        font-family: var(--mono) !important;
        background: var(--bg-elev) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 2px !important; color: var(--text) !important;
        font-variant-numeric: tabular-nums;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-dim) !important;
        opacity: 1 !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 1px var(--amber), 0 0 12px var(--amber-glow) !important;
    }
    [data-baseweb="select"] > div {
        background: var(--bg-elev) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 2px !important; font-family: var(--mono) !important;
        color: var(--text) !important;
    }
    [data-baseweb="select"] * {
        color: var(--text) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--border) !important; }
    .stSlider [role="slider"] {
        background: var(--amber) !important;
        box-shadow: 0 0 12px var(--amber), 0 0 0 4px rgba(255,184,0,0.2) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(10,13,20,0.6);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border) !important; border-radius: 2px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: var(--mono); letter-spacing: 0.08em;
        font-size: 0.8rem; color: var(--text-muted);
    }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 18px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--mono) !important; font-size: 0.8rem !important;
        font-weight: 600 !important; letter-spacing: 0.16em !important;
        text-transform: uppercase !important; color: var(--text-muted) !important;
        background: transparent !important;
        border: 1px solid transparent !important; border-bottom: none !important;
        border-radius: 2px 2px 0 0 !important; padding: 10px 24px !important; margin-right: 2px;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text) !important; background: rgba(255,255,255,0.02) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--amber) !important; border-color: var(--border) !important;
        border-bottom: 1px solid var(--bg) !important;
        background: rgba(255,184,0,0.04) !important;
        position: relative;
        text-shadow: 0 0 10px var(--amber-glow);
    }
    .stTabs [aria-selected="true"]::before {
        content: ""; position: absolute; top: -1px; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--amber), #FFCC33, var(--amber));
        background-size: 200% 100%;
        animation: border-flow 2s linear infinite;
        box-shadow: 0 0 8px var(--amber-glow);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ─── DataFrames ─── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border); border-radius: 2px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    }
    [data-testid="stDataFrame"] table { font-family: var(--mono) !important; font-variant-numeric: tabular-nums; }

    /* ─── Metrics ─── */
    [data-testid="stMetric"] {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border); padding: 12px 14px; border-radius: 2px;
        position: relative; transition: all 0.2s;
    }
    [data-testid="stMetric"]:hover { box-shadow: 0 0 16px rgba(255,184,0,0.08); }
    [data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; width: 2px; height: 100%;
        background: linear-gradient(180deg, var(--amber), var(--cyan));
        box-shadow: 0 0 8px var(--amber-glow); opacity: 0.6;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--mono) !important; font-size: 0.66rem !important;
        font-weight: 500 !important; letter-spacing: 0.14em !important;
        text-transform: uppercase !important; color: var(--text-dim) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--mono) !important; font-weight: 700 !important;
        color: var(--text) !important; font-variant-numeric: tabular-nums;
    }
    [data-testid="stAlert"] {
        font-family: var(--mono); border-radius: 2px !important;
        border: 1px solid var(--border) !important; font-size: 0.82rem;
        background: rgba(10,13,20,0.7) !important; backdrop-filter: blur(8px);
    }

    /* ─────────────  SIGNAL CARD  ───────────── */
    .sig-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, var(--bg-glass) 40%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(16,185,129,0.25);
        border-left: 3px solid var(--green);
        padding: 16px 20px;
        margin-bottom: 12px;
        font-family: var(--mono);
        position: relative; overflow: hidden;
        box-shadow:
            0 0 24px rgba(16,185,129,0.08),
            0 4px 20px rgba(0,0,0,0.4),
            0 0 0 1px rgba(16,185,129,0.05) inset;
        transition: all 0.25s;
    }
    .sig-card:hover {
        border-color: rgba(16,185,129,0.5);
        box-shadow: 0 0 40px rgba(16,185,129,0.14), 0 4px 24px rgba(0,0,0,0.5);
        transform: translateY(-1px);
    }
    .sig-card::after {
        content: "● SIGNAL";
        position: absolute; top: 12px; right: 16px;
        font-size: 0.65rem; letter-spacing: 0.2em;
        color: var(--green);
        text-shadow: 0 0 10px var(--green-glow);
        animation: glow-pulse 1.8s ease-in-out infinite;
    }
    /* corner shimmer */
    .sig-card::before {
        content: "";
        position: absolute; top: 0; right: 0;
        width: 80px; height: 80px;
        background: radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 70%);
        pointer-events: none;
    }
    .sig-head {
        display: flex; justify-content: space-between; align-items: baseline;
        margin-bottom: 10px; padding-right: 90px;
    }
    .sig-code {
        font-size: 1.1rem; font-weight: 700; color: var(--text);
        letter-spacing: 0.05em;
        text-shadow: 0 0 16px rgba(255,255,255,0.15);
    }
    .sig-code .code-num { color: var(--amber); margin-right: 10px; text-shadow: 0 0 8px var(--amber-glow); }
    .sig-sector { font-size: 0.72rem; color: var(--text-dim); letter-spacing: 0.08em; text-transform: uppercase; }
    .sig-row { font-size: 0.8rem; color: var(--text-muted); margin-top: 5px; }
    .sig-row .k   { color: var(--text-dim); }
    .sig-row .v   { color: var(--text); font-weight: 500; }
    .sig-row .up  { color: var(--green); text-shadow: 0 0 6px var(--green-glow); }
    .sig-row .dn  { color: var(--red); }
    .sig-row .bar-sep { color: var(--border-strong); margin: 0 8px; }
    .sig-verdict {
        margin-top: 10px; font-size: 0.78rem; color: var(--text-muted);
        padding: 8px 12px;
        background: rgba(255,255,255,0.02);
        border-left: 2px solid var(--amber);
        box-shadow: -2px 0 8px rgba(255,184,0,0.1);
    }
    .sig-flow {
        margin-top: 8px; padding: 8px 12px;
        background: rgba(255,255,255,0.02);
        border: 1px dashed var(--border);
        font-size: 0.76rem; color: var(--text-muted);
    }

    /* ─── Inline pills ─── */
    .inline-pill {
        display: inline-block; padding: 2px 10px;
        font-family: var(--mono); font-size: 0.72rem;
        border: 1px solid var(--border-strong); color: var(--text-muted);
        letter-spacing: 0.08em; background: rgba(255,255,255,0.03);
        margin-right: 6px; transition: all 0.15s;
    }
    .inline-pill.amber {
        border-color: rgba(255,184,0,0.4); color: var(--amber);
        background: rgba(255,184,0,0.06);
        box-shadow: 0 0 8px rgba(255,184,0,0.1);
    }

    /* ─── WL banner ─── */
    .wl-banner {
        display: flex; align-items: center; gap: 14px;
        padding: 13px 18px; margin: 14px 0;
        background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.02) 50%, transparent 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-left: 3px solid var(--green);
        font-family: var(--mono); font-size: 0.82rem; color: var(--text);
        box-shadow: 0 0 20px rgba(16,185,129,0.08);
        backdrop-filter: blur(8px);
    }
    .wl-banner .tag {
        color: var(--green); font-weight: 700; letter-spacing: 0.15em;
        font-size: 0.72rem; text-shadow: 0 0 8px var(--green-glow);
    }
    .wl-banner .path { color: var(--amber); font-weight: 500; }

    /* ─── Footer ─── */
    .term-footer {
        margin-top: 36px; padding: 16px 0;
        border-top: 1px solid var(--border);
        font-family: var(--mono); font-size: 0.7rem;
        color: var(--text-dim); text-align: center; letter-spacing: 0.1em;
        background: linear-gradient(0deg, rgba(255,184,0,0.02) 0%, transparent 100%);
    }
    .term-footer .sep { color: var(--border-strong); margin: 0 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI helper functions
# ---------------------------------------------------------------------------

def _render_kpi_cell(label, value, unit="", accent="", sub=""):
    klass = f"kpi-value {accent}".strip()
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-cell"><div class="kpi-label">{label}</div>'
            f'<div class="{klass}">{value}{unit_html}</div>{sub_html}</div>')


def _render_kpi_strip(cells):
    return f'<div class="kpi-strip">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Procedure step helper (defined before first use)
# ---------------------------------------------------------------------------

def _us_step(n: int, title: str, sub: str, active: int) -> str:
    state = "done" if n < active else ("active" if n == active else "pending")
    return (
        f'<div class="step {state}">'
        f'<div class="step-num">Step {n:02d}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-sub">{sub}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# SIDEBAR — must run first so all cfg variables are available for header/strips
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sb-status">'
        '<span class="dot" style="background:var(--green);"></span>'
        '&nbsp;US-STOCK · Control Panel'
        '</div>',
        unsafe_allow_html=True,
    )

    _last_run_sb = st.session_state.get("screen_result_date")
    _last_mode_sb = st.session_state.get("screen_result_mode", "")
    if _last_run_sb:
        _mode_label_sb = "offline" if _last_mode_sb == "offline_sample" else "live"
        st.markdown(
            f'<div style="font-size:0.72rem;color:var(--muted);margin-bottom:8px;">'
            f'Last screen: <span style="color:var(--cyan);">{_last_run_sb}</span>'
            f'&nbsp;<span class="inline-pill">{_mode_label_sb}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border-color:var(--border);margin:6px 0 10px;">', unsafe_allow_html=True)
    st.markdown("**▸ Universe & Dates**")

    use_offline_sample_data = st.checkbox(
        "Use offline sample data (no live API)",
        value=False,
        help="Loads bundled sample scan results from data/full_screen_latest.csv and skips live market/fundamental calls.",
    )

    universe_source = st.selectbox(
        "Universe source",
        [
            "S&P 500 (503 stocks)",
            "Top 500 by Volume (NASDAQ + S&P 500)",
            "Top 1000 by Volume (NASDAQ + S&P 500)",
        ],
        index=1,
        help=(
            "S&P 500: exact index members only (~503 tickers, fastest).\n"
            "Top 500 / Top 1000 by Volume: NASDAQ + S&P 500 combined pool ranked by 60-day avg volume. "
            "Top 1000 gives broader coverage closer to TradingView's US market universe."
        ),
    )

    tickers_text = st.text_area(
        "Custom tickers (optional, comma-separated)",
        value="",
        help="Leave blank to use the selected universe source above.",
    )
    screen_date = st.date_input("Screen date", value=date.today())
    monitor_date = st.date_input("Monitor date", value=date.today())
    always_include_text = st.text_input(
        "Always include tickers",
        value="CLS,JBL",
        help="These tickers are forced into the auto universe even if they are not in the top-volume ranking source.",
    )

    st.markdown('<hr style="border-color:var(--border);margin:10px 0;">', unsafe_allow_html=True)
    st.markdown("**▸ Data Sources**")

    fundamentals_file = st.file_uploader("Optional fundamentals snapshot CSV", type=["csv"])
    enable_price_audit = st.checkbox("Enable free price audit (Stooq)", value=False)
    audit_limit = st.number_input("Audit max tickers", min_value=10, max_value=1000, value=100, step=10)
    enable_live_fundamentals = st.checkbox("Auto-fill fundamentals values (Yahoo free)", value=False)
    enable_sec_audit = st.checkbox("Enable 2nd fundamentals audit (SEC free)", value=False)
    sec_audit_limit = st.number_input("SEC audit max tickers", min_value=10, max_value=300, value=50, step=10)

    st.markdown('<hr style="border-color:var(--border);margin:10px 0;">', unsafe_allow_html=True)
    st.markdown("**▸ Rules**")

    with st.expander("▸ Fundamental", expanded=False):
        fundamental_mode_label = st.selectbox(
            "Fundamental setup",
            ["TradingView match (TTM)", "Gemini match (YoY)", "TradingView setup (QoQ)"],
            index=0,
        )
        _is_tv_ttm = (fundamental_mode_label == "TradingView match (TTM)")
        if _is_tv_ttm:
            st.caption("Matches TradingView screener: EPS TTM YoY > threshold, Revenue TTM YoY > threshold, ROE TTM > threshold. All 3 must pass.")
        min_eps_yoy = st.number_input("Min EPS TTM YoY (%)" if _is_tv_ttm else "Min EPS YoY (%)", value=20.0, step=1.0)
        min_revenue_yoy = st.number_input("Min Revenue TTM YoY (%)" if _is_tv_ttm else "Min Revenue YoY (%)", value=20.0, step=1.0)
        min_roe_avg = st.number_input("Min ROE TTM (%)" if _is_tv_ttm else "Min 3Y average ROE (%)", value=15.0, step=1.0)
        roe_stability_max_std = st.number_input("ROE stability max std-dev", value=40.0, step=1.0)
        min_market_cap_b = st.number_input("Min Market Cap (B USD)", value=10.0, step=1.0)
        min_fundamental_rules_pass = st.number_input(
            "Min passed fundamental rules (out of 3)",
            min_value=1, max_value=3,
            value=3 if _is_tv_ttm else 2,
            step=1,
        )
        require_fundamentals = st.checkbox("Require fundamentals for pass", value=True)

    with st.expander("▸ Technical-Lazy", expanded=False):
        st.caption("MA checks use: MA50, MA150, MA200")
        apply_technical_filter = st.checkbox("Apply Technical-Lazy filter in scan", value=False)
        require_close_above_mas = st.checkbox("Require Close > MA50, MA150, MA200", value=True)
        require_ma_stack = st.checkbox("Require MA50 > MA150 > MA200", value=True)
        high_52w_within_pct = st.number_input("Within 52W high (%)", value=25.0, step=1.0)
        low_52w_above_pct = st.number_input("Above 52W low (%)", value=25.0, step=1.0)

    with st.expander("▸ Technical-VIX", expanded=False):
        st.caption("Separate option: Williams Vix Fix alert on watchlist monitor")
        enable_wvf_alert = st.checkbox("Enable Williams Vix Fix alert", value=False)
        wvf_pd = st.number_input("WVF LookBack Period Standard Deviation High", min_value=5, max_value=120, value=22, step=1)
        wvf_bbl = st.number_input("WVF Bollinger Band Length", min_value=5, max_value=120, value=20, step=1)
        wvf_mult = st.number_input("WVF Bollinger Band Std Dev Mult", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        wvf_lb = st.number_input("WVF Look Back Period Percentile High", min_value=10, max_value=250, value=50, step=1)
        wvf_ph = st.number_input("WVF Highest Percentile", min_value=0.50, max_value=1.20, value=0.85, step=0.01)
        wvf_pl = st.number_input("WVF Lowest Percentile", min_value=0.90, max_value=1.50, value=1.01, step=0.01)

    st.markdown('<hr style="border-color:var(--border);margin:10px 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.70rem;color:var(--muted);line-height:1.6;">'
        '<span style="color:var(--amber);">Tip:</span> Run Screen first, then Save passed list, '
        'then switch to Tab 2 to monitor daily signals.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Build config objects — sidebar must have run before this block
# ---------------------------------------------------------------------------

wvf_cfg = WVFConfig(
    lookback_std_high=int(wvf_pd),
    bollinger_length=int(wvf_bbl),
    bollinger_mult=float(wvf_mult),
    lookback_percentile=int(wvf_lb),
    high_percentile=float(wvf_ph),
    low_percentile=float(wvf_pl),
)

as_of_ts = pd.Timestamp(screen_date)
monitor_ts = pd.Timestamp(monitor_date)
selected_screen_date_str = as_of_ts.strftime("%Y-%m-%d")
prev_selected_screen_date = st.session_state.get("selected_screen_date_str")
if prev_selected_screen_date is not None and prev_selected_screen_date != selected_screen_date_str:
    # Prevent stale table/save behavior when user changes date but has not rerun scan yet.
    st.session_state.pop("screen_df", None)
    st.session_state.pop("screen_result_date", None)
    st.session_state.pop("screen_result_mode", None)
st.session_state["selected_screen_date_str"] = selected_screen_date_str

manual_tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
always_include = [x.strip().upper() for x in always_include_text.split(",") if x.strip()]
universe_note = ""
universe_preview = pd.DataFrame()
if use_offline_sample_data:
    tickers = []
    try:
        sample_df = load_sample_screen_data()
        preview_cols = [c for c in ["ticker", "company_name", "sector", "business_nature", "pass", "score"] if c in sample_df.columns]
        universe_preview = sample_df[preview_cols].head(20).copy()
        universe_note = f"Offline sample mode enabled ({len(sample_df)} sample rows)."
    except Exception as exc:
        universe_note = f"Offline sample load failed: {exc}"
elif manual_tickers:
    tickers = manual_tickers
    universe_note = f"Using custom universe ({len(tickers)} tickers)."
else:
    tickers = []
    universe_note = (
        f"Auto universe will be loaded when you click Run screen "
        "(uses the universe source selected above)."
    )

profile_map = build_profile_map(tickers) if tickers else {}

cfg = ScreenConfig(
    min_eps_yoy=float(min_eps_yoy),
    min_revenue_yoy=float(min_revenue_yoy),
    min_roe_avg=float(min_roe_avg),
    high_52w_within_pct=float(high_52w_within_pct),
    low_52w_above_pct=float(low_52w_above_pct),
    require_fundamentals=bool(require_fundamentals),
    require_close_above_mas=bool(require_close_above_mas),
    require_ma_stack=bool(require_ma_stack),
    use_tradingview_setup=(fundamental_mode_label == "TradingView setup (QoQ)"),
    min_market_cap=float(min_market_cap_b) * 1_000_000_000.0,
    roe_stability_max_std=float(roe_stability_max_std),
    fundamental_mode=(
        "tv_ttm" if fundamental_mode_label == "TradingView match (TTM)"
        else ("gemini_yoy" if fundamental_mode_label == "Gemini match (YoY)" else "tradingview_qoq")
    ),
    apply_technical_filter=bool(apply_technical_filter),
    min_fundamental_rules_pass=int(min_fundamental_rules_pass),
)

try:
    fdf = load_fundamentals_snapshot(fundamentals_file)
except Exception as exc:
    st.error(f"Failed to read fundamentals CSV: {exc}")
    fdf = None


# ---------------------------------------------------------------------------
# Dynamic Terminal Header — uses cfg / universe_note resolved above
# ---------------------------------------------------------------------------

_mode_chip_text = "Offline Sample" if use_offline_sample_data else ("Custom" if manual_tickers else "Auto")
if use_offline_sample_data:
    _univ_chip_label = "Offline Sample"
elif manual_tickers:
    _univ_chip_label = f"Custom ({len(manual_tickers)})"
elif universe_source.startswith("S&P 500"):
    _univ_chip_label = "S&P 500"
elif "1000" in universe_source:
    _univ_chip_label = "Top 1000 Vol"
else:
    _univ_chip_label = "Top 500 Vol"
_univ_size = len(tickers) if tickers else AUTO_UNIVERSE_TOP_N
_univ_chip_text = _univ_chip_label
_fund_mode_chip = (
    "TV-TTM" if fundamental_mode_label == "TradingView match (TTM)"
    else ("YoY" if fundamental_mode_label == "Gemini match (YoY)" else "QoQ")
)
_tech_chip = "Tech ON" if apply_technical_filter else "Tech OFF"

st.markdown(f"""
<div class="term-header">
    <div class="term-brand">
        <span class="brand">US-STOCK<span class="cursor"></span></span>
        <span class="slash">//</span>
        <span class="tagline">US Equity Lazy Monitor · Terminal</span>
    </div>
    <div class="term-header-right">
        <span class="status-chip live">
            <span class="dot"></span>Live
        </span>
        <span class="status-chip">
            <span class="kv-key">Date</span>
            <span class="kv-val">{selected_screen_date_str}</span>
        </span>
        <span class="status-chip">
            <span class="kv-key">Universe</span>
            <span class="kv-val">{_univ_chip_text}</span>
        </span>
        <span class="status-chip">
            <span class="kv-key">Mode</span>
            <span class="kv-val">{_mode_chip_text} · {_fund_mode_chip} · {_tech_chip}</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Procedure strip — dynamic active step
# ---------------------------------------------------------------------------

_wl_saved = bool(list(WATCHLIST_DIR.glob("US stock*.csv")))
_screen_done = "screen_df" in st.session_state
if _screen_done and _wl_saved:
    _active_step = 3
elif _screen_done:
    _active_step = 2
else:
    _active_step = 1

_steps_html = (
    _us_step(1, "Screen",          "Sidebar · set date & rules, then Run Screen", _active_step)
    + _us_step(2, "Save Watchlist", "Tab 1 · select passed stocks · save CSV",    _active_step)
    + _us_step(3, "Monitor",        "Tab 2 · load watchlist · run daily monitor", _active_step)
)
st.markdown(f"""
<div class="procedure">
    <div class="procedure-head">
        <span class="pt">▸</span>
        <span class="title">WORKFLOW · PROCEDURE</span>
        <span class="hint">Follow steps in order · Current: Step {_active_step:02d} / 03</span>
    </div>
    <div class="procedure-steps">{_steps_html}</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dynamic Rule strip — uses live cfg values
# ---------------------------------------------------------------------------

_tech_rule_str = (
    f"MA50 &gt; MA150 &gt; MA200 · Within {cfg.high_52w_within_pct:.0f}% of 52W High"
    if apply_technical_filter
    else "Technical-Lazy filter DISABLED"
)
_wvf_warn = '<span class="warn">WVF ON</span>' if enable_wvf_alert else ""
_eps_label = "EPS YoY" if _fund_mode_chip == "YoY" else "EPS QoQ"
_rev_label = "Rev YoY" if _fund_mode_chip == "YoY" else "Rev QoQ"

st.markdown(f"""
<div class="ticker-strip">
    <span>
        <span class="pt">▸</span><span class="k">Fundamental</span>
        <span class="v">{_eps_label} &ge; {cfg.min_eps_yoy:.0f}%</span>
        <span class="div">│</span>
        <span class="v">{_rev_label} &ge; {cfg.min_revenue_yoy:.0f}%</span>
        <span class="div">│</span>
        <span class="v">3Y ROE &ge; {cfg.min_roe_avg:.0f}%</span>
        <span class="div">│</span>
        <span class="v">Mkt Cap &ge; {min_market_cap_b:.0f}B</span>
        <span class="div">│</span>
        <span class="v">Pass &ge; {cfg.min_fundamental_rules_pass}/3 rules</span>
    </span>
    <span><span class="k">Technical</span><span class="v">{_tech_rule_str}</span>{_wvf_warn}</span>
    <span><span class="k">Universe</span><span class="v">{_univ_chip_text}</span></span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["1. Screen & Save", "2. Monitor Watchlist", "3. Saved Files"])

# ── Tab 1: Screen & Save ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Screen Stocks on Selected Date</div>', unsafe_allow_html=True)

    # Rule-box instructions
    if fundamental_mode_label == "TradingView match (TTM)":
        _setup_note = (
            "TradingView match (TTM): replicates TV screener exactly — "
            f"EPS Diluted TTM YoY, Revenue TTM YoY, and ROE TTM. At least {cfg.min_fundamental_rules_pass} must pass."
        )
        _roe_note = "ROE TTM = (trailing 12-month net income) / (avg quarterly equity) × 100."
    elif fundamental_mode_label == "Gemini match (YoY)":
        _setup_note = "Gemini match (YoY): year-over-year EPS &amp; Revenue continuity + 3-year average ROE + market cap."
        _roe_note = "ROE: Net Income / Shareholders' Equity (3-year annual average)."
    else:
        _setup_note = "TradingView setup (QoQ): quarter-over-quarter EPS &amp; Revenue + quarterly TTM-style ROE + market cap."
        _roe_note = "ROE TTM: rolling 4-quarter net income / avg equity."

    st.markdown(f"""
<div class="rule-box">
    <div class="rule-line"><span class="lbl">Step 1</span> Choose screen date in sidebar, configure rules, click <strong>Run screen</strong> → fundamental list.</div>
    <div class="rule-line"><span class="lbl">Step 2</span> Enable <em>Technical-Lazy filter</em> in sidebar to further narrow with MA stack &amp; 52-week position.</div>
    <div class="rule-line"><span class="lbl">Setup</span> {_setup_note}</div>
    <div class="rule-line"><span class="lbl">Pass logic</span> Require at least <span class="ix">{cfg.min_fundamental_rules_pass}</span> of 3 fundamental rules (EPS, Revenue, ROE) <em>plus</em> market cap &gt; {cfg.min_market_cap/1e9:.0f}B.</div>
    <div class="rule-line"><span class="lbl">ROE</span> {_roe_note}</div>
    <div class="rule-line"><span class="lbl">Technical</span> {_tech_rule_str} {'— <strong>active</strong>: applied after fundamental pass' if apply_technical_filter else '— inactive (fundamental-only screen)'}.</div>
</div>
""", unsafe_allow_html=True)

    last_run_date = st.session_state.get("screen_result_date")
    if last_run_date and last_run_date != selected_screen_date_str:
        st.warning("Screen date changed. Please click Run screen to refresh results for the newly selected date.")

    run_universe_note = st.session_state.get("screen_universe_note")
    run_universe_preview = st.session_state.get("screen_universe_preview")
    if use_offline_sample_data:
        st.caption(universe_note)
        if not universe_preview.empty:
            st.dataframe(universe_preview, use_container_width=True)
    elif tickers:
        st.caption(universe_note)
        if manual_tickers:
            st.caption(", ".join(tickers[:30]) + (" ..." if len(tickers) > 30 else ""))
        else:
            st.dataframe(universe_preview.head(20), use_container_width=True)
    else:
        st.info(universe_note)
    if run_universe_note and last_run_date == selected_screen_date_str:
        st.caption(run_universe_note)
    if isinstance(run_universe_preview, pd.DataFrame) and not run_universe_preview.empty and last_run_date == selected_screen_date_str:
        st.dataframe(run_universe_preview.head(20), use_container_width=True)

    left, right = st.columns(2)

    with left:
        if st.button("Run screen", use_container_width=True):
            if use_offline_sample_data:
                try:
                    st.session_state["screen_df"] = load_sample_screen_data().copy()
                    st.session_state["screen_result_date"] = selected_screen_date_str
                    st.session_state["screen_result_mode"] = "offline_sample"
                    st.success("Loaded bundled sample screen results.")
                except Exception as exc:
                    st.error(f"Failed to load sample screen data: {exc}")
            else:
                with st.spinner("Screening..."):
                    tickers_to_run = manual_tickers.copy()
                    universe_preview_to_show = pd.DataFrame()
                    run_universe_note = universe_note
                    if not tickers_to_run:
                        try:
                            if universe_source.startswith("S&P 500"):
                                # Use exact S&P 500 index members — fastest, no volume download needed
                                sp_profiles = load_sp500_profiles()
                                base_tickers = sp_profiles["ticker"].tolist()
                                universe_preview_to_show = sp_profiles[["ticker", "company_name", "sector"]].head(20)
                                top_n_used = len(base_tickers)
                                run_universe_note = f"Using S&P 500 universe: {len(base_tickers)} index members."
                            else:
                                top_n_used = 1000 if "1000" in universe_source else AUTO_UNIVERSE_TOP_N
                                base_tickers, universe_preview_to_show = top_volume_universe(
                                    as_of_ts.strftime("%Y-%m-%d"),
                                    top_n=top_n_used,
                                    lookback_days=60,
                                )
                                run_universe_note = (
                                    f"Using top {top_n_used} stocks by 60-day avg volume from NASDAQ + S&P 500 combined."
                                )
                            tickers_to_run = apply_always_include(base_tickers, always_include, top_n=len(base_tickers))
                            actually_forced = [t for t in always_include if t in tickers_to_run]
                            if actually_forced:
                                run_universe_note += f" Forced include: {', '.join(actually_forced)}."
                        except Exception as exc:
                            st.error(f"Auto universe load failed: {exc}")
                            tickers_to_run = []
                    if not tickers_to_run:
                        st.error("No valid universe available. Enter custom tickers or retry auto universe load.")
                        st.stop()
                    start_download = (as_of_ts - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
                    end_download = (as_of_ts + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                    st.session_state["screen_df"] = screen_on_date(
                        tickers=tickers_to_run,
                        as_of_date=as_of_ts,
                        start_download=start_download,
                        end_download=end_download,
                        cfg=cfg,
                        fdf=fdf,
                        profile_map=build_profile_map(tickers_to_run) if tickers_to_run else {},
                        enable_audit=bool(enable_price_audit),
                        audit_limit=int(audit_limit),
                        enable_live_fundamentals=bool(enable_live_fundamentals),
                        enable_sec_audit=bool(enable_sec_audit),
                        sec_audit_limit=int(sec_audit_limit),
                    )
                    st.session_state["screen_universe_note"] = run_universe_note
                    st.session_state["screen_universe_preview"] = universe_preview_to_show
                    st.session_state["screen_result_date"] = selected_screen_date_str
                    st.session_state["screen_result_mode"] = "live_scan"
                    st.success("Screen completed.")

    with right:
        if st.button("Save passed list", use_container_width=True):
            screen_df = st.session_state.get("screen_df")
            if screen_df is None or screen_df.empty:
                st.warning("Run the screen first.")
            else:
                passed_save = screen_df[screen_df["pass"]].copy()
                if passed_save.empty:
                    st.warning("No passed stocks to save.")
                else:
                    result_date_str = st.session_state.get("screen_result_date", selected_screen_date_str)
                    _save_path = save_watchlist(passed_save, pd.Timestamp(result_date_str))
                    st.session_state["last_saved_path"] = str(_save_path)
                    st.session_state["last_saved_count"] = len(passed_save)

    # wl-banner save confirmation
    _last_saved_path = st.session_state.get("last_saved_path")
    _last_saved_count = st.session_state.get("last_saved_count")
    if _last_saved_path:
        import pathlib as _pl
        _saved_name = _pl.Path(_last_saved_path).name
        st.markdown(f"""
<div class="wl-banner">
    <span class="tag">SAVED</span>
    <span class="path">{_saved_name}</span>
    &nbsp;·&nbsp;
    <span style="color:var(--cyan);">{_last_saved_count} stocks</span>
</div>
""", unsafe_allow_html=True)
        try:
            with open(_last_saved_path, "rb") as _f:
                st.download_button(
                    label=f"Download {_saved_name}",
                    data=_f.read(),
                    file_name=_saved_name,
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception:
            pass

    screen_df = st.session_state.get("screen_df")
    if screen_df is not None and not screen_df.empty:
        # KPI strip after screen
        _total_screened = len(screen_df)
        _passed_count = int(screen_df["pass"].sum()) if "pass" in screen_df.columns else 0
        _pass_rate = f"{(_passed_count / _total_screened * 100):.1f}" if _total_screened else "0.0"
        _screen_res_date = st.session_state.get("screen_result_date", selected_screen_date_str)
        _kpi_cells = [
            _render_kpi_cell("Total Screened", _total_screened, accent="cyan"),
            _render_kpi_cell("Passed", _passed_count, accent="cyan" if _passed_count > 0 else ""),
            _render_kpi_cell("Pass Rate", _pass_rate, unit="%", accent="amber" if float(_pass_rate) > 0 else ""),
            _render_kpi_cell("Screen Date", _screen_res_date),
        ]
        st.markdown(_render_kpi_strip(_kpi_cells), unsafe_allow_html=True)

        display_df = screen_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
        display_df = display_df.replace({np.nan: "N/A"})
        st.dataframe(display_df, use_container_width=True)

        passed = display_df[display_df["pass"] == True]
        st.markdown(f'<div class="section-header">Passed Stocks <span class="count">[ {len(passed)} ]</span></div>', unsafe_allow_html=True)
        if not passed.empty:
            if "close" in passed.columns and "screen_close" not in passed.columns:
                passed = passed.copy()
                passed["screen_close"] = passed["close"]
            preferred_passed_cols = [
                "ticker",
                "company_name",
                "sector",
                "business_nature",
                "screen_close",
                "score",
                "technical_pass",
                "fundamental_pass",
                "fundamental_rules_hit",
                "eps_ge_min_qtrs",
                "eps_consec_qtrs",
                "rev_ge_min_qtrs",
                "rev_consec_qtrs",
                "annual_revenue_uptrend",
                "roe_3y_avg_value",
                "roe_stable",
                "roe_quality",
                "eps_yoy_value",
                "rev_yoy_value",
                "revenue_value",
                "market_cap_value",
                "market_cap_pass",
                "ret_63d",
                "audit_close_stooq",
                "audit_close_diff_pct",
            ]
            available_passed_cols = [c for c in preferred_passed_cols if c in passed.columns]
            st.dataframe(
                passed[available_passed_cols] if available_passed_cols else passed,
                use_container_width=True,
            )
    else:
        st.info("No screen result yet.")


# ── Tab 2: Monitor Watchlist ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Monitor a Saved Watchlist</div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="rule-box">
    <div class="rule-line"><span class="lbl">Step</span> Select a saved watchlist (or upload a CSV), set the monitor date in sidebar, then click <strong>Run monitor</strong>.</div>
    <div class="rule-line"><span class="lbl">WVF</span> Williams Vix Fix is the 2nd indicator — it flags potential price-bottom reversals on watchlist stocks.</div>
    <div class="rule-line"><span class="lbl">WVF Status</span> {'<span class="ix">ENABLED</span> — green alerts will appear below when triggered.' if enable_wvf_alert else 'DISABLED — enable in sidebar under Technical-VIX.'}</div>
    <div class="rule-line"><span class="lbl">Actions</span> BUY / SELL / HOLD are active rules computed from <code>buy_zone_rule()</code> / <code>sell_zone_rule()</code> / <code>hold_zone_rule()</code>.</div>
</div>
""", unsafe_allow_html=True)

    files = list_saved_watchlists()
    uploaded_watchlist = st.file_uploader("Or upload watchlist CSV", type=["csv"], key="monitor_upload")

    selected_path: Optional[Path] = None
    if files:
        options = {p.name: p for p in files}
        selected_name = st.selectbox("Select saved watchlist", list(options.keys()))
        selected_path = options[selected_name]
    else:
        st.info("No saved watchlist files found. Upload a CSV below, or save one from Tab 1 first.")

    source_label = "none"
    watchlist_df_to_monitor: Optional[pd.DataFrame] = None
    if uploaded_watchlist is not None:
        try:
            watchlist_df_to_monitor = pd.read_csv(uploaded_watchlist)
            source_label = f"upload:{uploaded_watchlist.name}"
        except Exception as exc:
            st.error(f"Failed to read uploaded watchlist CSV: {exc}")
    elif selected_path is not None:
        watchlist_df_to_monitor = load_watchlist(selected_path)
        source_label = f"saved:{selected_path.name}"

    if st.button("Run monitor", use_container_width=True):
        if watchlist_df_to_monitor is None or watchlist_df_to_monitor.empty:
            st.warning("Please select or upload a non-empty watchlist CSV first.")
        else:
            st.session_state["monitor_df"] = monitor_watchlist(
                watchlist_df_to_monitor,
                monitor_ts,
                enable_wvf_alert=bool(enable_wvf_alert),
                wvf_cfg=wvf_cfg,
            )
            st.session_state["monitor_source_label"] = source_label
            st.success("Monitor run completed.")

    monitor_df = st.session_state.get("monitor_df")
    monitor_source_label = st.session_state.get("monitor_source_label")
    if monitor_df is not None and monitor_source_label == source_label:
        st.dataframe(monitor_df, use_container_width=True)

        buy_count = int((monitor_df["action"] == "BUY").sum())
        sell_count = int((monitor_df["action"] == "SELL").sum())
        hold_count = int((monitor_df["action"] == "HOLD").sum())
        watch_count = int((monitor_df["action"] == "WATCH").sum())

        # KPI strip for monitor results
        _mon_kpi_cells = [
            _render_kpi_cell("BUY", buy_count, accent="cyan" if buy_count > 0 else ""),
            _render_kpi_cell("SELL", sell_count, accent="amber" if sell_count > 0 else ""),
            _render_kpi_cell("HOLD", hold_count),
            _render_kpi_cell("WATCH", watch_count),
        ]
        st.markdown(_render_kpi_strip(_mon_kpi_cells), unsafe_allow_html=True)

        if bool(enable_wvf_alert):
            wvf_green_today_count = int((monitor_df["wvf_green_alert"] == True).sum())
            wvf_green_last3_count = int((monitor_df.get("wvf_green_last_3d", False) == True).sum())

            _wvf_kpi_cells = [
                _render_kpi_cell("WVF Green (Today)", wvf_green_today_count, accent="cyan" if wvf_green_today_count > 0 else ""),
                _render_kpi_cell("WVF Green (Last 3d)", wvf_green_last3_count, accent="cyan" if wvf_green_last3_count > 0 else ""),
            ]
            st.markdown(_render_kpi_strip(_wvf_kpi_cells), unsafe_allow_html=True)

            if wvf_green_last3_count > 0:
                green_rows = monitor_df[monitor_df["wvf_green_last_3d"] == True].copy()
                if "wvf_last_green_date" in green_rows.columns:
                    green_rows["wvf_last_green_date"] = pd.to_datetime(
                        green_rows["wvf_last_green_date"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")

                st.markdown(f'<div class="section-header">WVF Signals <span class="count">[ {wvf_green_last3_count} ]</span></div>', unsafe_allow_html=True)

                # sig-card for each WVF alert
                _sig_cards_html = ""
                for _, row in green_rows.iterrows():
                    _ticker = str(row.get("ticker", ""))
                    _company = str(row.get("company_name", ""))
                    _sector = str(row.get("sector", ""))
                    _price = row.get("current_price", "N/A")
                    _price_str = f"${float(_price):.2f}" if _price not in (None, "N/A", "") and str(_price) != "nan" else "N/A"
                    _wvf_val = row.get("wvf_value", "N/A")
                    _wvf_ub = row.get("wvf_upper_band", "N/A")
                    _wvf_rh = row.get("wvf_range_high", "N/A")
                    _wvf_date = str(row.get("wvf_last_green_date", "N/A"))
                    _wvf_trig = str(row.get("wvf_last_green_trigger", ""))

                    def _fmt_wvf(v):
                        try:
                            return f"{float(v):.4f}"
                        except Exception:
                            return str(v)

                    _verdict = (
                        "Strong bottom candidate — BOTH Bollinger and Percentile signals triggered."
                        if "BOTH" in _wvf_trig.upper()
                        else "Bottom signal — Williams Vix Fix green bar detected within the last 3 trading days."
                    )

                    _sig_cards_html += f"""
<div class="sig-card" style="--sig-label:'&#9679; SIGNAL'">
    <div class="sig-head">
        <span class="sig-code">{_ticker}</span>
        <span class="sig-sector">{_sector}</span>
    </div>
    <div style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;">{_company}</div>
    <div class="sig-row">
        <span class="sig-flow">Price</span>
        <span style="color:var(--cyan);">{_price_str}</span>
    </div>
    <div class="sig-row">
        <span class="sig-flow">WVF Value</span>
        <span>{_fmt_wvf(_wvf_val)}</span>
    </div>
    <div class="sig-row">
        <span class="sig-flow">Upper Band</span>
        <span>{_fmt_wvf(_wvf_ub)}</span>
    </div>
    <div class="sig-row">
        <span class="sig-flow">Range High</span>
        <span>{_fmt_wvf(_wvf_rh)}</span>
    </div>
    <div class="sig-row">
        <span class="sig-flow">Last Green Date</span>
        <span style="color:var(--amber);">{_wvf_date}</span>
    </div>
    <div class="sig-row">
        <span class="sig-flow">Trigger</span>
        <span class="inline-pill {'amber' if 'BOTH' in _wvf_trig.upper() else ''}">{_wvf_trig if _wvf_trig else 'N/A'}</span>
    </div>
    <div class="sig-verdict">{_verdict}</div>
</div>
"""
                st.markdown(_sig_cards_html, unsafe_allow_html=True)

                # detailed dataframe
                detail_cols = [
                    "ticker", "company_name", "sector", "business_nature",
                    "current_price", "wvf_green_last_3d",
                    "wvf_last_green_date", "wvf_last_green_trigger",
                    "wvf_last_green_value", "wvf_last_green_upper_band", "wvf_last_green_range_high",
                    "wvf_value", "wvf_upper_band", "wvf_range_high",
                ]
                cols = [c for c in detail_cols if c in green_rows.columns]
                st.dataframe(green_rows[cols] if cols else green_rows, use_container_width=True)
            else:
                st.info("WVF alert: no green bars in the last 3 trading days.")

        if buy_count > 0:
            st.warning("Action alert: at least one stock is in BUY zone.")
        elif sell_count > 0:
            st.error("Action alert: at least one stock is in SELL zone.")
        else:
            st.info("No BUY/SELL alerts yet. Current rules are still placeholders.")
    else:
        st.info("Select/upload a watchlist, then click Run monitor.")


# ── Tab 3: Saved Files ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Saved Watchlists</div>', unsafe_allow_html=True)
    files = list_saved_watchlists()
    if not files:
        st.info("No saved files yet. Run the screen in Tab 1 and save a passed list.")
    else:
        rows = []
        for p in files:
            try:
                df = pd.read_csv(p)
                _passed_in_file = int(df["pass"].sum()) if "pass" in df.columns else len(df)
                rows.append({
                    "File": p.name,
                    "Rows": len(df),
                    "Passed": _passed_in_file,
                    "Size (KB)": round(p.stat().st_size / 1024, 1),
                    "Path": str(p),
                })
            except Exception:
                rows.append({"File": p.name, "Rows": "—", "Passed": "—", "Size (KB)": "—", "Path": str(p)})
        _files_df = pd.DataFrame(rows)
        st.dataframe(
            _files_df[["File", "Rows", "Passed", "Size (KB)"]],
            use_container_width=True,
        )
        st.caption(f"{len(files)} watchlist file(s) found in {WATCHLIST_DIR}")


# ---------------------------------------------------------------------------
# Terminal footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="term-footer">
    <span>US-STOCK · Terminal</span>
    <span class="sep">│</span>
    <span>Screens using fundamental + technical template</span>
    <span class="sep">│</span>
    <span>Saves as <code>US stock dd-MMM-yyyy.csv</code></span>
    <span class="sep">│</span>
    <span style="color:var(--amber)">BUY / SELL / HOLD rules: active ? driven by buy_zone_rule() / sell_zone_rule() / hold_zone_rule()</span>
</div>
""", unsafe_allow_html=True)
