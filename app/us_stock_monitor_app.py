
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
WATCHLIST_DIR.mkdir(exist_ok=True, parents=True)
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
    fundamental_mode: str = "gemini_yoy"
    apply_technical_filter: bool = False
    min_fundamental_rules_pass: int = 2


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


def build_profile_map(tickers: List[str]) -> Dict[str, Dict[str, str]]:
    prof = load_sp500_profiles()
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
        out[t] = item
    # Fill missing profile info from yfinance for tickers not in S&P 500 list (for example forced includes).
    for t in tickers:
        if out[t].get("company_name") == "N/A" or out[t].get("sector") == "N/A" or out[t].get("business_nature") == "N/A":
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
        threads=True,
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
        recent_green = (
            recent_3.get("WVF_GREEN", pd.Series(dtype=bool)).fillna(False).astype(bool)
            if enable_wvf_alert
            else pd.Series(dtype=bool)
        )
        wvf_green_last_3d = bool(recent_green.any()) if enable_wvf_alert else False
        rows.append({
            "ticker": ticker,
            "action": derive_action_status(ticker, df, monitor_date),
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
            "monitor_note": "Rules pending",
        })

    return watchlist_df.merge(pd.DataFrame(rows), on="ticker", how="left")


st.set_page_config(page_title="US Stock Monitor", layout="wide")
st.title("US Stock Monitor")
st.caption("Screen on a chosen date, save the selected list, then monitor it daily until buy/sell/hold rules are added.")

with st.sidebar:
    st.header("Input")
    use_offline_sample_data = st.checkbox(
        "Use offline sample data (no live API)",
        value=False,
        help="Loads bundled sample scan results from data/full_screen_latest.csv and skips live market/fundamental calls.",
    )
    tickers_text = st.text_area(
        "Custom tickers (optional, comma-separated)",
        value="",
        help=f"Leave blank to auto-use top {AUTO_UNIVERSE_TOP_N} by average trading volume from combined NASDAQ + S&P 500 universe.",
    )
    screen_date = st.date_input("Screen date", value=date.today())
    monitor_date = st.date_input("Monitor date", value=date.today())
    always_include_text = st.text_input(
        "Always include tickers",
        value="CLS,JBL",
        help="These tickers are forced into the auto universe even if they are not in the top-volume ranking source.",
    )
    fundamentals_file = st.file_uploader("Optional fundamentals snapshot CSV", type=["csv"])
    enable_price_audit = st.checkbox("Enable free price audit (Stooq)", value=False)
    audit_limit = st.number_input("Audit max tickers", min_value=10, max_value=1000, value=100, step=10)
    enable_live_fundamentals = st.checkbox("Auto-fill fundamentals values (Yahoo free)", value=False)
    enable_sec_audit = st.checkbox("Enable 2nd fundamentals audit (SEC free)", value=False)
    sec_audit_limit = st.number_input("SEC audit max tickers", min_value=10, max_value=300, value=50, step=10)

    st.header("Rule")
    with st.expander("Fundamental", expanded=False):
        fundamental_mode_label = st.selectbox(
            "Fundamental setup",
            ["Gemini match (YoY)", "TradingView setup (QoQ)"],
            index=0,
        )
        min_eps_yoy = st.number_input("Min EPS YoY (%)", value=20.0, step=1.0)
        min_revenue_yoy = st.number_input("Min Revenue YoY (%)", value=20.0, step=1.0)
        min_roe_avg = st.number_input("Min 3Y average ROE (%)", value=15.0, step=1.0)
        roe_stability_max_std = st.number_input("ROE stability max std-dev", value=40.0, step=1.0)
        min_market_cap_b = st.number_input("Min Market Cap (B USD)", value=10.0, step=1.0)
        min_fundamental_rules_pass = st.number_input("Min passed fundamental rules (out of 3)", min_value=1, max_value=3, value=2, step=1)
        require_fundamentals = st.checkbox("Require fundamentals for pass", value=True)

    with st.expander("Technical-Lazy", expanded=False):
        st.caption("MA checks use: MA50, MA150, MA200")
        apply_technical_filter = st.checkbox("Apply Technical-Lazy filter in scan", value=False)
        require_close_above_mas = st.checkbox("Require Close > MA50, MA150, MA200", value=True)
        require_ma_stack = st.checkbox("Require MA50 > MA150 > MA200", value=True)
        high_52w_within_pct = st.number_input("Within 52W high (%)", value=25.0, step=1.0)
        low_52w_above_pct = st.number_input("Above 52W low (%)", value=25.0, step=1.0)

    with st.expander("Technical-VIX", expanded=False):
        st.caption("Separate option: Williams Vix Fix alert on watchlist monitor")
        enable_wvf_alert = st.checkbox("Enable Williams Vix Fix alert", value=False)
        wvf_pd = st.number_input("WVF LookBack Period Standard Deviation High", min_value=5, max_value=120, value=22, step=1)
        wvf_bbl = st.number_input("WVF Bollinger Band Length", min_value=5, max_value=120, value=20, step=1)
        wvf_mult = st.number_input("WVF Bollinger Band Std Dev Mult", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        wvf_lb = st.number_input("WVF Look Back Period Percentile High", min_value=10, max_value=250, value=50, step=1)
        wvf_ph = st.number_input("WVF Highest Percentile", min_value=0.50, max_value=1.20, value=0.85, step=0.01)
        wvf_pl = st.number_input("WVF Lowest Percentile", min_value=0.90, max_value=1.50, value=1.01, step=0.01)

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
    try:
        base_tickers, universe_preview = top_volume_universe(
            as_of_ts.strftime("%Y-%m-%d"),
            top_n=AUTO_UNIVERSE_TOP_N,
            lookback_days=60,
        )
        tickers = apply_always_include(base_tickers, always_include, top_n=AUTO_UNIVERSE_TOP_N)
        actually_forced = [t for t in always_include if t in tickers]
        universe_note = (
            f"Using auto universe: top {AUTO_UNIVERSE_TOP_N} stocks by average daily trading volume from NASDAQ + S&P 500 "
            "over the latest 60 trading days."
        )
        if actually_forced:
            universe_note += f" Forced include: {', '.join(actually_forced)}."
    except Exception as exc:
        tickers = []
        universe_note = f"Auto universe load failed: {exc}"

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
    fundamental_mode="gemini_yoy" if fundamental_mode_label == "Gemini match (YoY)" else "tradingview_qoq",
    apply_technical_filter=bool(apply_technical_filter),
    min_fundamental_rules_pass=int(min_fundamental_rules_pass),
)

try:
    fdf = load_fundamentals_snapshot(fundamentals_file)
except Exception as exc:
    st.error(f"Failed to read fundamentals CSV: {exc}")
    fdf = None

tab1, tab2, tab3 = st.tabs(["1. Screen & Save", "2. Monitor Watchlist", "3. Saved Files"])

with tab1:
    st.subheader("Screen stocks on selected date")
    st.caption(
        "Fundamental rule fields depend on selected setup: "
        "Gemini match uses YoY EPS/Revenue (latest-quarter momentum + relaxed continuity) + 3Y ROE + market cap; "
        "TradingView setup uses QoQ EPS/Revenue (latest-quarter momentum + relaxed continuity) + quarterly TTM-style ROE + market cap. "
        "Market Cap must be above minimum threshold."
    )
    st.caption("Pass logic: require at least N of 3 fundamental rules (EPS, Revenue, ROE), plus market cap.")
    st.caption("ROE formula: Net Income / Shareholders' Equity")
    last_run_date = st.session_state.get("screen_result_date")
    if last_run_date:
        st.caption(f"Last screen run date: {last_run_date}")
    if last_run_date and last_run_date != selected_screen_date_str:
        st.warning("Screen date changed. Please click Run screen to refresh results for the newly selected date.")
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
        st.error(universe_note)

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
            elif not tickers:
                st.error("No valid universe available. Check network or enter custom tickers.")
            else:
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
                        profile_map=profile_map,
                        enable_audit=bool(enable_price_audit),
                        audit_limit=int(audit_limit),
                        enable_live_fundamentals=bool(enable_live_fundamentals),
                        enable_sec_audit=bool(enable_sec_audit),
                        sec_audit_limit=int(sec_audit_limit),
                    )
                    st.session_state["screen_result_date"] = selected_screen_date_str
                    st.session_state["screen_result_mode"] = "live_scan"
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
                    result_date_str = st.session_state.get("screen_result_date", selected_screen_date_str)
                    path = save_watchlist(passed, pd.Timestamp(result_date_str))
                    st.success(f"Saved: {path.name}")

    screen_df = st.session_state.get("screen_df")
    if screen_df is not None and not screen_df.empty:
        display_df = screen_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
        display_df = display_df.replace({np.nan: "N/A"})
        st.dataframe(display_df, use_container_width=True)
        passed = display_df[display_df["pass"] == True]
        st.markdown(f"**Passed stocks:** {len(passed)}")
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

with tab2:
    st.subheader("Monitor a saved watchlist")
    files = list_saved_watchlists()
    uploaded_watchlist = st.file_uploader("Or upload watchlist CSV", type=["csv"], key="monitor_upload")

    selected_path: Optional[Path] = None
    if files:
        options = {p.name: p for p in files}
        selected_name = st.selectbox("Select saved watchlist", list(options.keys()))
        selected_path = options[selected_name]
    else:
        st.info("No saved watchlist files found. Upload a CSV below, or save one from tab 1 first.")

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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BUY", buy_count)
        c2.metric("SELL", sell_count)
        c3.metric("HOLD", hold_count)
        c4.metric("WATCH", watch_count)

        if bool(enable_wvf_alert):
            wvf_green_today_count = int((monitor_df["wvf_green_alert"] == True).sum())
            wvf_green_last3_count = int((monitor_df.get("wvf_green_last_3d", False) == True).sum())
            c5, c6 = st.columns(2)
            c5.metric("WVF Green (Monitor Date)", wvf_green_today_count)
            c6.metric("WVF Green (Last 3 Trading Days)", wvf_green_last3_count)

            if wvf_green_last3_count > 0:
                green_names = monitor_df.loc[monitor_df["wvf_green_last_3d"] == True, "ticker"].astype(str).tolist()
                st.success(
                    f"WVF alert: {wvf_green_last3_count} ticker(s) showed a green signal within the last 3 trading days: "
                    + ", ".join(green_names)
                )
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
