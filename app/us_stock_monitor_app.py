
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
    roe_stability_max_std: float = 8.0


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

    # Source 3: Small local fallback so app remains usable offline/restricted
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
    to_pct = lambda v: float(v) * 100.0 if v is not None and pd.notna(v) else np.nan
    return {
        "company_name": company_name,
        "sector": sector,
        "business_nature": business_nature,
        "eps_yoy_value": to_pct(eps_yoy_value),
        "rev_yoy_value": to_pct(rev_yoy_value),
        "roe_avg": to_pct(roe_avg),
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
            df = df.sort_values("end")
            yoy_vals: List[float] = []
            for _, latest in df.iterrows():
                prior = df[(df["fy"] == latest["fy"] - 1) & (df["fp"] == latest["fp"])]
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
        roe = (merged["net_income"] / merged["equity"]) * 100.0
        return float(roe.mean())
    return np.nan


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
        roe_avg = _sec_pick_roe_avg(facts, as_of)
        return {
            "eps_yoy_list": eps_yoy_list,
            "rev_yoy_list": rev_yoy_list,
            "rev_fy_list": rev_fy_list,
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
    tickers = load_sp500_tickers()
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
    sp500_profiles = load_sp500_profiles()
    vol_df = vol_df.merge(sp500_profiles, on="ticker", how="left")
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
    def _max_consecutive_ge(series: pd.Series, threshold: float) -> int:
        run = 0
        best = 0
        for v in series.dropna().tolist():
            if float(v) >= threshold:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        return best

    row = get_fund_row(ticker, as_of_date, fdf)
    source = "csv"
    eps_vals = pd.Series(dtype=float)
    rev_vals = pd.Series(dtype=float)
    rev_fy_vals = pd.Series(dtype=float)
    roe_vals = pd.Series(dtype=float)
    roe_avg = np.nan
    note = ""
    fundamentals_used = True

    if row is not None:
        eps_cols = [c for c in ["eps_q1_yoy", "eps_q2_yoy", "eps_q3_yoy", "eps_q4_yoy"] if c in row.index]
        rev_cols = [c for c in ["revenue_q1_yoy", "revenue_q2_yoy", "revenue_q3_yoy", "revenue_q4_yoy"] if c in row.index]
        roe_cols = [c for c in ["roe_y1", "roe_y2", "roe_y3"] if c in row.index]
        rev_fy_cols = [c for c in ["revenue_y1", "revenue_y2", "revenue_y3"] if c in row.index]
        eps_vals = pd.to_numeric(row[eps_cols], errors="coerce") if eps_cols else pd.Series(dtype=float)
        rev_vals = pd.to_numeric(row[rev_cols], errors="coerce") if rev_cols else pd.Series(dtype=float)
        rev_fy_vals = pd.to_numeric(row[rev_fy_cols], errors="coerce") if rev_fy_cols else pd.Series(dtype=float)
        roe_vals = pd.to_numeric(row[roe_cols], errors="coerce") if roe_cols else pd.Series(dtype=float)
        roe_avg = float(roe_vals.mean()) if len(roe_vals) >= 1 else np.nan
    else:
        sec = load_sec_fundamentals_for_rules(ticker, as_of_date.strftime("%Y-%m-%d"))
        if sec:
            source = "sec_companyfacts"
            eps_vals = pd.Series(sec.get("eps_yoy_list", []), dtype=float)
            rev_vals = pd.Series(sec.get("rev_yoy_list", []), dtype=float)
            rev_fy_vals = pd.Series(sec.get("rev_fy_list", []), dtype=float)
            roe_avg = float(sec.get("roe_avg", np.nan)) if sec.get("roe_avg", np.nan) is not None else np.nan
        else:
            live = load_live_fundamentals_yf(ticker)
            source = "yahoo_proxy"
            eps_val = live.get("eps_yoy_value", np.nan)
            rev_val = live.get("rev_yoy_value", np.nan)
            roe_avg = live.get("roe_avg", np.nan)
            eps_vals = pd.Series([eps_val] if pd.notna(eps_val) else [], dtype=float)
            rev_vals = pd.Series([rev_val] if pd.notna(rev_val) else [], dtype=float)

    eps_ge_min_qtrs = int((eps_vals >= cfg.min_eps_yoy).sum()) if len(eps_vals) > 0 else 0
    rev_ge_min_qtrs = int((rev_vals >= cfg.min_revenue_yoy).sum()) if len(rev_vals) > 0 else 0
    eps_pass = eps_ge_min_qtrs >= 3 if len(eps_vals) >= 3 else False
    eps_consec_qtrs = _max_consecutive_ge(eps_vals, cfg.min_eps_yoy)
    rev_consec_qtrs = _max_consecutive_ge(rev_vals, cfg.min_revenue_yoy)
    annual_revenue_uptrend = bool(
        len(rev_fy_vals.dropna()) >= 3 and
        rev_fy_vals.dropna().iloc[-1] > rev_fy_vals.dropna().iloc[-2] > rev_fy_vals.dropna().iloc[-3]
    )
    rev_pass = (rev_consec_qtrs >= 3) or annual_revenue_uptrend
    roe_pass = roe_avg >= cfg.min_roe_avg if pd.notna(roe_avg) else False
    roe_std = float(roe_vals.dropna().std()) if len(roe_vals.dropna()) >= 2 else np.nan
    roe_stable = (roe_std <= cfg.roe_stability_max_std) if pd.notna(roe_std) else True
    roe_pass = bool(roe_pass and roe_stable)
    roe_quality = "better(>17)" if pd.notna(roe_avg) and roe_avg > 17 else "ok(>15)"
    eps_yoy_value = float(eps_vals.dropna().iloc[-1]) if not eps_vals.dropna().empty else np.nan
    rev_yoy_value = float(rev_vals.dropna().iloc[-1]) if not rev_vals.dropna().empty else np.nan

    if source == "yahoo_proxy":
        note = "Yahoo proxy fundamentals (not 4-quarter SEC/CSV series)"
        fundamentals_used = True
    elif source == "sec_companyfacts":
        note = "Fundamentals from SEC Company Facts"
        fundamentals_used = True
    elif source == "csv":
        note = "Fundamentals from uploaded CSV"
        fundamentals_used = True

    # If strict fundamentals are required and only proxy data exists, enforce failure.
    strict_depth_ok = len(eps_vals) >= 3 and (len(rev_vals) >= 3 or len(rev_fy_vals.dropna()) >= 3) and pd.notna(roe_avg)
    if source == "yahoo_proxy" and cfg.require_fundamentals:
        strict_depth_ok = False

    fund_final_pass = bool(eps_pass and rev_pass and roe_pass and strict_depth_ok)
    if not fund_final_pass and not note:
        note = "Insufficient fundamentals depth"

    return fund_final_pass if cfg.require_fundamentals else True, {
        "fundamentals_used": fundamentals_used,
        "fundamentals_source": source,
        "eps_pass": bool(eps_pass),
        "rev_pass": bool(rev_pass),
        "roe_pass": bool(roe_pass),
        "roe_avg": roe_avg,
        "eps_yoy_value": eps_yoy_value,
        "rev_yoy_value": rev_yoy_value,
        "eps_ge_min_qtrs": eps_ge_min_qtrs,
        "rev_ge_min_qtrs": rev_ge_min_qtrs,
        "eps_consec_qtrs": eps_consec_qtrs,
        "rev_consec_qtrs": rev_consec_qtrs,
        "annual_revenue_uptrend": annual_revenue_uptrend,
        "roe_std": roe_std,
        "roe_stable": roe_stable,
        "roe_quality": roe_quality,
        "eps_vals": list(eps_vals.values),
        "rev_vals": list(rev_vals.values),
        "note": note,
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
            if profile.get("company_name") == "N/A":
                profile["company_name"] = str(live.get("company_name", "N/A"))
            if profile.get("sector") == "N/A":
                profile["sector"] = str(live.get("sector", "N/A"))
            if profile.get("business_nature") == "N/A":
                profile["business_nature"] = str(live.get("business_nature", "N/A"))
        final_pass = tech_pass and fund_pass

        score = 0
        score += 1 if tech.get("above_mas") else 0
        score += 1 if tech.get("ma_stack") else 0
        score += 1 if tech.get("within_high") else 0
        score += 1 if tech.get("above_low") else 0
        score += 1 if fund.get("eps_pass") else 0
        score += 1 if fund.get("rev_pass") else 0
        score += 1 if fund.get("roe_pass") else 0
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
            "roe_3y_avg_value": fund.get("roe_avg", np.nan),
            "eps_ge_min_qtrs": fund.get("eps_ge_min_qtrs", np.nan),
            "rev_ge_min_qtrs": fund.get("rev_ge_min_qtrs", np.nan),
            "eps_consec_qtrs": fund.get("eps_consec_qtrs", np.nan),
            "rev_consec_qtrs": fund.get("rev_consec_qtrs", np.nan),
            "annual_revenue_uptrend": fund.get("annual_revenue_uptrend", np.nan),
            "eps_pass": fund.get("eps_pass", np.nan),
            "rev_pass": fund.get("rev_pass", np.nan),
            "roe_pass": fund.get("roe_pass", np.nan),
            "roe_std": fund.get("roe_std", np.nan),
            "roe_stable": fund.get("roe_stable", np.nan),
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
    tickers_text = st.text_area(
        "Custom tickers (optional, comma-separated)",
        value="",
        help=f"Leave blank to auto-use top {AUTO_UNIVERSE_TOP_N} by average trading volume from S&P 500 constituents.",
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
    enable_live_fundamentals = st.checkbox("Auto-fill fundamentals values (Yahoo free)", value=True)
    enable_sec_audit = st.checkbox("Enable 2nd fundamentals audit (SEC free)", value=False)
    sec_audit_limit = st.number_input("SEC audit max tickers", min_value=10, max_value=300, value=50, step=10)

    st.header("Rule")
    with st.expander("Fundamental", expanded=False):
        min_eps_yoy = st.number_input("Min EPS YoY (%)", value=20.0, step=1.0)
        min_revenue_yoy = st.number_input("Min Revenue YoY (%)", value=20.0, step=1.0)
        min_roe_avg = st.number_input("Min 3Y average ROE (%)", value=15.0, step=1.0)
        require_fundamentals = st.checkbox("Require fundamentals for pass", value=True)

    with st.expander("Technical-Lazy", expanded=False):
        st.caption("MA checks use: MA50, MA150, MA200")
        require_close_above_mas = st.checkbox("Require Close > MA50, MA150, MA200", value=True)
        require_ma_stack = st.checkbox("Require MA50 > MA150 > MA200", value=True)
        high_52w_within_pct = st.number_input("Within 52W high (%)", value=25.0, step=1.0)
        low_52w_above_pct = st.number_input("Above 52W low (%)", value=25.0, step=1.0)

as_of_ts = pd.Timestamp(screen_date)
monitor_ts = pd.Timestamp(monitor_date)

manual_tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
always_include = [x.strip().upper() for x in always_include_text.split(",") if x.strip()]
universe_note = ""
universe_preview = pd.DataFrame()
if manual_tickers:
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
            f"Using auto universe: top {AUTO_UNIVERSE_TOP_N} S&P 500 stocks by average daily trading volume "
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
        "Fundamental rule fields: `eps_ge_min_qtrs` / `eps_consec_qtrs`; "
        "`rev_consec_qtrs` or `annual_revenue_uptrend`; "
        "`roe_3y_avg_value` with `roe_stable`."
    )
    st.caption("ROE formula: Net Income / Shareholders' Equity")
    if tickers:
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
            if not tickers:
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
        display_df = screen_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
        display_df = display_df.replace({np.nan: "N/A"})
        st.dataframe(display_df, use_container_width=True)
        passed = display_df[display_df["pass"] == True]
        st.markdown(f"**Passed stocks:** {len(passed)}")
        if not passed.empty:
            st.dataframe(
                passed[
                    [
                        "ticker",
                        "company_name",
                        "sector",
                        "business_nature",
                        "score",
                        "technical_pass",
                        "fundamental_pass",
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
                        "close",
                        "ret_63d",
                        "audit_close_stooq",
                        "audit_close_diff_pct",
                    ]
                ],
                use_container_width=True,
            )
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
