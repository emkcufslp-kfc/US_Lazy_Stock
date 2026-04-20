from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "reference_validation_report.json"
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-]*$")

SOURCES: Dict[str, Dict[str, object]] = {
    "sp500_constituents": {
        "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "type": "csv",
        "required_columns": ["Symbol", "Security", "GICS Sector"],
    },
    "sp500_financials": {
        "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/main/data/constituents-financials.csv",
        "type": "csv",
        "required_columns": ["Symbol", "Name", "Sector"],
    },
    "ate329_all": {
        "url": "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/all.csv",
        "type": "csv",
        "required_columns": ["symbol", "name", "industry"],
    },
    "ate329_sp500": {
        "url": "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv",
        "type": "csv",
        "required_columns": ["symbol", "name", "industry"],
    },
    "rreichel_nasdaq": {
        "url": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt",
        "type": "txt_ticker",
    },
    "rreichel_nyse": {
        "url": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt",
        "type": "txt_ticker",
    },
    "rreichel_amex": {
        "url": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt",
        "type": "txt_ticker",
    },
    "us_market_data": {
        "url": "https://raw.githubusercontent.com/SteelCerberus/us-market-data/main/data/us_market_data.csv",
        "type": "csv",
        "required_columns": ["Date", "Close", "Adjusted Close"],
    },
    "sp500_macro": {
        "url": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
        "type": "csv",
        "required_columns": ["Date", "SP500", "Dividend", "Earnings"],
    },
}


def _fetch(url: str, timeout: int = 20) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "USLazyStock/1.0"})
    r.raise_for_status()
    return r


def _validate_csv(text: str, required_columns: List[str]) -> Tuple[bool, Dict[str, object], List[Dict[str, str]]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    columns = list(rows[0].keys()) if rows else []
    missing = [c for c in required_columns if c not in columns]
    ok = bool(rows) and not missing
    meta = {
        "rows": len(rows),
        "columns": columns,
        "missing_columns": missing,
    }
    return ok, meta, rows


def _validate_txt_tickers(text: str) -> Tuple[bool, Dict[str, object], List[str]]:
    lines = [x.strip().upper().replace(".", "-") for x in text.splitlines() if x.strip()]
    invalid = [x for x in lines if not TICKER_RE.match(x)]
    unique = sorted(set(lines))
    ok = bool(unique) and len(invalid) == 0
    meta = {
        "rows": len(lines),
        "unique_rows": len(unique),
        "invalid_rows": len(invalid),
    }
    return ok, meta, unique


def _ticker_set(rows: List[Dict[str, str]], key: str) -> set[str]:
    out = set()
    for r in rows:
        t = str(r.get(key, "")).upper().strip().replace(".", "-")
        if TICKER_RE.match(t):
            out.add(t)
    return out


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    results: Dict[str, object] = {
        "generated_at_utc": now,
        "sources": {},
        "cross_checks": {},
        "overall_ok": False,
    }

    raw_cache: Dict[str, object] = {}
    all_ok = True

    for name, cfg in SOURCES.items():
        url = str(cfg["url"])
        typ = str(cfg["type"])
        entry: Dict[str, object] = {"url": url, "ok": False}
        try:
            resp = _fetch(url)
            entry["status_code"] = resp.status_code
            entry["bytes"] = len(resp.content)
            if typ == "csv":
                ok, meta, rows = _validate_csv(resp.text, list(cfg["required_columns"]))
                entry.update(meta)
                entry["ok"] = ok
                raw_cache[name] = rows
            elif typ == "txt_ticker":
                ok, meta, items = _validate_txt_tickers(resp.text)
                entry.update(meta)
                entry["ok"] = ok
                raw_cache[name] = items
            else:
                entry["error"] = f"Unsupported type: {typ}"
                all_ok = False
        except Exception as exc:
            entry["error"] = str(exc)
            all_ok = False

        if not entry.get("ok", False):
            all_ok = False
        results["sources"][name] = entry

    # Cross-check S&P 500 symbol consistency between two independent repos.
    sp_a = _ticker_set(raw_cache.get("sp500_constituents", []), "Symbol")
    sp_b = _ticker_set(raw_cache.get("ate329_sp500", []), "symbol")
    overlap = len(sp_a & sp_b)
    union = len(sp_a | sp_b)
    jaccard = (overlap / union) if union else 0.0
    sp_check_ok = bool(sp_a) and bool(sp_b) and (jaccard >= 0.80)
    results["cross_checks"]["sp500_symbol_overlap"] = {
        "ok": sp_check_ok,
        "source_a_count": len(sp_a),
        "source_b_count": len(sp_b),
        "overlap_count": overlap,
        "jaccard": round(jaccard, 4),
        "rule": "jaccard >= 0.80",
    }
    if not sp_check_ok:
        all_ok = False

    # Sanity check market history sizes.
    us_rows = len(raw_cache.get("us_market_data", []))
    macro_rows = len(raw_cache.get("sp500_macro", []))
    hist_ok = us_rows > 20000 and macro_rows > 1000
    results["cross_checks"]["history_row_sanity"] = {
        "ok": hist_ok,
        "us_market_data_rows": us_rows,
        "sp500_macro_rows": macro_rows,
        "rule": "us_market_data_rows > 20000 and sp500_macro_rows > 1000",
    }
    if not hist_ok:
        all_ok = False

    results["overall_ok"] = all_ok
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"overall_ok": all_ok, "report": str(REPORT_PATH)}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
