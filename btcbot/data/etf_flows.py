"""US spot bitcoin ETF daily net flows (US$ millions), as published by Farside Investors.

Farside sits behind Cloudflare, so the fetch uses cloudscraper and degrades gracefully to the
cached CSV. The CSV schema is: date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, GBTC, BTC, total
(one row per US trading day; negatives are outflows).

Look-ahead rule: a day's flow print is treated as *known* only at 16:00 ET + `publish_lag_hours`
(default 8h -> midnight ET), i.e. usable from the next day's pre-open session onward.
"""
from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ETF_COLS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]
ALL_COLS = ["date"] + ETF_COLS + ["total"]

_NUM_RE = re.compile(r"^\(?-?[\d,]*\.?\d+\)?$")


def parse_flow_value(v) -> float:
    """'(21.3)' -> -21.3, '1,234.5' -> 1234.5, '-' or '' -> 0.0."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return 0.0 if pd.isna(v) else float(v)
    s = str(v).strip().replace("−", "-").replace("US$", "").replace("$", "").replace("m", "")
    if s in ("", "-", "–", "—", "nan", "None"):
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return 0.0
    return -abs(x) if neg else x


def _flatten_columns(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if str(p) and not str(p).startswith("Unnamed")]
            c = parts[-1] if parts else ""
        out.append(str(c).strip())
    return out


def parse_farside_html(html: str) -> pd.DataFrame:
    """Find the flow table in Farside's HTML and return a tidy frame with ALL_COLS."""
    tables = pd.read_html(StringIO(html))
    cand = None
    for t in tables:
        cols = _flatten_columns(t.columns)
        if any(c.upper() == "IBIT" for c in cols) and any(c.upper() == "FBTC" for c in cols):
            cand = t.copy()
            cand.columns = cols
            break
    if cand is None:
        raise ValueError("no ETF flow table found in HTML")
    cand = cand.rename(columns={cand.columns[0]: "date"})
    upper = {c: c.upper() for c in cand.columns}
    cand = cand.rename(columns=upper).rename(columns={"DATE": "date", "TOTAL": "total"})
    cand["date"] = pd.to_datetime(cand["date"], format="%d %b %Y", errors="coerce")
    cand = cand.dropna(subset=["date"])  # drops Total/Average/Maximum footer rows
    for c in ETF_COLS + ["total"]:
        if c not in cand.columns:
            cand[c] = 0.0
        cand[c] = cand[c].map(parse_flow_value)
    out = cand[ALL_COLS].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    # Farside's total column is authoritative, but recompute if it is missing/zero
    calc = out[ETF_COLS].sum(axis=1)
    out.loc[out["total"] == 0.0, "total"] = calc[out["total"] == 0.0]
    return out


def fetch_farside_flows(url: str = "https://farside.co.uk/btc/", timeout: int = 60) -> pd.DataFrame:
    """Fetch through Cloudflare with cloudscraper. Raises on failure."""
    import cloudscraper  # lazy: optional at import time

    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "darwin", "desktop": True})
    r = scraper.get(url, timeout=timeout)
    if r.status_code != 200 or "IBIT" not in r.text:
        raise RuntimeError(f"farside fetch blocked: status={r.status_code} len={len(r.text)}")
    return parse_farside_html(r.text)


def load_flows_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ETF_COLS + ["total"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[ALL_COLS].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def save_flows_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def update_flows_csv(path: Path, url: str) -> tuple[pd.DataFrame | None, bool]:
    """Merge a fresh Farside pull into the CSV. Returns (frame, updated). Never raises."""
    existing = load_flows_csv(path)
    try:
        fresh = fetch_farside_flows(url)
    except Exception as e:  # noqa: BLE001 — degrade to cache
        log.warning("ETF flow refresh failed (%s); using cached CSV with %s rows", e, 0 if existing is None else len(existing))
        return existing, False
    merged = fresh if existing is None else pd.concat([existing, fresh]).sort_values("date").drop_duplicates("date", keep="last")
    merged = merged.reset_index(drop=True)
    save_flows_csv(merged, path)
    log.info("ETF flows updated: %d rows, latest %s", len(merged), merged["date"].iloc[-1].date())
    return merged, True


def flows_known_at(flows: pd.DataFrame, publish_lag_hours: float, tz: str = "America/New_York") -> pd.DataFrame:
    """Add `known_at` (UTC): 16:00 ET on the flow date + publish lag. Adds rolling 5d/20d totals too."""
    df = flows.sort_values("date").reset_index(drop=True).copy()
    close_et = pd.to_datetime(df["date"]).dt.tz_localize(tz) + pd.Timedelta(hours=16)
    df["known_at"] = (close_et + pd.Timedelta(hours=publish_lag_hours)).dt.tz_convert("UTC")
    df["total_5d"] = df["total"].rolling(5, min_periods=1).sum()
    df["total_20d"] = df["total"].rolling(20, min_periods=1).sum()
    nz = (df[ETF_COLS] != 0).sum(axis=1).replace(0, 1)
    df["pos_frac"] = (df[ETF_COLS] > 0).sum(axis=1) / nz
    return df
