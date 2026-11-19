"""BTC candle fetching.

Historical spot candles come from Coinbase (or Bitstamp) because Hyperliquid's public
candle endpoint only returns the most recent ~5000 bars. Hyperliquid candles are used
for live sanity checks (perp/spot basis) and are available for the recent window.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
COLS = ["open", "high", "low", "close", "volume"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "btc-transformer-research/0.1", "Accept": "application/json"})
    return s


def _request_json(session: requests.Session, method: str, url: str, retries: int = 6, **kw):
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.request(method, url, timeout=30, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(30.0, 1.5 ** attempt)
                log.warning("%s %s -> %s, retrying in %.1fs", method, url, r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # network hiccup
            last_err = e
            time.sleep(min(30.0, 1.5 ** attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {method} {url}: {last_err}")


def _to_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def fetch_coinbase(start: datetime, end: datetime, granularity: int = 900, product: str = "BTC-USD") -> pd.DataFrame:
    """Coinbase Exchange public candles. Max 300 candles per request. Returns OHLCV indexed by UTC bar start."""
    start, end = _to_utc(start), _to_utc(end)
    s = _session()
    rows: list[tuple] = []
    step = timedelta(seconds=granularity * 300)
    t0 = start
    n_req = 0
    while t0 < end:
        t1 = min(t0 + step, end)
        data = _request_json(
            s, "GET", COINBASE_URL.format(product=product),
            params={"granularity": granularity, "start": t0.isoformat(), "end": t1.isoformat()},
        )
        if isinstance(data, dict):  # error payload
            raise RuntimeError(f"coinbase error: {data}")
        # Coinbase row order: [time, low, high, open, close, volume]
        rows.extend((r[0], r[3], r[2], r[1], r[4], r[5]) for r in data)
        t0 = t1
        n_req += 1
        time.sleep(0.12)
    log.info("coinbase: %d requests, %d raw rows", n_req, len(rows))
    return _frame(rows)


def fetch_bitstamp(start: datetime, end: datetime, step: int = 900, pair: str = "btcusd") -> pd.DataFrame:
    """Bitstamp public OHLC. Max 1000 candles per request."""
    start, end = _to_utc(start), _to_utc(end)
    s = _session()
    rows: list[tuple] = []
    t0, end_ts = int(start.timestamp()), int(end.timestamp())
    while t0 < end_ts:
        d = _request_json(s, "GET", BITSTAMP_URL.format(pair=pair), params={"step": step, "limit": 1000, "start": t0})
        ohlc = d.get("data", {}).get("ohlc", [])
        if not ohlc:
            break
        rows.extend(
            (int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in ohlc
        )
        last = int(ohlc[-1]["timestamp"])
        if last <= t0:
            break
        t0 = last + step
        time.sleep(0.2)
    return _frame(rows)


def fetch_hyperliquid_candles(coin: str = "BTC", interval: str = "1h", start_ms: int | None = None,
                              end_ms: int | None = None, testnet: bool = False) -> pd.DataFrame:
    """Hyperliquid candleSnapshot (recent history only, ~5000 bars)."""
    url = HL_TESTNET_INFO_URL if testnet else HL_INFO_URL
    now_ms = int(time.time() * 1000)
    end_ms = end_ms or now_ms
    start_ms = start_ms or (end_ms - 5000 * _interval_ms(interval))
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms}}
    data = _request_json(_session(), "POST", url, json=payload)
    rows = [(int(c["t"]) // 1000, float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"])) for c in data]
    return _frame(rows)


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}[unit]
    return n * mult


def _frame(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows, columns=["time"] + COLS).drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    df = df.set_index("time").astype(float)
    df.index.name = "time"
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "time"
    return df[COLS].astype(float)


def load_or_fetch_prices(path: Path, start: datetime, end: datetime, source: str = "coinbase",
                         granularity: int = 900, update: bool = True) -> pd.DataFrame:
    """Load cached candles, extend them to `end` (if update), persist, return."""
    start, end = _to_utc(start), _to_utc(end)
    fetch = {"coinbase": fetch_coinbase, "bitstamp": fetch_bitstamp}[source]
    existing = _read_csv(path) if path.exists() else None
    if existing is not None and len(existing):
        if not update:
            return existing
        last = existing.index[-1].to_pydatetime()
        fetch_from = last - timedelta(seconds=granularity * 3)  # re-fetch the last few (possibly partial) bars
        if fetch_from >= end:
            return existing
        new = fetch(fetch_from, end, granularity)
        df = pd.concat([existing, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = fetch(start, end, granularity)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")
    return df


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample OHLCV to a coarser bar (label = bar start)."""
    rule = f"{minutes}min"
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def fill_gaps(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Reindex to a complete grid; a missing bar becomes a zero-volume bar at the previous close."""
    idx = pd.date_range(df.index[0], df.index[-1], freq=f"{minutes}min", tz="UTC")
    out = df.reindex(idx)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "time"
    return out
