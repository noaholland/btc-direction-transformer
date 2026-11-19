"""Descriptive day-of-week x session analysis: what happens if you buy BTC before / at / after
the NYSE open, by weekday, and what a 40x isolated position would have suffered along the way.

No model involved — this is the raw seasonality the model is asked to learn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_et import DOW_NAMES, is_trading_day
from .config import resolve
from .data.etf_flows import flows_known_at, load_flows_csv
from .kelly import liquidation_distance

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionDef:
    name: str
    start: str      # HH:MM ET
    end: str        # HH:MM ET
    end_day_offset: int = 0
    start_day_offset: int = 0
    note: str = ""


SESSIONS = [
    SessionDef("overnight_16to0930", "16:00", "09:30", start_day_offset=-1, note="prev close -> today's open"),
    SessionDef("preopen_0to0930", "00:00", "09:30", note="midnight -> open"),
    SessionDef("last_hour_preopen", "08:30", "09:30", note="the hour before the bell"),
    SessionDef("first_hour_open", "09:30", "10:30", note="the hour after the bell"),
    SessionDef("ny_session_0930to16", "09:30", "16:00", note="buy at the open, sell at the close"),
    SessionDef("post_open_10to16", "10:00", "16:00", note="buy 30 min after the open, sell at the close"),
    SessionDef("after_hours_16to24", "16:00", "24:00", note="close -> midnight"),
    SessionDef("full_day_0to24", "00:00", "24:00", note="calendar day (ET)"),
    SessionDef("buy_0900_hold_8h", "09:00", "17:00", note="bot's pre-open trade with the 8h horizon"),
    SessionDef("buy_1000_hold_8h", "10:00", "18:00", note="bot's post-open trade with the 8h horizon"),
    SessionDef("buy_0900_hold_24h", "09:00", "09:00", end_day_offset=1, note="pre-open, hold a day"),
    SessionDef("buy_1000_hold_24h", "10:00", "10:00", end_day_offset=1, note="post-open, hold a day"),
]


def _hm_to_td(hm: str) -> pd.Timedelta:
    h, m = hm.split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m))


def session_table(bars: pd.DataFrame, bar_minutes: int, tz: str, sess: SessionDef) -> pd.DataFrame:
    """One row per ET calendar date: entry/exit price, log return, path min/max relative to entry."""
    end_ts = bars.index + pd.Timedelta(minutes=bar_minutes)      # bar end timestamps (UTC)
    px = pd.Series(bars["close"].to_numpy(), index=end_ts)        # price known at bar end
    lows = bars["low"].to_numpy()
    highs = bars["high"].to_numpy()
    end_np = end_ts.asi8

    et_dates = pd.DatetimeIndex(end_ts.tz_convert(tz).normalize().unique())
    dates = et_dates.tz_localize(None)
    a_local = dates + pd.Timedelta(days=1) * 0 + _hm_to_td(sess.start) + pd.Timedelta(days=sess.start_day_offset)
    b_local = dates + _hm_to_td(sess.end) + pd.Timedelta(days=sess.end_day_offset)
    a_utc = a_local.tz_localize(tz, nonexistent="shift_forward", ambiguous=True).tz_convert("UTC")
    b_utc = b_local.tz_localize(tz, nonexistent="shift_forward", ambiguous=True).tz_convert("UTC")

    pa = px.reindex(a_utc).to_numpy()
    pb = px.reindex(b_utc).to_numpy()
    # path stats over bars whose end is in (A, B]
    ia = np.searchsorted(end_np, a_utc.asi8, side="right")
    ib = np.searchsorted(end_np, b_utc.asi8, side="right")
    mins = np.full(len(dates), np.nan)
    maxs = np.full(len(dates), np.nan)
    for k, (i0, i1) in enumerate(zip(ia, ib)):
        if i1 > i0 and i1 <= len(lows):
            mins[k] = lows[i0:i1].min()
            maxs[k] = highs[i0:i1].max()
    out = pd.DataFrame({
        "date": dates.date, "dow": dates.dayofweek, "entry": pa, "exit": pb,
        "ret": np.log(pb / pa), "min_rel": mins / pa - 1.0, "max_rel": maxs / pa - 1.0,
        "entry_ts": a_utc,
    })
    out["trading_day"] = [is_trading_day(d) for d in out["date"]]
    return out.dropna(subset=["ret"])


def summarize(tab: pd.DataFrame, liq_long: float, liq_short: float, per_year: float = 52.0) -> pd.DataFrame:
    rows = []
    for dow, g in tab.groupby("dow"):
        r = g["ret"].to_numpy()
        n = len(r)
        mean, std = r.mean(), r.std(ddof=1) if n > 1 else np.nan
        rows.append({
            "dow": DOW_NAMES[dow], "n": n, "mean_pct": 100 * mean, "median_pct": 100 * np.median(r), "std_pct": 100 * std,
            "hit_rate": float((r > 0).mean()), "t_stat": mean / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan,
            "sharpe_long": mean / std * np.sqrt(per_year) if std and std > 0 else np.nan,
            "liq_if_long_40x": float((g["min_rel"] <= -liq_long).mean()),
            "liq_if_short_40x": float((g["max_rel"] >= liq_short).mean()),
            "worst_pct": 100 * r.min(), "best_pct": 100 * r.max(),
        })
    r = tab["ret"].to_numpy()
    n = len(r)
    std = r.std(ddof=1)
    rows.append({
        "dow": "ALL", "n": n, "mean_pct": 100 * r.mean(), "median_pct": 100 * np.median(r), "std_pct": 100 * std,
        "hit_rate": float((r > 0).mean()), "t_stat": r.mean() / (std / np.sqrt(n)),
        "sharpe_long": r.mean() / std * np.sqrt(per_year * 7), "liq_if_long_40x": float((tab["min_rel"] <= -liq_long).mean()),
        "liq_if_short_40x": float((tab["max_rel"] >= liq_short).mean()), "worst_pct": 100 * r.min(), "best_pct": 100 * r.max(),
    })
    return pd.DataFrame(rows)


def flow_conditional(tab: pd.DataFrame, flows: pd.DataFrame | None, publish_lag_hours: float, tz: str) -> pd.DataFrame | None:
    """Session return conditional on the sign of the latest *known* ETF flow print at entry."""
    if flows is None or len(flows) == 0:
        return None
    fk = flows_known_at(flows, publish_lag_hours, tz).rename(columns={"known_at": "ts"}).sort_values("ts")
    left = tab[["entry_ts", "ret", "dow"]].rename(columns={"entry_ts": "ts"}).sort_values("ts")
    m = pd.merge_asof(left, fk[["ts", "total", "date"]], on="ts", direction="backward", tolerance=pd.Timedelta(days=5))
    m["flow_bucket"] = np.select([m["total"] > 0, m["total"] < 0], ["inflow", "outflow"], default="none/stale")
    g = m.groupby("flow_bucket")["ret"]
    out = pd.DataFrame({"n": g.size(), "mean_pct": 100 * g.mean(), "hit_rate": g.apply(lambda s: float((s > 0).mean())),
                        "t_stat": g.apply(lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan)})
    return out.reset_index()


def run_session_analysis(cfg: dict) -> dict[str, pd.DataFrame]:
    from .model.walkforward import load_dataset  # reuse loaders (no training)

    d, tcfg, fcfg = cfg["data"], cfg["trading"], cfg["features"]
    _, _, bars, _ = load_dataset(cfg)
    report_dir = resolve(cfg, cfg["walkforward"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    flows = load_flows_csv(resolve(cfg, d["etf_flows_csv"]))
    start = pd.Timestamp(cfg["walkforward"]["train_start"], tz="UTC")
    bars = bars[bars.index >= start - pd.Timedelta(days=2)]
    liq_long = liquidation_distance(float(tcfg["leverage"]), 40.0, +1)
    liq_short = liquidation_distance(float(tcfg["leverage"]), 40.0, -1)

    all_rows, cond_rows, lines = [], [], []
    lines.append(f"# BTC session analysis by weekday (ET), {bars.index[0].date()} -> {bars.index[-1].date()}\n")
    lines.append(f"Returns are log returns in %, spot BTC-USD ({d['price_source']}). `liq_if_long_40x` = share of sessions whose "
                 f"path fell >= {100*liq_long:.2f}% below entry (an isolated 40x long with no stop would have been liquidated); "
                 f"`liq_if_short_40x` uses the {100*liq_short:.2f}% rise. `sharpe_long` = annualised Sharpe of always being long that "
                 f"session on that weekday (52 obs/yr).\n")
    for sess in SESSIONS:
        tab = session_table(bars, int(d["bar_minutes"]), fcfg["tz"], sess)
        summ = summarize(tab, liq_long, liq_short)
        summ.insert(0, "session", sess.name)
        all_rows.append(summ)
        lines.append(f"\n## {sess.name}  ({sess.start} -> {sess.end} ET{', next day' if sess.end_day_offset else ''}; {sess.note})\n")
        lines.append(summ.round(3).to_string(index=False))
        lines.append("")
        cond = flow_conditional(tab, flows, float(d["etf_flow_publish_lag_hours"]), fcfg["tz"])
        if cond is not None:
            cond.insert(0, "session", sess.name)
            cond_rows.append(cond)
            lines.append("conditional on the latest known ETF flow print at entry:")
            lines.append(cond.round(3).to_string(index=False))
            lines.append("")
    by_dow = pd.concat(all_rows, ignore_index=True)
    by_dow.to_csv(report_dir / "sessions_by_dow.csv", index=False)
    if cond_rows:
        pd.concat(cond_rows, ignore_index=True).to_csv(report_dir / "sessions_by_flow.csv", index=False)
    (report_dir / "sessions_summary.md").write_text("\n".join(lines))
    _heatmap(by_dow, report_dir / "sessions_heatmap.png")
    log.info("session analysis written to %s", report_dir)
    print("\n".join(lines))
    return {"by_dow": by_dow}


def _heatmap(by_dow: pd.DataFrame, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    piv = by_dow[by_dow["dow"] != "ALL"].pivot(index="session", columns="dow", values="mean_pct")[DOW_NAMES]
    hit = by_dow[by_dow["dow"] != "ALL"].pivot(index="session", columns="dow", values="hit_rate")[DOW_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, title, fmt, cmap, center in (
        (axes[0], piv, "mean log return (%)", "{:.2f}", "RdYlGn", 0.0),
        (axes[1], hit, "hit rate (share of sessions up)", "{:.2f}", "RdYlGn", 0.5),
    ):
        vals = data.to_numpy(dtype=float)
        span = np.nanmax(np.abs(vals - center)) or 1.0
        im = ax.imshow(vals, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
        ax.set_xticks(range(len(data.columns)), data.columns)
        ax.set_yticks(range(len(data.index)), data.index)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("BTC by weekday and NYSE session (ET)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
