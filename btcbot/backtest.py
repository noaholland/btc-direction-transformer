"""Event-driven backtest of the out-of-sample predictions with Kelly sizing at 40x.

Fill model (deliberately conservative):
  * entry at the decision bar's close +/- slippage, taker fee both ways
  * on each later bar: liquidation is checked first, then the stop, then the take-profit
    (if both stop and TP are inside one bar the stop is assumed to have hit first)
  * stop and horizon exits are market orders (slippage applied); TP is a trigger too (taker)
  * liquidation loses the whole isolated margin
  * hourly Hyperliquid funding is charged for every hour boundary the position is open
  * one position at a time
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_et import DOW_NAMES
from .config import resolve
from .kelly import plan_trade

log = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: int
    p_conf: float
    risk_frac: float
    notional: float
    margin: float
    entry_px: float
    exit_px: float
    fees: float
    funding: float
    pnl: float
    equity_after: float
    exit_reason: str
    bars_held: int
    decision_time_et: str
    dow: str


def _decision_allowed(decision_ts_et: pd.DatetimeIndex, decision_times: list[str]) -> np.ndarray:
    labels = pd.Series(decision_ts_et.strftime("%H:%M"))
    return labels.isin(set(decision_times)).to_numpy()


def simulate(preds: pd.DataFrame, bars: pd.DataFrame, funding: pd.DataFrame, cfg: dict,
             sizing_mode: str | None = None, threshold: float | None = None, initial_equity: float | None = None,
             decision_times: list[str] | None = None, direction_override: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.Series]:
    tcfg = dict(cfg["trading"])
    if sizing_mode:
        tcfg["sizing_mode"] = sizing_mode
    if threshold is not None:
        tcfg["confidence_threshold"] = threshold
    decision_times = decision_times or list(tcfg["decision_times_et"])
    bar_min = int(cfg["data"]["bar_minutes"])
    H = int(cfg["features"]["horizon_bars"])
    equity = float(initial_equity if initial_equity is not None else cfg["backtest"]["initial_equity"])
    tz = cfg["features"]["tz"]
    fee = float(tcfg["taker_fee"])
    slip = float(tcfg["slippage_bps"]) / 1e4

    bars = bars.sort_index()
    end_ts = bars.index + pd.Timedelta(minutes=bar_min)
    close, high, low = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    pos_of = pd.Series(np.arange(len(bars)), index=end_ts)
    p_series = pd.Series(np.nan, index=end_ts)
    pr = preds.set_index("decision_ts")["p_up"]
    pr = pr[pr.index.isin(p_series.index)]
    p_series.loc[pr.index] = pr.to_numpy()
    p_arr = p_series.to_numpy()
    if direction_override is not None:  # for the random baseline: keep |confidence|, flip direction
        conf = np.maximum(p_arr, 1 - p_arr)
        p_arr = np.where(direction_override > 0, conf, 1 - conf)
    allowed = _decision_allowed(end_ts.tz_convert(tz), decision_times)
    fund = funding["funding_rate"].copy()
    fund.index = fund.index.floor("h")
    fund_map = fund.to_dict()
    et_labels = end_ts.tz_convert(tz)

    trades: list[Trade] = []
    eq_curve = np.full(len(bars), np.nan)
    i = 0
    n = len(bars)
    while i < n:
        eq_curve[i] = equity
        if not (allowed[i] and np.isfinite(p_arr[i])) or i + 1 >= n:
            i += 1
            continue
        plan, reason = plan_trade(equity, float(p_arr[i]), float(close[i]), tcfg, H)
        if plan is None:
            i += 1
            continue
        side = plan.side
        entry_px = close[i] * (1 + slip * side)
        notional = plan.size * entry_px
        margin = notional / float(tcfg["leverage"])
        fees = notional * fee
        stop_px, tp_px, liq_px = plan.stop_px, plan.tp_px, plan.liq_px
        exit_bar = min(i + H, n - 1)
        exit_px, reason, k_exit = None, "horizon", exit_bar
        for k in range(i + 1, exit_bar + 1):
            if side > 0:
                if low[k] <= liq_px:
                    exit_px, reason, k_exit = liq_px, "liquidated", k
                    break
                if low[k] <= stop_px:
                    exit_px, reason, k_exit = stop_px * (1 - slip), "stop", k
                    break
                if tp_px is not None and high[k] >= tp_px:
                    exit_px, reason, k_exit = tp_px, "take_profit", k
                    break
            else:
                if high[k] >= liq_px:
                    exit_px, reason, k_exit = liq_px, "liquidated", k
                    break
                if high[k] >= stop_px:
                    exit_px, reason, k_exit = stop_px * (1 + slip), "stop", k
                    break
                if tp_px is not None and low[k] <= tp_px:
                    exit_px, reason, k_exit = tp_px, "take_profit", k
                    break
        if exit_px is None:
            exit_px = close[exit_bar] * (1 - slip * side)
        # funding on every hour boundary in (entry, exit]
        hours = pd.date_range(end_ts[i].ceil("h"), end_ts[k_exit], freq="h", inclusive="left")
        if end_ts[i] == end_ts[i].ceil("h"):
            hours = hours[1:]
        funding_paid = sum(fund_map.get(hh, 0.0) for hh in hours) * notional * side
        if reason == "liquidated":
            pnl = -margin
        else:
            fees += plan.size * exit_px * fee
            pnl = side * (exit_px - entry_px) * plan.size - fees - funding_paid
            pnl = max(pnl, -margin)  # isolated margin caps the loss
        equity += pnl
        trades.append(Trade(end_ts[i], end_ts[k_exit], side, plan.p_conf, plan.risk_frac, notional, margin, entry_px, exit_px,
                            fees, funding_paid, pnl, equity, reason, k_exit - i, et_labels[i].strftime("%H:%M"), DOW_NAMES[et_labels[i].dayofweek]))
        eq_curve[i + 1 : k_exit + 1] = equity  # realised-only curve; fine at this granularity
        i = k_exit  # a new decision may be taken at the exit bar itself
        if equity <= 1.0:
            break
    eq = pd.Series(eq_curve, index=end_ts).ffill().fillna(equity)
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    return tdf, eq


def metrics(trades: pd.DataFrame, eq: pd.Series, initial_equity: float) -> dict:
    if len(trades) == 0:
        return {"n_trades": 0, "final_equity": initial_equity, "total_return": 0.0}
    daily = eq.resample("1D").last().dropna()
    dr = daily.pct_change().dropna()
    dd = (eq / eq.cummax() - 1.0)
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    win = trades["pnl"] > 0
    streak, worst = 0, 0
    for w in win:
        streak = 0 if w else streak + 1
        worst = max(worst, streak)
    gross_win, gross_loss = trades.loc[win, "pnl"].sum(), -trades.loc[~win, "pnl"].sum()
    return {
        "n_trades": int(len(trades)),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial_equity - 1.0),
        "cagr": float((eq.iloc[-1] / initial_equity) ** (365.0 / days) - 1.0) if eq.iloc[-1] > 0 else -1.0,
        "sharpe_daily": float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "win_rate": float(win.mean()),
        "avg_win_pct": float(100 * (trades.loc[win, "pnl"] / (trades.loc[win, "equity_after"] - trades.loc[win, "pnl"])).mean()) if win.any() else float("nan"),
        "avg_loss_pct": float(100 * (trades.loc[~win, "pnl"] / (trades.loc[~win, "equity_after"] - trades.loc[~win, "pnl"])).mean()) if (~win).any() else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_usd": float(trades["pnl"].mean()),
        "worst_losing_streak": int(worst),
        "total_fees": float(trades["fees"].sum()),
        "total_funding": float(trades["funding"].sum()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        "avg_notional": float(trades["notional"].mean()),
        "avg_margin_frac": float((trades["margin"] / (trades["equity_after"] - trades["pnl"])).mean()),
        "days": int(days),
    }


def calibration_table(preds: pd.DataFrame, bins=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01)) -> pd.DataFrame:
    p = preds["p_up"].to_numpy()
    y = preds["y"].to_numpy()
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5) == (y > 0.5)).astype(float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum() == 0:
            continue
        rows.append({"conf_bin": f"[{lo:.2f},{min(hi,1.0):.2f})", "n": int(sel.sum()), "share": float(sel.mean()),
                     "mean_conf": float(conf[sel].mean()), "realised_acc": float(correct[sel].mean()),
                     "mean_fwd_ret_bps_signed": float(1e4 * (np.sign(p[sel] - 0.5) * preds["fwd_ret"].to_numpy()[sel]).mean())})
    return pd.DataFrame(rows)


def decision_time_skill(preds: pd.DataFrame, tz: str, decision_times: list[str], thresholds=(0.5, 0.55, 0.6, 0.65)) -> pd.DataFrame:
    """Model skill restricted to the bars the bot actually trades on, at several confidence cut-offs."""
    from sklearn.metrics import roc_auc_score

    et = preds["decision_ts"].dt.tz_convert(tz)
    label = et.dt.strftime("%H:%M")
    rows = []
    for dt in list(decision_times) + ["ALL"]:
        sel = preds if dt == "ALL" else preds[label == dt]
        if len(sel) == 0:
            continue
        p, y, r = sel["p_up"].to_numpy(), sel["y"].to_numpy(), sel["fwd_ret"].to_numpy()
        conf = np.maximum(p, 1 - p)
        row = {"decision_time_et": dt, "n": len(sel), "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
               "conf_p50": float(np.median(conf)), "conf_p90": float(np.percentile(conf, 90)), "conf_max": float(conf.max())}
        for t in thresholds:
            m = conf >= t
            row[f"n@{t}"] = int(m.sum())
            row[f"acc@{t}"] = float(((p[m] >= 0.5) == (y[m] > 0.5)).mean()) if m.any() else np.nan
            row[f"ret_bps@{t}"] = float(1e4 * (np.sign(p[m] - 0.5) * r[m]).mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo(trades: pd.DataFrame, initial_equity: float, n_paths: int, seed: int = 0) -> dict:
    """Bootstrap the per-trade return-on-equity sequence to get a distribution of outcomes."""
    if len(trades) < 5:
        return {}
    r = (trades["pnl"] / (trades["equity_after"] - trades["pnl"])).to_numpy()
    rng = np.random.default_rng(seed)
    finals, maxdds = np.empty(n_paths), np.empty(n_paths)
    for k in range(n_paths):
        path = initial_equity * np.cumprod(1 + rng.choice(r, size=len(r), replace=True))
        finals[k] = path[-1]
        maxdds[k] = (path / np.maximum.accumulate(np.concatenate([[initial_equity], path]))[1:] - 1).min()
    return {"paths": n_paths, "median_final": float(np.median(finals)), "p05_final": float(np.percentile(finals, 5)),
            "p95_final": float(np.percentile(finals, 95)), "prob_loss": float((finals < initial_equity).mean()),
            "prob_dd_gt_50pct": float((maxdds < -0.5).mean()), "median_max_dd": float(np.median(maxdds))}


def run_backtest_report(cfg: dict) -> dict:
    from .model.walkforward import load_dataset

    report_dir = resolve(cfg, cfg["walkforward"]["report_dir"])
    preds = pd.read_csv(report_dir / "oos_predictions.csv", parse_dates=["decision_ts"])
    preds["decision_ts"] = pd.to_datetime(preds["decision_ts"], utc=True)
    _, _, bars, funding = load_dataset(cfg)
    init = float(cfg["backtest"]["initial_equity"])
    thr = float(cfg["trading"]["confidence_threshold"])
    lo, hi = preds["decision_ts"].min(), preds["decision_ts"].max()
    bars_oos = bars[(bars.index >= lo - pd.Timedelta(days=1)) & (bars.index <= hi + pd.Timedelta(days=2))]

    trades, eq = simulate(preds, bars_oos, funding, cfg)
    main = metrics(trades, eq, init)
    trades.to_csv(report_dir / "backtest_trades.csv", index=False)
    eq.to_csv(report_dir / "backtest_equity.csv", header=["equity"])

    # literal-margin variant and buy & hold for comparison
    trades_m, eq_m = simulate(preds, bars_oos, funding, cfg, sizing_mode="margin")
    margin_mode = metrics(trades_m, eq_m, init)
    bh_ret = float(bars_oos["close"].iloc[-1] / bars_oos.loc[bars_oos.index >= lo, "close"].iloc[0] - 1.0)

    # random-direction baseline with identical trade timing/confidence
    rng = np.random.default_rng(1)
    rand_final = []
    for _ in range(int(cfg["backtest"]["random_baseline_seeds"])):
        direction = rng.choice([-1, 1], size=len(bars_oos))
        t_r, e_r = simulate(preds, bars_oos, funding, cfg, direction_override=direction)
        rand_final.append(float(e_r.iloc[-1]))
    rand = {"median_final": float(np.median(rand_final)), "p90_final": float(np.percentile(rand_final, 90)),
            "share_beating_model": float((np.array(rand_final) >= main["final_equity"]).mean())}

    # sensitivity to the confidence threshold
    sens = []
    for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        tt, ee = simulate(preds, bars_oos, funding, cfg, threshold=t)
        m = metrics(tt, ee, init)
        sens.append({"threshold": t, "n_trades": m["n_trades"], "win_rate": m.get("win_rate"), "final_equity": m["final_equity"],
                     "max_drawdown": m.get("max_drawdown"), "sharpe_daily": m.get("sharpe_daily")})
    sens = pd.DataFrame(sens)
    sens.to_csv(report_dir / "threshold_sensitivity.csv", index=False)

    # by weekday / decision time
    by = None
    if len(trades):
        by = trades.groupby(["decision_time_et", "dow"]).agg(n=("pnl", "size"), win_rate=("pnl", lambda s: float((s > 0).mean())),
                                                              pnl_usd=("pnl", "sum"), avg_pnl=("pnl", "mean")).reset_index()
        by.to_csv(report_dir / "backtest_by_dow.csv", index=False)
    calib = calibration_table(preds)
    calib.to_csv(report_dir / "calibration.csv", index=False)
    skill = decision_time_skill(preds, cfg["features"]["tz"], list(cfg["trading"]["decision_times_et"]))
    skill.to_csv(report_dir / "decision_time_skill.csv", index=False)
    mc = monte_carlo(trades, init, int(cfg["backtest"]["monte_carlo_paths"]))

    summary = {"period": [str(lo), str(hi)], "config": {k: cfg["trading"][k] for k in ("leverage", "sizing_mode", "kelly_multiplier",
               "max_risk_per_trade", "stop_loss_pct", "take_profit_pct", "confidence_threshold", "decision_times_et")},
               "risk_mode": main, "margin_mode": margin_mode, "buy_and_hold_return": bh_ret, "random_direction_baseline": rand,
               "monte_carlo": mc}
    with open(report_dir / "backtest_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _plot(eq, eq_m, bars_oos, init, report_dir / "equity_curve.png", thr)

    print(f"\n=== Backtest {lo.date()} -> {hi.date()}  (OOS predictions only, ${init:.0f} start) ===")
    print(f"risk mode   : {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in main.items()}, default=str)}")
    print(f"margin mode : final ${margin_mode['final_equity']:.2f}, maxDD {margin_mode.get('max_drawdown', 0):.1%}, trades {margin_mode['n_trades']}, exits {margin_mode.get('exit_reasons')}")
    print(f"buy & hold  : {bh_ret:+.1%}")
    print(f"random dirs : median final ${rand['median_final']:.2f}, p90 ${rand['p90_final']:.2f}, share of random runs >= model: {rand['share_beating_model']:.1%}")
    print(f"monte carlo : {mc}")
    print("\nthreshold sensitivity:\n" + sens.round(3).to_string(index=False))
    print("\ncalibration (OOS, all bars):\n" + calib.round(3).to_string(index=False))
    print("\nskill at the bot's decision times (OOS):\n" + skill.round(3).to_string(index=False))
    if by is not None:
        print("\nby decision time x weekday:\n" + by.round(2).to_string(index=False))
    return summary


def _plot(eq: pd.Series, eq_m: pd.Series, bars: pd.DataFrame, init: float, path, thr: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, label="model, risk-mode Kelly (40x)", lw=1.4)
    ax.plot(eq_m.index, eq_m.values, label="model, literal 2.5% margin @40x", lw=1.0, alpha=0.8)
    bh = init * bars["close"] / bars["close"].iloc[0]
    ax.plot(bh.index, bh.values, label="buy & hold BTC (1x)", lw=1.0, alpha=0.7)
    ax.axhline(init, color="grey", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_title(f"Out-of-sample equity, confidence threshold {thr}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
