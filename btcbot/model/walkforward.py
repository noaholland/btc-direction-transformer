"""Walk-forward (expanding window) training and out-of-sample prediction.

For each test quarter: train on everything before it (minus an embargo of `horizon_bars` so no
training label peeks into the test period), hold out the most recent `val_fraction` of the train
window for early stopping + temperature calibration, then score the test quarter. The union of
test-quarter predictions is the honest out-of-sample record the backtest uses.

A logistic regression on the last bar's features is fitted on the same folds as a sanity
baseline: if the transformer cannot beat it, the sequence model is not adding anything.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ..config import resolve
from ..data.etf_flows import load_flows_csv
from ..data.funding import load_or_fetch_funding
from ..data.prices import fill_gaps, load_or_fetch_prices, resample_bars
from ..features import build_features
from .dataset import FeatureScaler, windows_view
from .train import fit_temperature, get_device, predict_ensemble, train_ensemble

log = logging.getLogger(__name__)


def load_dataset(cfg: dict, update: bool = False) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """Returns (features, feature_cols, bars30, funding)."""
    d = cfg["data"]
    data_dir = resolve(cfg, d["data_dir"])
    start = datetime.fromisoformat(d["start"]).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    px15 = load_or_fetch_prices(data_dir / f"btc_15m_{d['price_source']}.csv", start, end, d["price_source"], 900, update=update)
    bars = fill_gaps(resample_bars(px15, int(d["bar_minutes"])), int(d["bar_minutes"]))
    funding = load_or_fetch_funding(data_dir / "funding_btc.csv", cfg["trading"]["coin"], start, end, update=update)
    flows = load_flows_csv(resolve(cfg, d["etf_flows_csv"]))
    if flows is None:
        log.warning("no ETF flow CSV found — flow features are zeros")
    feats, cols = build_features(bars, funding, flows, cfg)
    return feats, cols, bars, funding


def make_folds(decision_ts: pd.Series, first_test_start: str, test_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(first_test_start, tz="UTC")
    end_all = decision_ts.max()
    folds = []
    while start < end_all:
        stop = start + pd.DateOffset(months=test_months)
        folds.append((start, min(stop, end_all + pd.Timedelta(seconds=1))))
        start = stop
    return folds


def _metrics(p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(float)
    conf = np.maximum(p, 1 - p)
    sel = conf >= thr
    out = {
        "n": int(len(y)),
        "acc": float((pred == y).mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "coverage@thr": float(sel.mean()),
        "acc@thr": float((pred[sel] == y[sel]).mean()) if sel.any() else float("nan"),
        "n@thr": int(sel.sum()),
    }
    return out


def run_walkforward(cfg: dict, skip_final: bool = False) -> pd.DataFrame:
    wf, mcfg, fcfg = cfg["walkforward"], cfg["model"], cfg["features"]
    seq_len, H = int(fcfg["seq_len"]), int(fcfg["horizon_bars"])
    thr = float(cfg["trading"]["confidence_threshold"])
    report_dir = resolve(cfg, wf["report_dir"])
    model_dir = resolve(cfg, wf["model_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(mcfg.get("device", "auto"))
    torch.set_num_threads(max(1, torch.get_num_threads()))

    feats, cols, _, _ = load_dataset(cfg)
    feats = feats.reset_index()  # positional rows; 'time' column = bar start
    ts = feats["decision_ts"]
    train_start = pd.Timestamp(wf["train_start"], tz="UTC")
    labeled = feats["y"].notna().to_numpy()
    y_all = np.nan_to_num(feats["y"].to_numpy(dtype=np.float32), nan=0.0)
    log.info("dataset: %d rows, %d features, %s -> %s, device=%s", len(feats), len(cols), ts.iloc[0], ts.iloc[-1], device)

    folds = make_folds(ts, wf["first_test_start"], int(wf["test_months"]))
    preds, rows = [], []
    for k, (t0, t1) in enumerate(folds):
        tr_mask = (ts >= train_start) & (ts < t0 - pd.Timedelta(minutes=int(cfg["data"]["bar_minutes"]) * H)) & labeled
        te_mask = (ts >= t0) & (ts < t1) & labeled
        tr_rows = np.flatnonzero(tr_mask.to_numpy())
        te_rows = np.flatnonzero(te_mask.to_numpy())
        tr_rows = tr_rows[tr_rows >= seq_len - 1]
        te_rows = te_rows[te_rows >= seq_len - 1]
        if len(tr_rows) < 2000 or len(te_rows) == 0:
            log.warning("fold %d skipped (train=%d test=%d)", k, len(tr_rows), len(te_rows))
            continue
        n_val = int(len(tr_rows) * float(wf["val_fraction"]))
        fit_rows, val_rows = tr_rows[: -n_val - H], tr_rows[-n_val:]  # embargo H bars between fit and val

        scaler = FeatureScaler(cols).fit(feats.iloc[fit_rows])
        X = scaler.transform(feats)
        win = windows_view(X, seq_len)
        models, T, info = train_ensemble(win, y_all, fit_rows, val_rows, len(cols), seq_len, mcfg, device)
        logits = predict_ensemble(models, win, te_rows, seq_len, device)
        p = 1.0 / (1.0 + np.exp(-logits / T))
        m = _metrics(p, y_all[te_rows], thr)

        # baseline: logistic regression on the last bar only
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(X[fit_rows], y_all[fit_rows])
        lr_val_logit = lr.decision_function(X[val_rows])
        lr_T = max(fit_temperature(lr_val_logit, y_all[val_rows]), float(mcfg.get("min_temperature", 0.0)))
        lr_p = 1.0 / (1.0 + np.exp(-lr.decision_function(X[te_rows]) / lr_T))
        mb = _metrics(lr_p, y_all[te_rows], thr)

        rows.append({"fold": k, "test_start": t0.date(), "test_end": t1.date(), "n_train": len(fit_rows), "epochs": str(info["epochs"]),
                     "temperature": round(T, 3), "val_acc": round(info["val_acc"], 4), "seconds": round(info["seconds"]),
                     **{f"tf_{a}": b for a, b in m.items()},
                     "lr_acc": mb["acc"], "lr_auc": mb["auc"], "lr_acc@thr": mb["acc@thr"], "lr_coverage@thr": mb["coverage@thr"]})
        log.info("fold %d %s..%s  n_train=%d  TF acc=%.3f auc=%.3f acc@%.2f=%.3f cov=%.2f | LR acc=%.3f auc=%.3f  (%.0fs, T=%.2f, val_acc=%.3f)",
                 k, t0.date(), t1.date(), len(fit_rows), m["acc"], m["auc"], thr, m["acc@thr"], m["coverage@thr"], mb["acc"], mb["auc"],
                 info["seconds"], T, info["val_acc"])
        preds.append(pd.DataFrame({"decision_ts": ts.iloc[te_rows].to_numpy(), "p_up": p, "logit": logits, "temperature": T,
                                   "p_up_lr": lr_p, "y": y_all[te_rows], "fwd_ret": feats["fwd_ret"].iloc[te_rows].to_numpy(),
                                   "close": feats["close"].iloc[te_rows].to_numpy(), "fold": k}))

    oos = pd.concat(preds).sort_values("decision_ts").reset_index(drop=True)
    oos.to_csv(report_dir / "oos_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(report_dir / "walkforward_metrics.csv", index=False)
    overall = _metrics(oos["p_up"].to_numpy(), oos["y"].to_numpy(), thr)
    overall_lr = _metrics(oos["p_up_lr"].to_numpy(), oos["y"].to_numpy(), thr)
    log.info("OOS overall  TF: %s", {k: round(v, 4) for k, v in overall.items()})
    log.info("OOS overall  LR: %s", {k: round(v, 4) for k, v in overall_lr.items()})
    with open(report_dir / "walkforward_summary.json", "w") as fh:
        json.dump({"transformer": overall, "logistic_baseline": overall_lr, "folds": len(rows),
                   "oos_start": str(oos["decision_ts"].min()), "oos_end": str(oos["decision_ts"].max())}, fh, indent=2, default=str)

    if not skip_final:
        train_final(cfg, feats, cols, y_all, labeled, device)
    return oos


def train_final(cfg: dict, feats: pd.DataFrame, cols: list[str], y_all: np.ndarray, labeled: np.ndarray, device) -> Path:
    """Fit the deployable model on all labeled data (val = most recent slice) and save the bundle."""
    wf, mcfg, fcfg = cfg["walkforward"], cfg["model"], cfg["features"]
    seq_len, H = int(fcfg["seq_len"]), int(fcfg["horizon_bars"])
    ts = feats["decision_ts"]
    rows = np.flatnonzero(((ts >= pd.Timestamp(wf["train_start"], tz="UTC")) & labeled).to_numpy())
    rows = rows[rows >= seq_len - 1]
    n_val = int(len(rows) * float(wf["val_fraction"]))
    fit_rows, val_rows = rows[: -n_val - H], rows[-n_val:]
    scaler = FeatureScaler(cols).fit(feats.iloc[fit_rows])
    win = windows_view(scaler.transform(feats), seq_len)
    models, T, info = train_ensemble(win, y_all, fit_rows, val_rows, len(cols), seq_len, mcfg, device)
    out = resolve(cfg, wf["model_dir"]) / "final"
    out.mkdir(parents=True, exist_ok=True)
    for k, model in enumerate(models):
        torch.save(model.state_dict(), out / f"model_{k}.pt")
    bundle = {
        "feature_cols": cols, "scaler": scaler.to_dict(), "temperature": T, "seq_len": seq_len, "horizon_bars": H,
        "bar_minutes": int(cfg["data"]["bar_minutes"]), "n_models": len(models),
        "model": {**{k: mcfg[k] for k in ("d_model", "n_heads", "n_layers", "dropout")}, "arch": mcfg.get("arch", "transformer")},
        "trained_at": datetime.now(timezone.utc).isoformat(), "data_end": str(ts.iloc[-1]),
        "n_train": int(len(fit_rows)), "val_acc": info.get("val_acc"), "best_val_loss": info.get("best_val_loss"),
    }
    with open(out / "bundle.json", "w") as fh:
        json.dump(bundle, fh, indent=2)
    log.info("final model saved to %s (val_acc=%.3f, T=%.2f, %d train rows)", out, info.get("val_acc", float("nan")), T, len(fit_rows))
    return out
