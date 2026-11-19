"""Walk-forward benchmark of sequence architectures on order-flow bars, for two targets:
direction of the mid over the next H bars (classification) and traded volume over the next H bars (regression).

    .venv/bin/python -m orderflow.train data/ticks/hl_BTC*.csv --bar 2s --archs all --seeds 3
    .venv/bin/python -m orderflow.train data/ticks/ibit_*.csv --bar 1s --fee-bps 0 --archs transformer,mamba

Each test block is predicted only by models trained on earlier blocks (H-bar embargo). Every architecture sees the
same folds, features and seeds, next to linear baselines (logistic for direction, ridge for volume) and a
"volume persists" rule. Differences versus the linear baseline get moving-block bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from btcbot.model.archs import ARCHS
from btcbot.model.dataset import FeatureScaler, windows_view
from btcbot.model.train import get_device, predict_logits, train_model

from .features import FEATURES, add_features_and_labels, book_events, lee_ready_accuracy, load_ticks, make_bars, valid_window_ends

MODEL = {"d_model": 32, "n_heads": 4, "n_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256,
         "max_epochs": 30, "patience": 5, "label_smoothing": 0.0, "seed": 7, "min_temperature": 1.0}


def build(paths: list[str], bar: str, horizon: int, data_tz: str, tz: str) -> tuple[pd.DataFrame, float | None]:
    frames, acc = [], []
    for k, p in enumerate(sorted(paths)):          # one file per day/security is fine; sessions never span files
        ev = book_events(load_ticks(p, tz=data_tz))
        ev["session"] = ev["session"] + 10_000 * k
        a = lee_ready_accuracy(ev)
        if a is not None:
            acc.append(a)
        frames.append(add_features_and_labels(make_bars(ev, bar), horizon=horizon, tz=tz))
    frames.sort(key=lambda x: x.index[0])
    for prev, nxt in zip(frames, frames[1:]):
        if nxt.index[0] <= prev.index[-1]:   # walk-forward by row position needs one chronological series
            raise SystemExit("tick files overlap in time (two securities or two recorders?): run one series at a time")
    return pd.concat(frames), (float(np.mean(acc)) if acc else None)


def auc(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) > 1 else np.nan


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return r2_score(y[m], p[m]) if m.sum() > 2 else np.nan


def block_bootstrap(metric, y, a, b, block: int, n: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """metric(y, a) - metric(y, b) with a 90% circular moving-block bootstrap interval (keeps autocorrelation)."""
    rng = np.random.default_rng(seed)
    N = len(y)
    nb = int(np.ceil(N / block))
    diffs = []
    for _ in range(n):
        idx = ((rng.integers(0, N, nb)[:, None] + np.arange(block)) % N).ravel()[:N]
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    lo, hi = np.nanpercentile(diffs, [5, 95])
    return float(metric(y, a) - metric(y, b)), float(lo), float(hi)


def cost_test(p, fwd_bps, spread_bps, fee_bps: float, thr: float) -> dict:
    """Trade every bar with confidence >= thr: cross the spread in and out (one full spread) plus fees both ways."""
    conf = np.maximum(p, 1 - p)
    m = (conf >= thr) & np.isfinite(fwd_bps) & np.isfinite(p)
    gross = np.where(p >= 0.5, 1.0, -1.0)[m] * fwd_bps[m]
    net = gross - spread_bps[m] - 2 * fee_bps
    return {"threshold": thr, "trades": int(m.sum()), "gross_bps": float(gross.mean()) if m.any() else np.nan,
            "net_bps": float(net.mean()) if m.any() else np.nan, "hit_rate": float((gross > 0).mean()) if m.any() else np.nan}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticks", nargs="+")
    ap.add_argument("--bar", default="2s")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--folds", type=int, default=4, help="contiguous time blocks; block 0 is train-only")
    ap.add_argument("--archs", default="transformer", help=f"comma list from {','.join(ARCHS)}, or 'all'")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=MODEL["max_epochs"])
    ap.add_argument("--fee-bps", type=float, default=4.5, help="per side (Hyperliquid taker 4.5; US ETF commissions ~0)")
    ap.add_argument("--data-tz", default="UTC", help="timezone of naive timestamps (Bloomberg Excel exports are local time)")
    ap.add_argument("--tz", default="America/New_York", help="clock for the time-of-day feature")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="reports/orderflow")
    a = ap.parse_args(argv)
    archs = list(ARCHS) if a.archs == "all" else a.archs.split(",")

    f, lr_acc = build(a.ticks, a.bar, a.horizon, a.data_tz, a.tz)
    sess = f["session"].to_numpy()
    ends = valid_window_ends(sess, a.seq_len)
    print(f"{len(f)} bars of {a.bar}, {len(np.unique(sess))} sessions, {len(ends)} usable windows, horizon {a.horizon} bars; "
          f"Lee-Ready vs true aggressor: {'n/a' if lr_acc is None else f'{lr_acc:.1%}'}", flush=True)
    y_dir, y_vol = f["y_dir"].to_numpy(), f["y_vol"].to_numpy()
    blocks = np.array_split(ends, a.folds)
    device = get_device(a.device)
    cfg = dict(MODEL, max_epochs=a.epochs)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    # every finished (fold, arch, seed) is appended to partial.csv, so an interrupted run resumes where it stopped
    ck, ck_key = outd / "partial.csv", outd / "partial.json"
    key = json.dumps({k: v for k, v in vars(a).items() if k not in ("archs", "seeds", "device", "out")}, sort_keys=True)
    done = pd.read_csv(ck, parse_dates=["time"]) if ck.exists() and ck_key.exists() and ck_key.read_text() == key else None
    if done is None:
        ck.unlink(missing_ok=True)
        ck_key.write_text(key)
    have = set() if done is None else set(zip(done["fold"], done["arch"], done["seed"]))
    out_rows = []
    for k in range(1, a.folds):
        test = blocks[k]
        train = np.concatenate(blocks[:k])
        train = train[train < test[0] - a.horizon]                 # embargo: no training label may reach into the test block
        n_val = max(1, int(0.15 * len(train)))
        fit, val = train[: -n_val - a.horizon], train[-n_val:]
        if len(fit) < 500:
            print(f"fold {k}: only {len(fit)} training windows, skipped")
            continue
        # time of day only means something once training covers several days; on a few hours it is a disguised
        # clock index whose test values fall outside the training range
        feats = FEATURES if f.index[fit[-1]] - f.index[fit[0]] >= pd.Timedelta("3D") else [c for c in FEATURES if not c.startswith("tod")]
        X = FeatureScaler(feats).fit(f.iloc[fit]).transform(f)
        win = windows_view(X, a.seq_len)
        lab = lambda r: r[np.isfinite(y_dir[r])]  # noqa: E731 — bars whose move cleared the dead zone
        ok = lambda r: r[np.isfinite(y_vol[r])]  # noqa: E731
        yd = np.nan_to_num(y_dir)
        mu, sd = y_vol[ok(fit)].mean(), y_vol[ok(fit)].std() + 1e-9
        yv = np.nan_to_num((y_vol - mu) / sd)
        shared = {"time": f.index[test], "fold": k, "y_dir": y_dir[test], "y_vol": y_vol[test],
                  "fwd_ret_bps": f["fwd_ret_bps"].to_numpy()[test], "spread_bps": f["spread_bps"].to_numpy()[test]}
        logit = LogisticRegression(C=0.1, max_iter=2000).fit(X[lab(fit)], yd[lab(fit)]).predict_proba(X[test])[:, 1]
        ridge = Ridge(alpha=1.0).fit(X[ok(fit)], y_vol[ok(fit)]).predict(X[test])
        out_rows.append(pd.DataFrame(shared | {"arch": "baseline", "seed": 0, "p_up": logit, "vol_pred": ridge}))
        out_rows.append(pd.DataFrame(shared | {"arch": "persistence", "seed": 0, "p_up": np.nan, "vol_pred": f["past_vol"].to_numpy()[test]}))
        for arch in archs:
            for s in range(a.seeds):
                if (k, arch, s) in have:
                    continue
                t0 = time.time()
                c = dict(cfg, arch=arch, seed=cfg["seed"] + 101 * s)
                m_dir, T, _ = train_model(win, yd, lab(fit), lab(val), len(feats), a.seq_len, c, device)
                p = 1 / (1 + np.exp(-predict_logits(m_dir, win, test, a.seq_len, device) / T))
                m_vol, _, _ = train_model(win, yv, ok(fit), ok(val), len(feats), a.seq_len, dict(c, regression=True), device)
                v = predict_logits(m_vol, win, test, a.seq_len, device) * sd + mu
                row = pd.DataFrame(shared | {"arch": arch, "seed": s, "p_up": p, "vol_pred": v, "sec": time.time() - t0,
                                             "params": sum(q.numel() for q in m_dir.parameters())})
                row.to_csv(ck, mode="a", header=not ck.exists(), index=False)
                out_rows.append(row)
                print(f"fold {k} {arch:12s} seed {s}: direction AUC {auc(y_dir[test], p):.3f} (logistic {auc(y_dir[test], logit):.3f}) | "
                      f"volume R2 {r2(y_vol[test], v):.3f} (ridge {r2(y_vol[test], ridge):.3f}) | {time.time() - t0:.0f}s", flush=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
    if not out_rows:
        raise SystemExit("no fold had enough data; record more ticks or use a smaller --bar")

    P = pd.concat(out_rows + ([done] if done is not None else []), ignore_index=True)
    P.to_csv(outd / "oos_predictions.csv", index=False)
    runs = P.drop_duplicates(["arch", "fold", "seed"]).groupby("arch")
    params, timing = runs["params"].first().to_dict(), runs["sec"].sum().to_dict()
    ens = P.groupby(["arch", "time", "fold"], sort=False).agg(
        p_up=("p_up", "mean"), vol_pred=("vol_pred", "mean"), y_dir=("y_dir", "first"), y_vol=("y_vol", "first"),
        fwd_ret_bps=("fwd_ret_bps", "first"), spread_bps=("spread_bps", "first")).reset_index()
    base = ens[ens["arch"] == "baseline"].set_index("time")
    refs = {"baseline": base}
    if "softmax" in archs:   # the attention control: same scaffold and readout as every new architecture
        refs["softmax"] = ens[ens["arch"] == "softmax"].set_index("time").loc[base.index]
    block = 10 * a.horizon
    rows = []
    for arch, g in ens.groupby("arch", sort=False):
        g = g.set_index("time").loc[base.index]
        per_seed = P[P["arch"] == arch].groupby("seed")
        sa = per_seed.apply(lambda d: auc(d["y_dir"].to_numpy(), d["p_up"].to_numpy()))
        sr = per_seed.apply(lambda d: r2(d["y_vol"].to_numpy(), d["vol_pred"].to_numpy()))
        row = {"arch": arch, "params": params.get(arch), "train_sec": round(timing.get(arch, 0.0)),
               "dir_auc": auc(g["y_dir"].to_numpy(), g["p_up"].to_numpy()), "dir_auc_seed_sd": float(sa.std()) if len(sa) > 1 else np.nan,
               "vol_r2": r2(g["y_vol"].to_numpy(), g["vol_pred"].to_numpy()), "vol_r2_seed_sd": float(sr.std()) if len(sr) > 1 else np.nan}
        if arch not in ("baseline", "persistence"):
            yd_, yv_ = g["y_dir"].to_numpy(), g["y_vol"].to_numpy()
            for name, ref in refs.items():
                if name != arch:
                    row[f"dir_auc_vs_{name}"] = block_bootstrap(auc, yd_, g["p_up"].to_numpy(), ref["p_up"].to_numpy(), block)
                    row[f"vol_r2_vs_{name}"] = block_bootstrap(r2, yv_, g["vol_pred"].to_numpy(), ref["vol_pred"].to_numpy(), block)
        if arch != "persistence":
            for thr in (0.55, 0.6):
                c = cost_test(g["p_up"].to_numpy(), g["fwd_ret_bps"].to_numpy(), g["spread_bps"].to_numpy(), a.fee_bps, thr)
                row[f"trades@{thr}"], row[f"net_bps@{thr}"] = c["trades"], c["net_bps"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(outd / "arch_comparison.csv", index=False)
    meta = {"bars": len(f), "features": feats, "bar": a.bar, "horizon_bars": a.horizon, "seq_len": a.seq_len, "folds": a.folds, "seeds": a.seeds,
            "test_bars": int(len(base)), "dir_labeled_share": float(np.isfinite(base["y_dir"]).mean()),
            "mean_spread_bps": float(base["spread_bps"].mean()), "fee_bps_per_side": a.fee_bps, "lee_ready_accuracy": lr_acc,
            "files": a.ticks, "first_test_bar": str(base.index.min()), "last_test_bar": str(base.index.max())}
    (outd / "run.json").write_text(json.dumps(meta, indent=2))
    fmt = lambda t: "" if not isinstance(t, tuple) else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
    show = table.copy()
    for col in [c for c in show if "_vs_" in c]:
        show[col] = show[col].map(fmt)
    print("\n" + json.dumps(meta))
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
