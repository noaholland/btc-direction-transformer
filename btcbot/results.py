"""Render reports/RESULTS.md from the report CSV/JSON files. Run: python -m btcbot.results"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_config, resolve


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(cfg: dict, report_dir: Path, title: str) -> str:
    wf = json.load(open(report_dir / "walkforward_summary.json"))
    bt = json.load(open(report_dir / "backtest_summary.json"))
    folds = pd.read_csv(report_dir / "walkforward_metrics.csv")
    sens = pd.read_csv(report_dir / "threshold_sensitivity.csv")
    calib = pd.read_csv(report_dir / "calibration.csv")
    skill = pd.read_csv(report_dir / "decision_time_skill.csv")
    tf, lr = wf["transformer"], wf["logistic_baseline"]
    thr = cfg["trading"]["confidence_threshold"]
    H = cfg["features"]["horizon_bars"] * cfg["data"]["bar_minutes"] / 60
    out = [f"## {title}", "",
           f"_Generated {date.today()} from `{report_dir.name}/`. Horizon {H:.0f}h, walk-forward OOS {wf['oos_start'][:10]} to {wf['oos_end'][:10]}, {wf['folds']} quarterly folds._", "",
           "### Walk-forward skill (every prediction made by a model that never saw that quarter)", "",
           "| model | bars | accuracy | AUC | log-loss (coin flip = 0.6931) | Brier (coin flip = 0.25) | coverage at conf >= %.2f | accuracy on covered |" % thr,
           "|---|---|---|---|---|---|---|---|",
           f"| transformer ensemble | {tf['n']} | {tf['acc']:.3f} | {tf['auc']:.3f} | {tf['logloss']:.4f} | {tf['brier']:.4f} | {100*tf['coverage@thr']:.1f}% | {_fmt(tf['acc@thr'])} |",
           f"| logistic baseline (last bar) | {lr['n']} | {lr['acc']:.3f} | {lr['auc']:.3f} | {lr['logloss']:.4f} | {lr['brier']:.4f} | {100*lr['coverage@thr']:.1f}% | {_fmt(lr['acc@thr'])} |", "",
           "Per fold (transformer acc / AUC, then logistic acc / AUC):", ""]
    out.append("| fold | test quarter | train rows | TF acc | TF AUC | LR acc | LR AUC | temperature |")
    out.append("|---|---|---|---|---|---|---|---|")
    for _, r in folds.iterrows():
        out.append(f"| {int(r['fold'])} | {r['test_start']} to {r['test_end']} | {int(r['n_train'])} | {r['tf_acc']:.3f} | {r['tf_auc']:.3f} | {r['lr_acc']:.3f} | {r['lr_auc']:.3f} | {r['temperature']:.2f} |")
    out += ["", "### Calibration: does the confidence mean anything?", "",
            "| confidence bin | bars | share of bars | realised accuracy | mean signed forward return (bps) |", "|---|---|---|---|---|"]
    for _, r in calib.iterrows():
        out.append(f"| {r['conf_bin']} | {int(r['n'])} | {100*r['share']:.1f}% | {r['realised_acc']:.3f} | {r['mean_fwd_ret_bps_signed']:+.1f} |")
    out += ["", "### Skill at the bot's decision times only", "",
            "| decision time (ET) | bars | AUC | median conf | 90th pct conf | max conf | n at >=0.55 | acc at >=0.55 | n at >=0.60 | acc at >=0.60 | n at >=%.2f |" % thr,
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in skill.iterrows():
        out.append(f"| {r['decision_time_et']} | {int(r['n'])} | {r['auc']:.3f} | {r['conf_p50']:.3f} | {r['conf_p90']:.3f} | {r['conf_max']:.3f} | "
                   f"{int(r['n@0.55'])} | {_fmt(r['acc@0.55'])} | {int(r['n@0.6'])} | {_fmt(r['acc@0.6'])} | {int(r[f'n@{thr}'])} |")
    m = bt["risk_mode"]
    out += ["", "### Backtest with the configured rules (Kelly risk mode, 40x, %s stop / %s take-profit, $%.0f start)" % (
        cfg["trading"]["stop_loss_pct"], cfg["trading"]["take_profit_pct"], cfg["backtest"]["initial_equity"]), "",
        f"* trades: **{m['n_trades']}**, final equity **${m['final_equity']:.2f}** ({100*m['total_return']:+.1f}%)" + (
            f", win rate {100*m['win_rate']:.1f}%, max drawdown {100*m['max_drawdown']:.1f}%, exits {m['exit_reasons']}" if m["n_trades"] else ""),
        f"* literal margin mode: {bt['margin_mode']['n_trades']} trades, final ${bt['margin_mode']['final_equity']:.2f}",
        f"* buy & hold over the same window: {100*bt['buy_and_hold_return']:+.1f}%",
        f"* random-direction baseline (same timing/confidence, 200 seeds): median final ${bt['random_direction_baseline']['median_final']:.2f}; "
        f"{100*bt['random_direction_baseline']['share_beating_model']:.0f}% of random runs did at least as well as the model",
        "", "### Threshold sensitivity (what happens if you lower the confidence bar)", "",
        "| threshold | trades | win rate | final equity | max drawdown | daily Sharpe |", "|---|---|---|---|---|---|"]
    for _, r in sens.iterrows():
        out.append(f"| {r['threshold']:.2f} | {int(r['n_trades'])} | {_fmt(r['win_rate'])} | ${r['final_equity']:.2f} | {_fmt(r['max_drawdown'])} | {_fmt(r['sharpe_daily'], 2)} |")
    b = (cfg["trading"]["take_profit_pct"] - 0.0011) / (cfg["trading"]["stop_loss_pct"] + 0.0011)
    out += ["", f"Break-even win rate for this stop/take-profit structure is 1/(1+b) with b = {b:.2f}, i.e. **{100/(1+b):.1f}%**. "
            "Any threshold whose win rate sits below that loses money regardless of leverage; leverage only sets how fast.", ""]
    return "\n".join(out)


def main() -> None:
    cfg = load_config()
    main_dir = resolve(cfg, cfg["walkforward"]["report_dir"])
    parts = [render(cfg, main_dir, "Main run: 8h horizon")]
    h24 = main_dir / "h24"
    if (h24 / "backtest_summary.json").exists():
        cfg24 = load_config(Path(cfg["_root"]) / "config_h24.yaml")
        parts.append(render(cfg24, h24, "Sensitivity run: 24h horizon"))
    text = "# Results\n\n" + "\n\n".join(parts) + "\n"
    (main_dir / "RESULTS.md").write_text(text)
    readme = Path(cfg["_root"]) / "README.md"
    if readme.exists():
        src = readme.read_text()
        start = src.index("## 6. What the numbers say")
        end = src.index("## 7. Running it live")
        lines = text.replace("# Results\n\n", "").splitlines()
        body = "\n".join(("#" + l) if l.startswith("## ") or l.startswith("### ") else l for l in lines)  # demote one level
        readme.write_text(src[:start] + "## 6. What the numbers say\n\n" + body + "\n" + src[end:])
    print(text)


if __name__ == "__main__":
    main()
