from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "data": {
        "price_source": "coinbase",
        "start": "2023-12-01",
        "data_dir": "data",
        "bar_minutes": 30,
        "etf_flows_csv": "data/etf_flows.csv",
        "farside_url": "https://farside.co.uk/btc/",
        "etf_flow_publish_lag_hours": 8,
    },
    "features": {"seq_len": 48, "horizon_bars": 16, "tz": "America/New_York"},
    "model": {
        "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.2, "lr": 3e-4,
        "weight_decay": 1e-2, "batch_size": 256, "max_epochs": 40, "patience": 6,
        "label_smoothing": 0.05, "seed": 42, "n_ensemble": 3, "min_temperature": 1.0, "device": "auto",
    },
    "walkforward": {
        "train_start": "2024-01-11", "first_test_start": "2025-01-01", "test_months": 3,
        "val_fraction": 0.15, "model_dir": "models", "report_dir": "reports",
    },
    "trading": {
        "coin": "BTC", "leverage": 40, "sizing_mode": "risk", "kelly_multiplier": 0.25,
        "max_risk_per_trade": 0.025, "stop_loss_pct": 0.008, "take_profit_pct": 0.016,
        "confidence_threshold": 0.65, "decision_times_et": ["09:00", "10:00"],
        "max_margin_fraction": 0.5, "stop_liq_buffer": 0.8, "taker_fee": 0.00045,
        "maker_fee": 0.00015, "slippage_bps": 1.0, "min_notional_usd": 10.0,
    },
    "backtest": {"initial_equity": 1000.0, "random_baseline_seeds": 200, "monte_carlo_paths": 2000},
    "live": {
        "mode": "dry_run", "poll_seconds": 20, "state_file": "logs/bot_state.json", "log_dir": "logs",
        "daily_loss_limit": 0.10, "max_consecutive_losses": 4, "max_flow_age_days": 5,
        "etf_refresh_hour_et": 20, "pause_file": "logs/PAUSE",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (relative paths resolve against the project root) merged over DEFAULTS."""
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    user = {}
    if p.exists():
        with open(p) as fh:
            user = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULTS, user)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg.get("_root", PROJECT_ROOT)) / p
