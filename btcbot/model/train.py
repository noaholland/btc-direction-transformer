"""Train one model on a time-ordered train/val split, with early stopping and temperature calibration."""
from __future__ import annotations

import copy
import logging
import math
import random
import time

import numpy as np
import torch
from torch import nn

from .dataset import window_index
from .archs import make_model

log = logging.getLogger(__name__)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _smooth(y: torch.Tensor, eps: float) -> torch.Tensor:
    return y * (1.0 - eps) + 0.5 * eps


@torch.no_grad()
def predict_logits(model: nn.Module, win: torch.Tensor, end_rows: np.ndarray, seq_len: int,
                   device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    widx = torch.as_tensor(window_index(end_rows, seq_len), dtype=torch.long)
    out = np.empty(len(widx), dtype=np.float32)
    for s in range(0, len(widx), batch_size):
        xb = win[widx[s : s + batch_size]].to(device)
        out[s : s + batch_size] = model(xb).float().cpu().numpy()
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: T minimising BCE(sigmoid(logit/T), y). Grid + refinement, no scipy needed."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    bce = nn.BCEWithLogitsLoss()

    def loss(T: float) -> float:
        return float(bce(z / T, t))

    grid = np.logspace(-1.0, 1.0, 81)
    best = min(grid, key=loss)
    lo, hi = best / 1.15, best * 1.15
    for _ in range(3):
        fine = np.linspace(lo, hi, 41)
        best = min(fine, key=loss)
        lo, hi = best / 1.03, best * 1.03
    return float(best)


def train_model(win: torch.Tensor, y_all: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray,
                n_features: int, seq_len: int, mcfg: dict, device: torch.device | None = None) -> tuple[nn.Module, float, dict]:
    device = device or get_device(mcfg.get("device", "auto"))
    set_seed(int(mcfg["seed"]))
    model = make_model(mcfg.get("arch", "transformer"), n_features, seq_len, d_model=int(mcfg["d_model"]),
                       n_heads=int(mcfg["n_heads"]), n_layers=int(mcfg["n_layers"]), dropout=float(mcfg["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"]))
    bs = int(mcfg["batch_size"])
    max_epochs, patience = int(mcfg["max_epochs"]), int(mcfg["patience"])
    steps_per_epoch = max(1, math.ceil(len(train_rows) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(mcfg["lr"]), total_steps=max_epochs * steps_per_epoch, pct_start=0.15)
    regression = bool(mcfg.get("regression", False))  # MSE on a real-valued target instead of BCE on 0/1
    eps = 0.0 if regression else float(mcfg.get("label_smoothing", 0.0))
    loss_fn = nn.MSELoss() if regression else nn.BCEWithLogitsLoss()

    y_t = torch.as_tensor(y_all, dtype=torch.float32)
    train_rows_t = torch.as_tensor(np.asarray(train_rows), dtype=torch.long)
    val_rows_t = torch.as_tensor(np.asarray(val_rows), dtype=torch.long)
    tr_w = torch.as_tensor(window_index(train_rows, seq_len), dtype=torch.long)
    va_w = torch.as_tensor(window_index(val_rows, seq_len), dtype=torch.long)
    y_val = y_t[val_rows_t].to(device)

    best_state, best_val, best_acc, bad = None, float("inf"), float("nan"), 0
    history = []
    rng = np.random.default_rng(int(mcfg["seed"]))
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(len(tr_w)), dtype=torch.long)
        tot, n = 0.0, 0
        for s in range(0, len(perm), bs):
            sel = perm[s : s + bs]
            xb = win[tr_w[sel]].to(device)
            yb = y_t[train_rows_t[sel]].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, _smooth(yb, eps)) + getattr(model, "aux_loss", 0.0)  # MoE load balancing
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(sel)
            n += len(sel)
        model.eval()
        with torch.no_grad():
            vl = []
            for s in range(0, len(va_w), 2048):
                xb = win[va_w[s : s + 2048]].to(device)
                vl.append(model(xb))
            v_logits = torch.cat(vl)
            val_loss = float(loss_fn(v_logits, y_val))
            val_acc = float(((v_logits > 0).float() == y_val).float().mean())
        history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val_loss, "val_acc": val_acc})
        log.debug("epoch %d train %.4f val %.4f acc %.3f", epoch, tot / max(n, 1), val_loss, val_acc)
        if val_loss < best_val - 1e-5:
            best_val, best_acc, bad = val_loss, val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    v_logits = predict_logits(model, win, val_rows, seq_len, device)
    T = 1.0 if regression else max(fit_temperature(v_logits, y_all[val_rows]), float(mcfg.get("min_temperature", 0.0)))
    info = {"epochs": len(history), "best_val_loss": best_val, "temperature": T, "seconds": time.time() - t0, "val_acc": best_acc}
    return model, T, info


def train_ensemble(win: torch.Tensor, y_all: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray, n_features: int,
                   seq_len: int, mcfg: dict, device: torch.device | None = None) -> tuple[list[nn.Module], float, dict]:
    """Train `n_ensemble` models with different seeds; the ensemble logit is their mean.

    Temperature is fitted on the ensemble's validation logits and floored at `min_temperature`, so
    calibration can only make the model less sure of itself, never more.
    """
    device = device or get_device(mcfg.get("device", "auto"))
    n = int(mcfg.get("n_ensemble", 1))
    models, v_logits, infos = [], [], []
    for k in range(n):
        m_cfg = dict(mcfg, seed=int(mcfg["seed"]) + 1000 * k)
        model, _, info = train_model(win, y_all, train_rows, val_rows, n_features, seq_len, m_cfg, device)
        models.append(model)
        v_logits.append(predict_logits(model, win, val_rows, seq_len, device))
        infos.append(info)
    v = np.mean(v_logits, axis=0)
    T = max(fit_temperature(v, y_all[val_rows]), float(mcfg.get("min_temperature", 0.0)))
    val_acc = float(((v > 0) == (y_all[val_rows] > 0.5)).mean())
    info = {"epochs": [i["epochs"] for i in infos], "best_val_loss": float(np.mean([i["best_val_loss"] for i in infos])),
            "temperature": T, "seconds": sum(i["seconds"] for i in infos), "val_acc": val_acc, "n_models": n}
    return models, T, info


def predict_ensemble(models: list[nn.Module], win: torch.Tensor, rows: np.ndarray, seq_len: int, device: torch.device) -> np.ndarray:
    return np.mean([predict_logits(m, win, rows, seq_len, device) for m in models], axis=0)
