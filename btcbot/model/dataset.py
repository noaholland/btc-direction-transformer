"""Scaling and windowing utilities shared by training, backtesting and the live bot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..features import ONE_HOT_COLS

CLIP_Z = 5.0


class FeatureScaler:
    """Per-column z-scoring for continuous features; one-hot columns pass through. Fit on train rows only."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = list(feature_cols)
        self.cont = [c for c in feature_cols if c not in ONE_HOT_COLS]
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        for c in self.cont:
            col = df[c].to_numpy(dtype=np.float64)
            self.mean[c] = float(np.nanmean(col))
            self.std[c] = float(max(np.nanstd(col), 1e-8))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=np.float32).copy()
        for j, c in enumerate(self.feature_cols):
            if c in self.mean:
                X[:, j] = np.clip((X[:, j] - self.mean[c]) / self.std[c], -CLIP_Z, CLIP_Z)
        return np.nan_to_num(X, nan=0.0, posinf=CLIP_Z, neginf=-CLIP_Z)

    def to_dict(self) -> dict:
        return {"feature_cols": self.feature_cols, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        s = cls(d["feature_cols"])
        s.mean, s.std = dict(d["mean"]), dict(d["std"])
        return s


def windows_view(X: np.ndarray, seq_len: int) -> torch.Tensor:
    """(N, F) -> zero-copy view (N - seq_len + 1, seq_len, F); window k ends at row k + seq_len - 1."""
    t = torch.from_numpy(np.ascontiguousarray(X))
    return t.unfold(0, seq_len, 1).permute(0, 2, 1)


def window_index(end_rows: np.ndarray, seq_len: int) -> np.ndarray:
    """Map row indices (window ends) to indices into `windows_view`; caller must ensure end_rows >= seq_len - 1."""
    return np.asarray(end_rows) - (seq_len - 1)
