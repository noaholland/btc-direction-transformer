import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from btcbot.model.archs import ARCHS, MoE, LinearAttention, Mamba, RWKVTimeMix, make_model
from btcbot.model.dataset import windows_view
from btcbot.model.train import predict_logits, train_model


@pytest.mark.parametrize("arch", ARCHS)
def test_every_arch_maps_a_window_to_one_number(arch):
    m = make_model(arch, 5, 12, d_model=16, n_heads=2, n_layers=2, dropout=0.0)
    y = m(torch.randn(3, 12, 5))
    assert y.shape == (3,) and torch.isfinite(y).all()


@pytest.mark.parametrize("mixer", [lambda: Mamba(8), lambda: RWKVTimeMix(8, 0, 2)])
def test_recurrent_mixers_are_causal(mixer):
    torch.manual_seed(0)
    m = mixer().eval()
    x = torch.randn(2, 10, 8)
    x2 = x.clone()
    x2[:, 6:] += torch.randn(2, 4, 8)            # change only the future
    y, y2 = m(x), m(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:])


def test_linear_attention_matches_explicit_formula():
    torch.manual_seed(0)
    m = LinearAttention(8, 2)
    x = torch.randn(2, 5, 8)
    q, k, v = m.qkv(x).view(2, 5, 3, 2, 4).unbind(2)
    q, k = F.elu(q) + 1, F.elu(k) + 1
    w = torch.einsum("bihd,bjhd->bhij", q, k)              # phi(q_i).phi(k_j), an L x L matrix
    w = w / w.sum(-1, keepdim=True)
    ref = m.out(torch.einsum("bhij,bjhd->bihd", w, v).reshape(2, 5, 8))
    assert torch.allclose(m(x), ref, atol=1e-5)


def test_rwkv_recurrence_matches_closed_form():
    torch.manual_seed(0)
    m = RWKVTimeMix(4, 0, 1)
    x = torch.randn(1, 6, 4)
    xx = F.pad(x, (0, 0, 1, -1))
    k = m.key(x * m.mix_k + xx * (1 - m.mix_k))[0].double()
    v = m.value(x * m.mix_v + xx * (1 - m.mix_v))[0].double()
    r = torch.sigmoid(m.receptance(x * m.mix_r + xx * (1 - m.mix_r)))[0].double()
    w, u = torch.exp(m.decay).double(), m.first.double()
    rows = []
    for t in range(6):
        num = torch.exp(u + k[t]) * v[t]
        den = torch.exp(u + k[t])
        for i in range(t):
            e = torch.exp(-(t - 1 - i) * w + k[i])
            num, den = num + e * v[i], den + e
        rows.append(num / den)
    ref = m.output((r * torch.stack(rows)).float())
    assert torch.allclose(m(x)[0], ref, atol=1e-5)


def test_moe_gates_and_balance_loss():
    torch.manual_seed(0)
    m = MoE(8, dropout=0.0, n_experts=4, top_k=2, alpha=0.01).eval()
    torch.nn.init.zeros_(m.router.weight)          # uniform router: every expert prob 1/4, ties go to experts 0 and 1
    x = torch.randn(2, 3, 8)
    y = m(x)
    assert torch.allclose(y, 0.5 * m.experts[0](x) + 0.5 * m.experts[1](x), atol=1e-6)
    # f = (1, 0, 0, 0) (all first choices on expert 0), P = 1/4 each -> alpha * 4 * (1 * 1/4) = alpha
    assert math.isclose(float(m.aux_loss), 0.01, rel_tol=1e-6)


def test_mamba_dt_bias_inverts_softplus():
    m = Mamba(16)
    dt = F.softplus(m.dt_proj.bias)
    assert dt.min() >= 1e-3 - 1e-6 and dt.max() <= 0.1 + 1e-6
    assert torch.allclose(-torch.exp(m.A_log[0]), -torch.arange(1, 17, dtype=torch.float32))


@pytest.mark.parametrize("arch", ARCHS)
def test_every_arch_learns_a_simple_sequence_rule_through_train_model(arch):
    """Label = was the sum of feature 0 over the first half of the window positive? Needs memory across the window."""
    rng = np.random.default_rng(0)
    n, L = 1600, 12
    X = rng.standard_normal((n + L, 3)).astype(np.float32)
    s = np.convolve(X[:, 0], np.ones(L // 2), "full")[: len(X)]          # sum of the last L/2 values ending at row i
    y = np.zeros(len(X), dtype=np.float32)
    y[L - 1:] = (s[L // 2 - 1: len(X) - L // 2] > 0)                       # rows i-L+1 .. i-L/2 = first half of the window
    win = windows_view(X, L)
    rows = np.arange(L - 1, len(X))
    fit, val, test = rows[:1200], rows[1200:1400], rows[1400:]
    cfg = {"arch": arch, "d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 0.0,
           "batch_size": 64, "max_epochs": 25, "patience": 25, "label_smoothing": 0.0, "seed": 0}
    m, T, _ = train_model(win, y, fit, val, 3, L, cfg, torch.device("cpu"))
    acc = ((predict_logits(m, win, test, L, torch.device("cpu")) > 0) == (y[test] > 0.5)).mean()
    assert acc > 0.8, f"{arch} only reached {acc:.2f}"
