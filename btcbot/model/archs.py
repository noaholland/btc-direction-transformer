"""Sequence models beyond softmax attention. Every model has the DirectionTransformer contract:
forward(x: (batch, seq_len, n_features)) -> (batch,) one logit (classification) or value (regression).

    softmax      standard softmax attention + MLP blocks, read out at the newest bar: the attention control
    transformer  softmax attention read out from a CLS token (btcbot.model.transformer.DirectionTransformer, kept so
                 saved BTC models load; its CLS readout struggles to single out the newest bar, compare with softmax)
    linear       linear attention, Katharopoulos et al. 2020, phi(x) = elu(x) + 1, O(L d^2) instead of O(L^2 d)
    moe          softmax attention whose feed-forward is a top-2 mixture of experts (Switch-style balance loss)
    mamba        selective state-space blocks, Gu & Dao 2023, pure PyTorch scan (no CUDA kernels, runs on MPS)
    rwkv         RWKV-4 time-mix + channel-mix, Peng et al. 2023, stable WKV recurrence
    hybrid       Jamba-style stack: Mamba+MLP, attention+MLP, Mamba+MoE, no positional embedding
    lstm         plain LSTM, the pre-transformer reference

Models with mixture-of-experts layers set `self.aux_loss` in forward; train_model adds it to the loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import DirectionTransformer

ARCHS = ("softmax", "transformer", "linear", "moe", "mamba", "rwkv", "hybrid", "lstm")


def _shift(x: torch.Tensor) -> torch.Tensor:
    """Previous time step (zeros at t=0): (B, L, D) -> (B, L, D)."""
    return F.pad(x, (0, 0, 1, -1))


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Bidirectional softmax self-attention."""
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


class LinearAttention(nn.Module):
    """softmax(QK^T)V replaced by phi(Q)(phi(K)^T V) / (phi(Q) sum_j phi(K_j)): the L x L matrix is never formed."""
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, D // self.h).unbind(2)       # each (B, L, H, Dh)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = torch.einsum("blhd,blhe->bhde", k, v)                              # sum_j phi(k_j) v_j^T
        z = 1.0 / (torch.einsum("blhd,bhd->blh", q, k.sum(1)) + 1e-6)          # 1 / phi(q_i) . sum_j phi(k_j)
        return self.out(torch.einsum("blhd,bhde,blh->blhe", q, kv, z).reshape(B, L, D))


class MoE(nn.Module):
    """Top-k routed experts. Gates are a softmax over the k selected router logits (Mixtral); the balance loss is
    Switch Transformer's alpha * E * sum_i f_i * P_i, f_i = share of tokens whose first choice is expert i,
    P_i = mean router probability of expert i."""
    def __init__(self, d: int, dropout: float, n_experts: int = 4, top_k: int = 2, alpha: float = 0.01):
        super().__init__()
        self.k, self.alpha = top_k, alpha
        self.router = nn.Linear(d, n_experts, bias=False)
        # each expert is half the dense MLP's width, so top-2 activates about as many parameters as one dense MLP
        self.experts = nn.ModuleList(MLP(d, 2 * d, dropout) for _ in range(n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        logits = self.router(x)                                                  # (B, L, E)
        top_v, top_i = logits.topk(self.k, dim=-1)
        # ponytail: every expert runs on every token, fine at this size; a real MoE dispatches tokens so compute ~ k/E
        every = torch.stack([e(x) for e in self.experts], dim=-2)                 # (B, L, E, D)
        chosen = every.gather(-2, top_i.unsqueeze(-1).expand(*top_i.shape, x.shape[-1]))
        y = (top_v.softmax(-1).unsqueeze(-1) * chosen).sum(-2)
        E = logits.shape[-1]
        f = F.one_hot(top_i[..., 0], E).float().mean((0, 1))
        P = logits.softmax(-1).mean((0, 1))
        self.aux_loss = self.alpha * E * (f * P).sum()
        return y


class Mamba(nn.Module):
    """Selective SSM block (Gu & Dao 2023, mamba_simple.py), written out step by step.

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t,   y_t = C_t . h_t + D * x_t,   out = W_out (y * silu(z))
    dt, B, C depend on the input (that is the 'selective' part); A is a fixed-per-channel negative decay.
    """
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        di, self.n, self.r = expand * d, d_state, math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.r + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.r, di)
        nn.init.uniform_(self.dt_proj.weight, -self.r ** -0.5, self.r ** -0.5)
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))           # inverse softplus: softplus(bias) = dt
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))  # S4D-real, A_n = -n
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, x):
        L = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2))     # causal depthwise conv
        dt, Bm, Cm = self.x_proj(xi).split([self.r, self.n, self.n], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                         # (B, L, di)
        dA = torch.exp(dt.unsqueeze(-1) * -torch.exp(self.A_log))                 # (B, L, di, N)
        dBx = (dt * xi).unsqueeze(-1) * Bm.unsqueeze(2)                           # (B, L, di, N)
        h = xi.new_zeros(dA.shape[0], dA.shape[2], self.n)
        hs = []
        # unbind once: indexing dA[:, t] in the loop makes autograd write a full-size gradient per step (25x slower on MPS)
        # ponytail: sequential scan is fine for ~60 steps; a chunked parallel scan pays off past a few hundred
        for a_t, b_t in zip(dA.unbind(1), dBx.unbind(1)):
            h = a_t * h + b_t
            hs.append(h)
        y = torch.einsum("bldn,bln->bld", torch.stack(hs, dim=1), Cm) + xi * self.D
        return self.out_proj(y * F.silu(z))


def _rwkv_ratios(layer: int, n_layers: int) -> tuple[float, float]:
    return (layer / (n_layers - 1) if n_layers > 1 else 0.0), 1.0 - layer / n_layers


class RWKVTimeMix(nn.Module):
    """RWKV-4 time mixing: an attention-like weighted average of past values with per-channel exponential decay.

    wkv_t = (sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t) / (same with v -> 1),  out = W_o (sigmoid(r_t) * wkv_t)
    """
    def __init__(self, d: int, layer: int = 0, n_layers: int = 1):
        super().__init__()
        r01, r10 = _rwkv_ratios(layer, n_layers)
        h = torch.arange(d, dtype=torch.float32) / max(d - 1, 1)
        self.decay = nn.Parameter(-5 + 8 * h ** (0.7 + 1.3 * r01))               # w = exp(decay) > 0
        zigzag = torch.tensor([((i + 1) % 3 - 1) * 0.5 for i in range(d)])
        self.first = nn.Parameter(torch.full((d,), math.log(0.3)) + zigzag)      # bonus u for the current token
        x = torch.arange(d, dtype=torch.float32) / d
        self.mix_k = nn.Parameter(x ** r10)
        self.mix_v = nn.Parameter(x ** r10 + 0.3 * r01)
        self.mix_r = nn.Parameter(x ** (0.5 * r10))
        self.key, self.value, self.receptance, self.output = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        xx = _shift(x)
        k = self.key(x * self.mix_k + xx * (1 - self.mix_k))
        v = self.value(x * self.mix_v + xx * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r)))
        w, u = -torch.exp(self.decay), self.first
        aa = torch.zeros_like(k[:, 0])
        bb = torch.zeros_like(k[:, 0])
        pp = torch.full_like(k[:, 0], -1e38)                                      # running max exponent, keeps exp() finite
        out = []
        for kt, vt in zip(k.unbind(1), v.unbind(1)):
            ww = u + kt
            p = torch.maximum(pp, ww)
            e1, e2 = torch.exp(pp - p), torch.exp(ww - p)
            out.append((e1 * aa + e2 * vt) / (e1 * bb + e2))
            ww = pp + w
            p = torch.maximum(ww, kt)
            e1, e2 = torch.exp(ww - p), torch.exp(kt - p)
            aa, bb, pp = e1 * aa + e2 * vt, e1 * bb + e2, p
        return self.output(r * torch.stack(out, dim=1))


class RWKVChannelMix(nn.Module):
    """RWKV feed-forward: squared-ReLU MLP on a token-shifted input, gated by a sigmoid receptance."""
    def __init__(self, d: int, layer: int = 0, n_layers: int = 1, hidden: int = 4):
        super().__init__()
        _, r10 = _rwkv_ratios(layer, n_layers)
        x = torch.arange(d, dtype=torch.float32) / d
        self.mix_k, self.mix_r = nn.Parameter(x ** r10), nn.Parameter(x ** r10)
        self.key = nn.Linear(d, hidden * d, bias=False)
        self.value = nn.Linear(hidden * d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)

    def forward(self, x):
        xx = _shift(x)
        k = torch.square(torch.relu(self.key(x * self.mix_k + xx * (1 - self.mix_k))))
        return torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r))) * self.value(k)


class Block(nn.Module):
    """Pre-norm residual block: x + mixer(LN(x)), then x + ffn(LN(x)) when there is a feed-forward."""
    def __init__(self, d: int, mixer: nn.Module, ffn: nn.Module | None, dropout: float):
        super().__init__()
        self.n1, self.mixer, self.ffn, self.drop = nn.LayerNorm(d), mixer, ffn, nn.Dropout(dropout)
        self.n2 = nn.LayerNorm(d) if ffn is not None else None

    def forward(self, x):
        x = x + self.drop(self.mixer(self.n1(x)))
        return x if self.ffn is None else x + self.drop(self.ffn(self.n2(x)))


class SeqModel(nn.Module):
    """Input projection -> blocks -> LayerNorm -> MLP head on the LAST time step (the newest bar)."""
    def __init__(self, n_features: int, seq_len: int, d: int, blocks: list[nn.Module], positional: bool, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d) * 0.02) if positional else None
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        h = self.input_proj(x)
        if self.pos_emb is not None:
            h = h + self.pos_emb[:, : x.shape[1]]
        for b in self.blocks:
            h = b(h)
        self.aux_loss = sum((m.aux_loss for m in self.modules() if isinstance(m, MoE)), torch.zeros((), device=x.device))
        return self.head(self.norm(h[:, -1])).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, d: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, d, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def make_model(arch: str, n_features: int, seq_len: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
               dropout: float = 0.2) -> nn.Module:
    d, n = int(d_model), int(n_layers)
    mlp = lambda: MLP(d, 4 * d, dropout)  # noqa: E731
    if arch == "transformer":
        return DirectionTransformer(n_features, seq_len, d, n_heads, n, dropout)
    if arch == "lstm":
        return LSTMModel(n_features, d, n, dropout)
    if arch == "softmax":
        blocks, pos = [Block(d, Attention(d, n_heads, dropout), mlp(), dropout) for _ in range(n)], True
    elif arch == "linear":
        blocks, pos = [Block(d, LinearAttention(d, n_heads), mlp(), dropout) for _ in range(n)], True
    elif arch == "moe":
        blocks, pos = [Block(d, Attention(d, n_heads, dropout), MoE(d, dropout), dropout) for _ in range(n)], True
    elif arch == "mamba":  # two Mamba blocks per transformer layer as in the paper; its 2:1 size match assumes d_model >> d_state,
        # so at d=32 this stack is ~1.4x the transformer's parameters (the benchmark prints parameter counts)
        blocks, pos = [Block(d, Mamba(d), None, dropout) for _ in range(2 * n)], False
    elif arch == "rwkv":
        blocks, pos = [Block(d, RWKVTimeMix(d, i, n), RWKVChannelMix(d, i, n), dropout) for i in range(n)], False
    elif arch == "hybrid":  # Jamba: Mamba layers carry position and memory, one attention layer does exact recall, MoE on alternate layers
        blocks = [Block(d, Mamba(d), mlp(), dropout), Block(d, Attention(d, n_heads, dropout), mlp(), dropout),
                  Block(d, Mamba(d), MoE(d, dropout), dropout)]
        pos = False
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    return SeqModel(n_features, seq_len, d, blocks, pos, dropout)


if __name__ == "__main__":
    x = torch.randn(4, 30, 11)
    for a in ARCHS:
        m = make_model(a, 11, 30, d_model=32)
        y = m(x)
        assert y.shape == (4,) and torch.isfinite(y).all(), a
        print(f"{a:12s} params {sum(p.numel() for p in m.parameters()):>7,d}")
