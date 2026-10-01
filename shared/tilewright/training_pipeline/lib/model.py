# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GenericTwoTower student + LightGBM LambdaRank teacher.

"Generic" because the same scorer is intended to be deployable on top
of any kernel-selection runtime (hipBLASLt, Composable Kernel, Triton,
...) -- nothing in the model code is hipBLASLt-specific.

Self-contained scorer: a dot-product of two MLP towers (query / item)
with a learned temperature, plus a small MLP on the interaction-feature
vector. Loss is a temperature-scaled KL of the student's softmax against
the teacher's softmax over the per-GEMM candidate group, plus a soft
cross-entropy over tolerance-clamped latency ratios.

Lazy-import torch / lightgbm so importing this module is cheap when
a stage only needs the helpers below.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None
    nn = None

try:
    import lightgbm as lgb

    _LGB_OK = True
except ImportError:
    _LGB_OK = False
    lgb = None


# ── student: GenericTwoTower ─────────────────────────────────────────────────

if _TORCH_OK:

    class GenericTwoTower(nn.Module):
        """Two-tower scorer:
            q_proj: q_dim -> embed_dim   (problem-side embedding)
            i_proj: i_dim -> embed_dim   (kernel-side embedding)
            score  = dot(eq, et) / T + inter_mlp(x)
        where `T` is a learned positive temperature."""

        def __init__(
            self,
            q_dim: int,
            i_dim: int,
            x_dim: int,
            embed_dim: int = 32,
            hidden_dim: int = 64,
            inter_hidden: int = 16,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            def mlp(d_in: int, hid: int, d_out: int, depth: int = 3) -> nn.Sequential:
                layers: List[nn.Module] = []
                d = d_in
                for _ in range(depth - 1):
                    layers += [nn.Linear(d, hid), nn.ReLU()]
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    d = hid
                layers.append(nn.Linear(d, d_out))
                return nn.Sequential(*layers)

            self.q_proj = mlp(q_dim, hidden_dim, embed_dim, depth=3)
            self.i_proj = mlp(i_dim, hidden_dim, embed_dim, depth=2)
            self.inter_mlp = nn.Sequential(
                nn.Linear(x_dim, inter_hidden),
                nn.ReLU(),
                nn.Linear(inter_hidden, 1),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def score_pairs(
            self,
            q_per_pair: "torch.Tensor",
            i_per_pair: "torch.Tensor",
            x_per_pair: "torch.Tensor",
        ) -> "torch.Tensor":
            """All inputs already broadcasted to (N_pairs, dim).
            Returns: (N_pairs,) scores."""
            eg = self.q_proj(q_per_pair)
            et = self.i_proj(i_per_pair)
            T = self.temperature.abs().clamp(min=0.1)
            dot = (eg * et).sum(dim=-1) / T
            ix = self.inter_mlp(x_per_pair).squeeze(-1)
            return dot + ix


# ── whitening helper ─────────────────────────────────────────────────────────


def whiten(
    qvec: np.ndarray, ivec: np.ndarray, xvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """Per-feature zero-mean unit-std using train data. Returns whitened
    arrays + a `norms` dict the trained model serializes alongside its
    state_dict so inference can apply the same transform."""

    def _norm(a):
        mu = a.mean(axis=0, keepdims=True)
        sd = a.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return (a - mu) / sd, mu.ravel(), sd.ravel()

    q_w, q_mu, q_sd = _norm(qvec)
    i_w, i_mu, i_sd = _norm(ivec)
    x_w, x_mu, x_sd = _norm(xvec)
    norms = {
        "q_mean": q_mu.tolist(),
        "q_std": q_sd.tolist(),
        "i_mean": i_mu.tolist(),
        "i_std": i_sd.tolist(),
        "x_mean": x_mu.tolist(),
        "x_std": x_sd.tolist(),
    }
    return (
        q_w.astype(np.float32),
        i_w.astype(np.float32),
        x_w.astype(np.float32),
        norms,
    )


def apply_whiten(
    q: np.ndarray, i: np.ndarray, x: np.ndarray, norms: Dict[str, list]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply train-time `norms` to a held-out feature batch (stage04b
    validation calls this; stage06 deploy embeds it into the export)."""

    def _apply(arr: np.ndarray, mu: np.ndarray, sd: np.ndarray):
        sd = np.where(sd < 1e-6, 1.0, sd)
        return ((arr - mu) / sd).astype(np.float32)

    return (
        _apply(
            q,
            np.asarray(norms["q_mean"], dtype=np.float32),
            np.asarray(norms["q_std"], dtype=np.float32),
        ),
        _apply(
            i,
            np.asarray(norms["i_mean"], dtype=np.float32),
            np.asarray(norms["i_std"], dtype=np.float32),
        ),
        _apply(
            x,
            np.asarray(norms["x_mean"], dtype=np.float32),
            np.asarray(norms["x_std"], dtype=np.float32),
        ),
    )


# ── LightGBM teacher (LambdaRank) ────────────────────────────────────────────


def fit_lgb_teacher(
    qvec: np.ndarray,
    ivec: np.ndarray,
    xvec: np.ndarray,
    gemm_idx: np.ndarray,
    us: np.ndarray,
    winner_us: np.ndarray,
    *,
    n_estimators: int = 300,
    num_leaves: int = 31,
    lr: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Any, np.ndarray]:
    """Train a LambdaRank teacher. Each GEMM is one ranking group; the
    label is a 6-bucket relevance derived from log(us / winner_us).
    Returns (booster, teacher_scores_for_every_pair)."""
    if not _LGB_OK:
        raise SystemExit("lightgbm not installed; pip install lightgbm")
    X = np.concatenate([qvec[gemm_idx], ivec, xvec], axis=1)
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    bins = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    rel = 5 - np.searchsorted(bins, log_ratio, side="right")
    rel = np.clip(rel, 0, 5).astype(np.int32)
    order = np.argsort(gemm_idx, kind="stable")
    Xs = X[order]
    ys = rel[order]
    gids = gemm_idx[order]
    if gids.size == 0:
        raise ValueError(
            "fit_lgb_teacher: no rows after sorting by gemm_idx -- "
            "every gemm in this cell is empty"
        )
    group_sizes = []
    cur = gids[0]
    cnt = 0
    for g in gids:
        if g == cur:
            cnt += 1
        else:
            group_sizes.append(cnt)
            cur = g
            cnt = 1
    group_sizes.append(cnt)
    ds = lgb.Dataset(Xs, label=ys, group=group_sizes, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "verbosity": -1,
        "min_data_in_leaf": 5,
        "feature_pre_filter": False,
        "seed": seed,
    }
    booster = lgb.train(params, ds, num_boost_round=n_estimators)
    # Score the ORIGINAL row order so the student can index by gemm_idx
    # at training time without an inverse permutation.
    teacher_scores = booster.predict(X)
    return booster, teacher_scores.astype(np.float32)
