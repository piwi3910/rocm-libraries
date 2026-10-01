#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage05_train - Grid-based per-cell distill trainer for the ML GEMM
kernel recommender (pipeline_v3).

Input  : one or more folders of enriched chunk_*.csv from stage04.
         Repeating --enriched-csv-dir trains on the UNION (cross-round
         data augmentation in --mode active / --mode full).
Output : per-cell student weights + per-epoch eval metrics + a JSON
         summary.

Pipeline per round:

  1. Load all chunks into memory; drop skip rows (sol_idx_global == "");
     group remaining rows by GEMM.
  2. Assign every GEMM to one of 96 grid cells:
        M, N tier: Tiny<=32, Small<=128, Mid<=512, Large>512
        K    tier: TinyK<=32, MidK<=512, LargeK>512
        batch tier: Bnone (B=1), Bany (B>1)
  3. Per cell:
        a) LightGBM LambdaRank teacher: fit once on (Q+I+X) features,
           ranking by negative log(winner_us / us).
        b) GenericTwoTower student: train for N epochs.
           Loss = lam * KL(teacher_softmax/T || student_softmax/T) * T^2
                + (1-lam) * soft_CE(alpha) on tolerance-clamped log-ratios.
        c) Per epoch: random-sample `eval_sample_frac` of the cell's
           GEMMs and report `sel_eff = winner_us / pick_us` (geomean,
           in (0, 1]; 1.0 = perfect).
  4. Write:
        <out>/cells.json          per-cell metadata + best sel_eff
        <out>/models.pt           per-cell student state_dict + feat norms
        <out>/training_log.csv    epoch | cell | train_loss | sel_eff |
                                  n_eval_sample | n_gemms
        <out>/metrics.json        global + per-cell sel_eff summary
        <out>/train.log           full stdout

Feature engineering: pipeline-v3 catalog from lib/features.py.
  * Leading-dim (log2_lda/b/c/d, ld*_pad_ratio) and swizzle_a/b features
    were dropped: zero variance on the bf16/TN workload (every shape
    uses natural strides, swizzle=0 always).
  * workGroupMapping is a hipBLASLt runtime knob, not a Tensile-codegen
    kernel field, so it never reaches the bench measurement and is not
    a feature.
  * workspace_size / workspace_size_per_elem_c are hardcoded to
    UINT64_MAX in hipblaslt's PredictionLibrary.hpp -- constant across
    every kernel, zero per-kernel signal, dropped.
  * Raw occupancy duplicates occupancy_norm (= occupancy / 9); only the
    normalized form is kept.
  * The hand-tuned `_proxy` features (pgr_eff_proxy, t_math_proxy,
    nlca/nlcb_proxy, wave_pack_proxy) and perfectly-redundant linear
    combinations (log2_tile_area / log2_tile_volume / log2_aspect_mn /
    mt_*_ratio / log2_mn|mk|nk / log2_*_mb) are all dropped from the v3
    catalog.

CLI examples (standalone):
    # full one-shot training on a single round's enriched data
    python3 stage05_train.py \\
        --enriched-csv-dir <run>/round_0/stage04 \\
        --output-dir       <run>/round_0/stage05

    # quick smoke on one cell
    python3 stage05_train.py \\
        --enriched-csv-dir <run>/round_0/stage04 \\
        --output-dir       /tmp/smoke_train \\
        --only-cells       'Tiny|Tiny|TinyK|Bnone' \\
        --epochs 20

    # train on the UNION of two rounds (active-learning data augmentation)
    python3 stage05_train.py \\
        --enriched-csv-dir <run>/round_0/stage04 \\
        --enriched-csv-dir <run>/round_1/stage04 \\
        --output-dir       <run>/round_1/stage05

    # train without the LightGBM teacher (student-only soft_ce)
    python3 stage05_train.py --no-teacher ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import lightgbm as lgb

    _LGB_OK = True
except ImportError:
    _LGB_OK = False
    lgb = None  # type: ignore

# Self-contained import: lib/features.py is vendored inside pipeline_v3.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_V3 = os.path.dirname(THIS_DIR)
# training_pipeline/ lives at <rocm-libraries>/shared/tilewright/training_pipeline.
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_V3, "..", "..", ".."))
if PIPELINE_V3 not in sys.path:
    sys.path.insert(0, PIPELINE_V3)
from lib import features as fs  # noqa: E402

# Origami is wired in lazily inside main() (path is a CLI arg). Stays None
# unless --origami-baseline is passed.
origami = None  # type: ignore
_ORIGAMI_OK = False


# ── terminal UI ───────────────────────────────────────────────────────────────


class C:
    """ANSI color codes; auto-disabled when stdout is not a TTY."""

    _on = sys.stdout.isatty()
    HEADER = "\033[95m" if _on else ""
    BLUE = "\033[94m" if _on else ""
    CYAN = "\033[96m" if _on else ""
    GREEN = "\033[92m" if _on else ""
    YELLOW = "\033[93m" if _on else ""
    RED = "\033[91m" if _on else ""
    GREY = "\033[90m" if _on else ""
    ENDC = "\033[0m" if _on else ""
    BOLD = "\033[1m" if _on else ""


def _banner(title: str, color: str = "") -> None:
    color = color or C.BOLD
    bar = "=" * 80
    print(f"\n{color}{bar}{C.ENDC}", flush=True)
    print(f"{color}  {title}{C.ENDC}", flush=True)
    print(f"{color}{bar}{C.ENDC}", flush=True)


def _info(tag: str, msg: str) -> None:
    print(f"{C.CYAN}[{tag}]{C.ENDC} {msg}", flush=True)


def _ok(tag: str, msg: str) -> None:
    print(f"{C.GREEN}[{tag}]{C.ENDC} {msg}", flush=True)


def _warn(tag: str, msg: str) -> None:
    print(f"{C.YELLOW}[{tag}]{C.ENDC} {msg}", flush=True)


def _err(tag: str, msg: str) -> None:
    print(f"{C.RED}[{tag}]{C.ENDC} {msg}", flush=True)


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return " --.-%"
    return f"{100.0 * num / den:5.1f}%"


def _jsonable(obj: Any) -> Any:
    """Coerce argparse / Path / set values to JSON-serializable types,
    recursing into lists / tuples / dicts. Keeps unknown types as-is
    so json.dump can still throw on actually unsupported types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_jsonable(v) for v in sorted(obj)]
    return obj


def best_sel_eff_pct(r: Dict[str, Any]) -> float:
    """Return the model best_sel_eff as a percentage, NaN-safe."""
    se = r.get("best_sel_eff", float("nan"))
    if se is None or (isinstance(se, float) and math.isnan(se)):
        return float("nan")
    return float(se) * 100.0


# ── grid cell assignment (mirrors phase_8_grid_seed.py) ───────────────────────


def _mn_tier(x: int) -> str:
    if x <= 32:
        return "Tiny"
    if x <= 128:
        return "Small"
    if x <= 512:
        return "Mid"
    return "Large"


def _k_tier(x: int) -> str:
    if x <= 32:
        return "TinyK"
    if x <= 512:
        return "MidK"
    return "LargeK"


def grid_cell(m: int, n: int, k: int, batch: int) -> str:
    """96-cell label: '<Mtier>|<Ntier>|<Ktier>|<Btier>'."""
    bt = "Bnone" if batch == 1 else "Bany"
    return f"{_mn_tier(m)}|{_mn_tier(n)}|{_k_tier(k)}|{bt}"


# Generated for sanity-check / output ordering.
_M_TIERS = ("Tiny", "Small", "Mid", "Large")
_K_TIERS = ("TinyK", "MidK", "LargeK")
_B_TIERS = ("Bnone", "Bany")


def all_cell_labels() -> List[str]:
    return [
        f"{m}|{n}|{k}|{b}"
        for m in _M_TIERS
        for n in _M_TIERS
        for k in _K_TIERS
        for b in _B_TIERS
    ]


# lib/features.py exports the clean v3 catalog; no per-feature excludes
# at this layer.


def _build_feature_lists() -> Tuple[List[str], List[str], List[str]]:
    return (
        fs.query_feature_names(),
        fs.item_feature_names(),
        fs.interaction_feature_names(),
    )


def compute_problem_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a CSV row's problem columns to compute_generic_features kwargs.

    `mi_dtype` is the MATRIX-INSTRUCTION dtype the MFMA hardware
    operates on -- the INPUT dtype, NOT the accumulator. For a
    bf16-in/f32-acc workload (compute_type=c_f32_r, a_type=bf16_r)
    the MI runs `v_mfma_f32_*xbf16` and the instruction dtype is
    bfloat16, not float. Origami's runtime (PredictionLibrary.hpp +
    lib/origami.py:make_problem) sets `mi_dtype = a_dtype` for the
    non-xfloat32 case for exactly this reason; the parity check
    fails when this routine drifts from that convention. The only
    exception is xfloat32, which is an emulated-tf32 path where
    the MI runs on xf32 quantities; we detect that via the
    compute_type field carrying `xf32`.
    """
    compute_type = (row.get("compute_type") or "").strip().lower()
    is_xf32 = "xf32" in compute_type or "xfloat32" in compute_type
    a_dtype = _dt(row["a_type"])
    mi_dtype = _dt("xf32_r") if is_xf32 else a_dtype
    return {
        "m": int(row["m"]),
        "n": int(row["n"]),
        "k": int(row["k"]),
        "batch": int(row["batch_count"]),
        "a_transpose": row["transA"],
        "b_transpose": row["transB"],
        "a_dtype": a_dtype,
        "b_dtype": _dt(row["b_type"]),
        "c_dtype": _dt(row["c_type"]),
        "d_dtype": _dt(row["d_type"]),
        "mi_dtype": mi_dtype,
    }


def compute_config_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a CSV row's kernel columns to compute_generic_features kwargs.

    pipeline_v3 omits `workgroup_mapping` (a hipBLASLt runtime knob, not
    a kernel feature) and `workspace_size*` (hardcoded UINT64_MAX in
    PredictionLibrary.hpp -- no per-kernel signal). The former config_ml_t
    kwargs (cache_hints_c/d/e, prefetch / LDS-RVW / LSU / LDS pads) were
    removed along with the feature group that consumed them.
    """

    def _i(k, d=0):
        v = row.get(k, "")
        if v == "" or v is None:
            return d
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    return {
        "mt_m": _i("mt_m", 1),
        "mt_n": _i("mt_n", 1),
        "mt_k": _i("mt_k", 1),
        "mi_m": _i("mi_m", 1),
        "mi_n": _i("mi_n", 1),
        "mi_k": _i("mi_k", 1),
        "occupancy": max(_i("occupancy", 1), 1),
        "cache_hints_a": _i("cache_hints_a", 0),
        "cache_hints_b": _i("cache_hints_b", 0),
        "grvw_a": max(_i("grvw_a", 1), 1),
        "grvw_b": max(_i("grvw_b", 1), 1),
        "gwvw_d": max(_i("gwvw_d", 1), 1),
    }


_DT_MAP = {
    "f32_r": "float",
    "bf16_r": "bfloat16",
    "f16_r": "half",
    "f8_r": "float8",
    "bf8_r": "bfloat8",
    "xf32_r": "xfloat32",
    "c_f32_r": "float",
}


def _dt(t: str) -> str:
    return _DT_MAP.get((t or "").strip().lower(), "bfloat16")


def features_for_pair(
    problem_kwargs: Dict[str, Any],
    config_kwargs: Dict[str, Any],
    arch: str,
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute the full feature OrderedDict via lib.features.compute_generic_features,
    then return (query, item, interaction) sub-vectors in the order of
    q/i/x names."""
    full = fs.compute_generic_features(
        **problem_kwargs,
        **config_kwargs,
        arch=arch,
        as_dict=True,
    )
    q = [float(full[n]) for n in q_names]
    i = [float(full[n]) for n in i_names]
    x = [float(full[n]) for n in x_names]
    return q, i, x


# ── data loading ──────────────────────────────────────────────────────────────


def load_enriched_chunks(
    folder, quiet: bool, split_tree: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, List[dict]], dict]:
    """Walk chunk_*.csv files and group rows by GEMM, then by grid cell
    (or sub-cell if `split_tree` is non-empty).

    `folder` can be either a single path (string / Path) OR a list of
    paths. When a list is passed we UNION the chunks across all
    folders -- this is how pipeline_v3 implements cross-round data
    augmentation (round_N retrains on round_0's stage04 +
    round_1's stage04 + ... + round_N's stage04 enriched data).

    `split_tree` is the cumulative split tree returned by
    `lib.subcells.load_cumulative_split_tree(prior_round_dirs)`. When
    provided, every GEMM is routed to its leaf sub-cell instead of
    just its base 96-cell label.

    Keeps BOTH tested and skip rows as training candidates:
      - tested rows: sol_idx_global is the real hipblaslt global solution idx.
      - skip rows: no sol_idx_global in the log; we assign a synthetic
        negative id `-(rank + 1)` so the dedup key still works. Skip rows
        carry kernel params (origami) and a `us` label from the single
        warm-up iteration the bench reported, plus an `is_skip` marker.

    Returns:
      cells: dict[cell_label -> list of GEMM dicts]   (sub-cell labels
             when split_tree is non-empty)
      stats: counters (rows_total, tested_rows, skip_rows_used, bad_us,
             gemms, n_source_dirs).
    """
    t0 = time.time()
    folders: List[str] = (
        [folder] if isinstance(folder, (str, os.PathLike)) else list(folder)
    )
    chunk_files: List[Tuple[str, str]] = []  # (folder, fname)
    for fd in folders:
        fd = str(fd)
        if not os.path.isdir(fd):
            continue
        for p in sorted(os.listdir(fd)):
            if p.startswith("chunk_") and p.endswith(".csv"):
                chunk_files.append((fd, p))
    if not chunk_files:
        raise SystemExit(f"no chunk_*.csv in any of {folders}")

    stats = defaultdict(int)
    stats["n_source_dirs"] = len(folders)
    gemms_by_key: Dict[Tuple, dict] = {}
    for fi, (fd, fname) in enumerate(chunk_files, 1):
        path = os.path.join(fd, fname)
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                stats["rows_total"] += 1
                # is_skip is now an explicit column (the enricher fills
                # sol_idx_global for skip rows too via .dat top-level index).
                is_skip = row.get("is_skip", "False") == "True"
                sig = row.get("sol_idx_global", "")
                try:
                    us = float(row["us"])
                except (KeyError, TypeError, ValueError):
                    stats["bad_us"] += 1
                    continue
                if us <= 0 or math.isnan(us):
                    stats["bad_us"] += 1
                    continue
                try:
                    mt_m_check = int(row.get("mt_m", 0) or 0)
                    mi_m_check = int(row.get("mi_m", 0) or 0)
                except (TypeError, ValueError):
                    mt_m_check, mi_m_check = 0, 0
                if mt_m_check == 0 or mi_m_check == 0:
                    stats["bad_kernel_params"] += 1
                    continue
                rank_val = int(row.get("rank", -1) or -1)
                try:
                    sig_int = int(sig) if sig not in ("", None) else None
                except (TypeError, ValueError):
                    sig_int = None
                if sig_int is None:
                    if rank_val < 0:
                        stats["skip_no_rank"] += 1
                        continue
                    sig_int = -(rank_val + 1)
                if is_skip:
                    stats["skip_rows_used"] += 1
                else:
                    stats["tested_rows"] += 1
                key = (
                    int(row["m"]),
                    int(row["n"]),
                    int(row["k"]),
                    int(row["batch_count"]),
                    row["transA"],
                    row["transB"],
                    row["a_type"],
                    row["b_type"],
                    row["c_type"],
                    row["d_type"],
                    row["compute_type"],
                )
                g = gemms_by_key.get(key)
                if g is None:
                    g = {
                        "m": key[0],
                        "n": key[1],
                        "k": key[2],
                        "batch_count": key[3],
                        "transA": key[4],
                        "transB": key[5],
                        "a_type": key[6],
                        "b_type": key[7],
                        "c_type": key[8],
                        "d_type": key[9],
                        "compute_type": key[10],
                        "lda": int(row.get("lda", 0) or 0),
                        "ldb": int(row.get("ldb", 0) or 0),
                        "ldc": int(row.get("ldc", 0) or 0),
                        "ldd": int(row.get("ldd", 0) or 0),
                        "candidates": [],
                    }
                    gemms_by_key[key] = g
                try:
                    sol_idx_local_val = (
                        int(row["sol_idx_local"])
                        if row.get("sol_idx_local") not in ("", None)
                        else -1
                    )
                except (TypeError, ValueError):
                    sol_idx_local_val = -1
                cand = {
                    "sol_idx_global": sig_int,
                    "sol_idx_local": sol_idx_local_val,
                    "us": us,
                    "rank": rank_val,
                    "is_winner": (row.get("is_winner", "False") == "True"),
                    "is_skip": is_skip,
                    "mt_m": int(row.get("mt_m", 0) or 0),
                    "mt_n": int(row.get("mt_n", 0) or 0),
                    "mt_k": int(row.get("mt_k", 0) or 0),
                    "mi_m": int(row.get("mi_m", 0) or 0),
                    "mi_n": int(row.get("mi_n", 0) or 0),
                    "mi_k": int(row.get("mi_k", 0) or 0),
                    "occupancy": max(int(row.get("occupancy", 1) or 1), 1),
                    "cache_hints_a": int(row.get("cache_hints_a", 0) or 0),
                    "cache_hints_b": int(row.get("cache_hints_b", 0) or 0),
                    "grvw_a": max(int(row.get("grvw_a", 1) or 1), 1),
                    "grvw_b": max(int(row.get("grvw_b", 1) or 1), 1),
                    "gwvw_d": max(int(row.get("gwvw_d", 1) or 1), 1),
                    "lda": g["lda"],
                    "ldb": g["ldb"],
                    "ldc": g["ldc"],
                    "ldd": g["ldd"],
                    "transA": g["transA"],
                    "transB": g["transB"],
                    "a_type": g["a_type"],
                    "b_type": g["b_type"],
                    "c_type": g["c_type"],
                    "d_type": g["d_type"],
                    "compute_type": g["compute_type"],
                }
                g["candidates"].append(cand)
        if not quiet and fi % 200 == 0:
            print(
                f"  [load] {fi:>4d}/{len(chunk_files)}  rows={_fmt_int(stats['rows_total']):>11s}  "
                f"gemms={_fmt_int(len(gemms_by_key)):>8s}  "
                f"({_fmt_dur(time.time()-t0)})",
                flush=True,
            )

    if not quiet:
        _ok(
            "load",
            f"chunks: {len(chunk_files)} files  rows={_fmt_int(stats['rows_total'])}  "
            f"tested={_fmt_int(stats['tested_rows'])}  "
            f"skip-used={_fmt_int(stats['skip_rows_used'])}  "
            f"bad_us={_fmt_int(stats['bad_us'])}  "
            f"bad_kp={_fmt_int(stats['bad_kernel_params'])}  "
            f"({_fmt_dur(time.time()-t0)})",
        )

    # Deduplicate candidates per GEMM by sol_idx_global, keep min us.
    # Winner is the tested-row min (skip-row us comes from a single
    # warmup and is noisier — never trust it as the true winner).
    for g in gemms_by_key.values():
        by_idx: Dict[int, dict] = {}
        for c in g["candidates"]:
            sid = c["sol_idx_global"]
            if sid not in by_idx or c["us"] < by_idx[sid]["us"]:
                by_idx[sid] = c
        g["candidates"] = list(by_idx.values())
        tested_us = [c["us"] for c in g["candidates"] if not c.get("is_skip", False)]
        if tested_us:
            winner_us = min(tested_us)
        else:
            winner_us = min(c["us"] for c in g["candidates"])
        g["winner_us"] = winner_us
        for r, c in enumerate(sorted(g["candidates"], key=lambda x: x["us"])):
            c["rank_us"] = r

    cells: Dict[str, List[dict]] = defaultdict(list)
    if split_tree:
        from lib import subcells as sc

        for g in gemms_by_key.values():
            base = grid_cell(g["m"], g["n"], g["k"], g["batch_count"])
            label = sc.assign_subcell(
                base,
                int(g["m"]),
                int(g["n"]),
                int(g["k"]),
                int(g["batch_count"]),
                split_tree,
            )
            g["cell"] = label
            cells[label].append(g)
    else:
        for g in gemms_by_key.values():
            label = grid_cell(g["m"], g["n"], g["k"], g["batch_count"])
            g["cell"] = label
            cells[label].append(g)

    stats["gemms"] = len(gemms_by_key)
    stats["cells_populated"] = len(cells)
    if not quiet:
        _ok(
            "load",
            f"GEMMs: {_fmt_int(stats['gemms'])}  "
            f"populated cells: {stats['cells_populated']}/96",
        )
    return cells, dict(stats)


# ── student model ─────────────────────────────────────────────────────────────

if _TORCH_OK:

    class GenericTwoTower(nn.Module):
        """Two-tower scorer: dot(Q-tower(problem), I-tower(kernel)) /
        learned-T + small MLP on (problem x kernel) interaction
        features. "Generic" because the architecture isn't bound to any
        specific kernel-selection runtime."""

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
                    d = hid
                layers.append(nn.Linear(d, d_out))
                return nn.Sequential(*layers)

            self.q_proj = mlp(q_dim, hidden_dim, embed_dim, depth=3)
            self.i_proj = mlp(i_dim, hidden_dim, embed_dim, depth=2)
            self.inter_mlp = nn.Sequential(
                nn.Linear(x_dim, inter_hidden), nn.ReLU(), nn.Linear(inter_hidden, 1)
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))
            # FUNCTIONAL dropout (applied in score_pairs during training only).
            # Kept OUT of the nn.Sequential modules so the weight-tensor names
            # (q_proj.0/2/4, i_proj.0/2, inter_mlp.0/2) -- which the deploy
            # packer (_WEIGHT_ORDER), the C++ reader, and the parity/eval
            # reconstruction all depend on -- are IDENTICAL with or without
            # dropout. Regularizes the embeddings (the dot term) + the
            # interaction-MLP input (the overfit-prone tile tie-breaker).
            self.dropout_p = float(dropout)

        def forward(
            self, q: torch.Tensor, i: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            """q: (N_gemms, Q)
               i: (T_total, I)  one row per (gemm, candidate) pair
               x: (T_total, X)  one row per (gemm, candidate) pair
               gemm_idx_for_pair: (T_total,) which gemm each pair belongs to
            Returns: scores of shape (T_total,) — caller groups by gemm."""
            # This forward is for the (N_pairs, ?) flat layout that
            # _train_one_epoch uses below: pair-wise concat of (q[gid], i, x).
            # For simplicity we treat this forward as already-paired.
            raise NotImplementedError("use score_pairs() instead")

        def score_pairs(
            self,
            q_per_pair: torch.Tensor,
            i_per_pair: torch.Tensor,
            x_per_pair: torch.Tensor,
        ) -> torch.Tensor:
            """All inputs already broadcasted to (N_pairs, dim).
            Returns: (N_pairs,) scores."""
            eg = self.q_proj(q_per_pair)
            et = self.i_proj(i_per_pair)
            xin = x_per_pair
            p = getattr(self, "dropout_p", 0.0)
            if p > 0 and self.training:
                import torch.nn.functional as _F

                eg = _F.dropout(eg, p=p, training=True)
                et = _F.dropout(et, p=p, training=True)
                xin = _F.dropout(xin, p=p, training=True)
            T = self.temperature.abs().clamp(min=0.1)
            dot = (eg * et).sum(dim=-1) / T
            ix = self.inter_mlp(xin).squeeze(-1)
            return dot + ix


# ── cell-level feature tensors ────────────────────────────────────────────────


def build_cell_tensors(
    gemms: List[dict],
    arch: str,
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
) -> Tuple[Any, ...]:
    """Materialize (q_per_gemm, i_per_pair, x_per_pair, gemm_idx_for_pair,
    us_per_pair, winner_us_per_pair, rank_us_per_pair) for a cell."""
    q_per_gemm: List[List[float]] = []
    i_per_pair: List[List[float]] = []
    x_per_pair: List[List[float]] = []
    gemm_idx_for_pair: List[int] = []
    us_per_pair: List[float] = []
    winner_us_per_pair: List[float] = []
    rank_us_per_pair: List[int] = []

    sol_idx_to_local: Dict[int, int] = {}
    for gi, g in enumerate(gemms):
        # Build query vector from problem only — same for all candidates
        prob_kwargs = compute_problem_kwargs(
            {
                "m": g["m"],
                "n": g["n"],
                "k": g["k"],
                "batch_count": g["batch_count"],
                "transA": g["transA"],
                "transB": g["transB"],
                "a_type": g["a_type"],
                "b_type": g["b_type"],
                "c_type": g["c_type"],
                "d_type": g["d_type"],
                "compute_type": g["compute_type"],
                "lda": g["lda"],
                "ldb": g["ldb"],
                "ldc": g["ldc"],
                "ldd": g["ldd"],
            }
        )
        # Use a placeholder config (mt=mi=defaults) just to get the query vector;
        # we'll only read the q-slice. (item / interaction slices are recomputed
        # per candidate below.)
        first_cand = g["candidates"][0]
        placeholder_cfg = compute_config_kwargs(first_cand)
        q, _, _ = features_for_pair(
            prob_kwargs, placeholder_cfg, arch, q_names, i_names, x_names
        )
        q_per_gemm.append(q)
        winner_us = g["winner_us"]
        for c in g["candidates"]:
            cfg_kwargs = compute_config_kwargs(c)
            _, i_vec, x_vec = features_for_pair(
                prob_kwargs, cfg_kwargs, arch, q_names, i_names, x_names
            )
            i_per_pair.append(i_vec)
            x_per_pair.append(x_vec)
            gemm_idx_for_pair.append(gi)
            us_per_pair.append(c["us"])
            winner_us_per_pair.append(winner_us)
            rank_us_per_pair.append(c["rank_us"])
    return (
        np.asarray(q_per_gemm, dtype=np.float32),
        np.asarray(i_per_pair, dtype=np.float32),
        np.asarray(x_per_pair, dtype=np.float32),
        np.asarray(gemm_idx_for_pair, dtype=np.int64),
        np.asarray(us_per_pair, dtype=np.float64),
        np.asarray(winner_us_per_pair, dtype=np.float64),
        np.asarray(rank_us_per_pair, dtype=np.int64),
    )


def whiten(
    qvec: np.ndarray, ivec: np.ndarray, xvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-feature zero-mean unit-std using train data. Returns whitened
    tensors and a dict of (mean, std) so the saved model can apply at inference."""

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
    return q_w.astype(np.float32), i_w.astype(np.float32), x_w.astype(np.float32), norms


# ── LightGBM teacher ──────────────────────────────────────────────────────────


def fit_lgb_teacher(
    qvec: np.ndarray,
    ivec: np.ndarray,
    xvec: np.ndarray,
    gemm_idx: np.ndarray,
    us: np.ndarray,
    winner_us: np.ndarray,
    n_estimators: int = 300,
    num_leaves: int = 31,
    lr: float = 0.05,
    seed: int = 0,
    quiet: bool = False,
) -> Any:
    """Train a LambdaRank teacher. Each GEMM is one ranking group; label is
    a discretized "how close to winner" relevance (5 = winner, 0 = worst)."""
    if not _LGB_OK:
        raise SystemExit("lightgbm not installed; " "pip install -r requirements.txt")
    # Concatenate features: each pair is its gemm's q + its own i + x
    X = np.concatenate([qvec[gemm_idx], ivec, xvec], axis=1)
    # Relevance label per pair: bucket of log-ratio winner_us/us.
    # log_ratio >= 0; smaller is better (winner has 0).
    # Bucket into 0..5 (higher = closer to winner) at log-space thresholds.
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    bins = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    rel = 5 - np.searchsorted(bins, log_ratio, side="right")
    rel = np.clip(rel, 0, 5).astype(np.int32)
    # LightGBM requires rows sorted by group (gemm_idx) when using `group=`.
    order = np.argsort(gemm_idx, kind="stable")
    Xs = X[order]
    rs = rel[order]
    _, group_sizes = np.unique(gemm_idx[order], return_counts=True)
    # Use the NATIVE Booster API to avoid the lightgbm.sklearn dependency
    # on scikit-learn (which pipeline_v3 deliberately doesn't require).
    train_set = lgb.Dataset(Xs, label=rs, group=group_sizes, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": int(num_leaves),
        "learning_rate": float(lr),
        "verbose": -1 if quiet else 0,
        "force_col_wise": True,
        "seed": int(seed),
        # Pin LightGBM to the SAME thread budget as torch. Unset, LightGBM
        # grabs all cores -- harmless when stage05 runs one cell at a time,
        # but a major oversubscription source under the parallel process
        # pool (each worker's torch.set_num_threads(t) sets this to t).
        "num_threads": max(1, torch.get_num_threads()),
    }
    callbacks = [lgb.log_evaluation(period=0)] if quiet else None
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(n_estimators),
        callbacks=callbacks,
    )
    teacher_scores = booster.predict(X)
    return booster, teacher_scores.astype(np.float32)


# ── per-GEMM softmax helpers ──────────────────────────────────────────────────


def per_gemm_softmax_targets_np(
    us: np.ndarray,
    winner_us: np.ndarray,
    gemm_idx: np.ndarray,
    alpha: float,
    target_tolerance: float = 0.05,
) -> np.ndarray:
    """Soft target: target_j ∝ exp(-alpha * max(0, log(us/winner_us) - tol)).
    Normalized per gemm. Returns array shape (T_total,)."""
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    # Clamp tolerance: anything within `target_tolerance` log-units of the
    # winner is treated as equally good (relative penalty 0).
    adj = np.maximum(log_ratio - target_tolerance, 0.0)
    logits = -alpha * adj
    # Per-gemm softmax
    out = np.zeros_like(logits, dtype=np.float64)
    for gid in np.unique(gemm_idx):
        mask = gemm_idx == gid
        l = logits[mask]
        l = l - l.max()
        e = np.exp(l)
        s = e / np.maximum(e.sum(), 1e-12)
        out[mask] = s
    return out


def per_gemm_softmax_T(
    scores: torch.Tensor, gemm_idx: torch.Tensor, T: float, n_groups: int
) -> torch.Tensor:
    """Vectorized per-group softmax with temperature T.

    Uses scatter_reduce for log-sum-exp normalization, avoiding the
    Python loop over groups that dominated training cost.
    Returns shape (T_total,) probabilities, normalized within each group."""
    s = scores / T
    # 1) max per group, then subtract for numerical stability
    group_max = torch.full((n_groups,), float("-inf"), device=s.device, dtype=s.dtype)
    group_max.scatter_reduce_(0, gemm_idx, s, reduce="amax", include_self=False)
    s_centered = s - group_max.index_select(0, gemm_idx)
    e = torch.exp(s_centered)
    # 2) sum per group, then divide
    group_sum = torch.zeros(n_groups, device=s.device, dtype=s.dtype)
    group_sum.scatter_add_(0, gemm_idx, e)
    return e / group_sum.index_select(0, gemm_idx).clamp(min=1e-12)


def per_gemm_log_softmax(
    scores: torch.Tensor, gemm_idx: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Vectorized per-group log_softmax. Returns shape (T_total,)."""
    group_max = torch.full(
        (n_groups,), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    group_max.scatter_reduce_(0, gemm_idx, scores, reduce="amax", include_self=False)
    s_centered = scores - group_max.index_select(0, gemm_idx)
    e = torch.exp(s_centered)
    group_sum = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
    group_sum.scatter_add_(0, gemm_idx, e)
    log_norm = group_sum.clamp(min=1e-12).log()
    return s_centered - log_norm.index_select(0, gemm_idx)


# ── training loop ─────────────────────────────────────────────────────────────


def evaluate_sel_eff(
    student: Any,
    q_t: torch.Tensor,
    i_t: torch.Tensor,
    x_t: torch.Tensor,
    gemm_idx_t: torch.Tensor,
    us_t: torch.Tensor,
    winner_us_t: torch.Tensor,
    sample_set_mask: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    """Compute geomean sel_eff = winner_us / pick_us over the GEMMs
    selected by `sample_set_mask` (bool over n_gemms).

    Returns a dict with:
      sel_eff           geomean(winner_us / pick_us)  in (0, 1]
      pick_us_geomean   geomean of model's picked-kernel latencies (us)
      winner_us_geomean geomean of empirical-best latencies on the sample
      n_eval            number of GEMMs that contributed to all geomeans

    Tie-breaking is deterministic: among all (pair, gemm) pairs that
    share the group-max score, the LOWEST pair index wins. This avoids
    the non-deterministic behaviour of fancy-index assignment on CUDA
    (where duplicate-write order is undefined) and makes the saved
    "best epoch" reproducible.

    `sample_set_mask` is computed ONCE per cell (in train_cell) so
    every epoch's sel_eff measurement is on the same GEMMs."""
    n_gemms = q_t.shape[0]
    n_pairs = gemm_idx_t.shape[0]
    pair_in_sample = sample_set_mask.index_select(0, gemm_idx_t)

    student.eval()
    with torch.no_grad():
        q_per_pair = q_t.index_select(0, gemm_idx_t)
        scores = student.score_pairs(q_per_pair, i_t, x_t)
        scores_masked = scores.clone()
        scores_masked[~pair_in_sample] = float("-inf")
        group_max = torch.full(
            (n_gemms,), float("-inf"), device=device, dtype=scores_masked.dtype
        )
        group_max.scatter_reduce_(
            0, gemm_idx_t, scores_masked, reduce="amax", include_self=False
        )
        is_pick = (
            scores_masked == group_max.index_select(0, gemm_idx_t)
        ) & pair_in_sample
        # Deterministic pick selection: lowest pair index that hits the
        # group max. amin over masked pair-index values (non-picks get
        # a sentinel larger than any real index).
        sentinel = n_pairs
        pair_indices = torch.arange(n_pairs, device=device, dtype=torch.long)
        masked_pair_idx = torch.where(
            is_pick, pair_indices, torch.full_like(pair_indices, sentinel)
        )
        winner_pair_idx = torch.full(
            (n_gemms,), sentinel, device=device, dtype=torch.long
        )
        winner_pair_idx.scatter_reduce_(
            0, gemm_idx_t, masked_pair_idx, reduce="amin", include_self=False
        )
        winner_us_per_gemm = torch.full(
            (n_gemms,), float("nan"), device=device, dtype=winner_us_t.dtype
        )
        winner_us_per_gemm.scatter_reduce_(
            0, gemm_idx_t, winner_us_t, reduce="amin", include_self=False
        )
        has_pick = (
            (winner_pair_idx < sentinel)
            & sample_set_mask
            & ~torch.isnan(winner_us_per_gemm)
        )
        pick_us_all = us_t[winner_pair_idx.clamp(max=n_pairs - 1)]
        pick_us = pick_us_all[has_pick]
        winner_us_g = winner_us_per_gemm[has_pick]
        log_pick = torch.log(pick_us.clamp(min=1e-9))
        log_win = torch.log(winner_us_g.clamp(min=1e-9))
        log_ratios = log_pick - log_win
    student.train()
    n = int(log_ratios.numel())
    if n == 0:
        return {
            "sel_eff": 0.0,
            "pick_us_geomean": float("nan"),
            "winner_us_geomean": float("nan"),
            "n_eval": 0,
        }
    geomean_ratio = float(torch.exp(log_ratios.mean()).item())
    return {
        "sel_eff": 1.0 / max(geomean_ratio, 1e-9),
        "pick_us_geomean": float(torch.exp(log_pick.mean()).item()),
        "winner_us_geomean": float(torch.exp(log_win.mean()).item()),
        "n_eval": n,
    }


def _select_smart_k_oracle(
    gemms: List[Dict[str, Any]], k: int = 10
) -> List[Tuple[int, ...]]:
    """ORACLE-BASED greedy max-coverage (the original baseline).

    Maximizes `mean(log(winner_us / min_us_in_K))` assuming the
    student would always pick the oracle-best kernel from K. This
    gives the UPPER BOUND on what's achievable from any K-set; it
    becomes optimistic when the student mis-ranks pairs of similar
    kernels. Kept as a reference / sanity check; the deploy
    artifact uses the student-aware variant.
    """
    from lib.dat import sig_from_row as _sig

    kernel_best_us: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(dict)
    winner_us: Dict[int, float] = {}
    for gi, g in enumerate(gemms):
        w = float(g.get("winner_us", 0) or 0)
        if w <= 0:
            continue
        winner_us[gi] = w
        for c in g.get("candidates", []):
            us = float(c.get("us", 0) or 0)
            if us <= 0:
                continue
            sig = tuple(int(v) for v in _sig(c))
            prev = kernel_best_us[sig].get(gi)
            if prev is None or us < prev:
                kernel_best_us[sig][gi] = us
    if not winner_us or not kernel_best_us:
        return []
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    chosen: List[Tuple[int, ...]] = []
    chosen_set: set = set()
    best_us_per_gemm: Dict[int, float] = {gi: float("inf") for gi in winner_us}
    for _step in range(int(k)):
        best_sig: Optional[Tuple[int, ...]] = None
        best_score = -float("inf")
        for sig, us_by_g in kernel_best_us.items():
            if sig in chosen_set:
                continue
            total_log = 0.0
            count = 0
            for gi, _w in winner_us.items():
                cur = best_us_per_gemm[gi]
                cand_us = us_by_g.get(gi)
                if cand_us is not None and cand_us < cur:
                    cur = cand_us
                if cur == float("inf"):
                    continue
                total_log += log_win[gi] - math.log(cur)
                count += 1
            if count == 0:
                continue
            avg_log = total_log / count
            if avg_log > best_score:
                best_score = avg_log
                best_sig = sig
        if best_sig is None:
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, us in kernel_best_us[best_sig].items():
            if us < best_us_per_gemm[gi]:
                best_us_per_gemm[gi] = us
    return chosen


def _student_score_cache(
    gemms: List[Dict[str, Any]],
    student: Any,
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    arch: str,
    norms: Dict[str, Any],
    device: str,
) -> Tuple[Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]], Dict[int, float]]:
    """Build (sig -> {gi: (us, student_score)}) and {gi: winner_us}.

    Filters per (gi, kernel) by `is_kernel_feasible` so the cache
    contains exactly the (sig, gi) pairs the deploy runtime would
    score. Computes student scores in batched MLP forwards (one
    batch per GEMM). Returns the cache + per-GEMM winner_us. The
    student-aware K-selector consumes both.
    """
    from lib.dat import sig_from_row as _sig
    from lib.origami import is_kernel_feasible as _is_feas
    from lib.origami import pool_has_cache_hints as _pool_has_hints

    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]] = defaultdict(
        dict
    )
    winner_us: Dict[int, float] = {}
    student.eval()
    with torch.no_grad():
        for gi, g in enumerate(gemms):
            w = float(g.get("winner_us", 0) or 0)
            if w <= 0:
                continue
            prob_kwargs = compute_problem_kwargs(
                {
                    "m": g["m"],
                    "n": g["n"],
                    "k": g["k"],
                    "batch_count": g["batch_count"],
                    "transA": g["transA"],
                    "transB": g["transB"],
                    "a_type": g["a_type"],
                    "b_type": g["b_type"],
                    "c_type": g["c_type"],
                    "d_type": g["d_type"],
                    "compute_type": g["compute_type"],
                    "lda": g.get("lda", 0),
                    "ldb": g.get("ldb", 0),
                    "ldc": g.get("ldc", 0),
                    "ldd": g.get("ldd", 0),
                }
            )
            us_list: List[float] = []
            sig_list: List[Tuple[int, ...]] = []
            i_rows: List[List[float]] = []
            x_rows: List[List[float]] = []
            q_vec: Optional[List[float]] = None
            _pool_hints = _pool_has_hints(g.get("candidates", []))
            for c in g.get("candidates", []):
                us = float(c.get("us", 0) or 0)
                if us <= 0:
                    continue
                # Same gate the deploy runtime applies. If a kernel
                # wouldn't survive at deploy time, it's irrelevant to
                # K-selection.
                if not _is_feas(
                    g["m"],
                    g["n"],
                    g["k"],
                    g["batch_count"],
                    g["a_type"],
                    g["b_type"],
                    g["transA"],
                    g["transB"],
                    int(c.get("mt_m", 0) or 0),
                    int(c.get("mt_n", 0) or 0),
                    int(c.get("mt_k", 0) or 0),
                    int(c.get("mi_m", 0) or 0),
                    int(c.get("mi_n", 0) or 0),
                    int(c.get("mi_k", 0) or 0),
                    int(c.get("cache_hints_a", 0) or 0),
                    int(c.get("cache_hints_b", 0) or 0),
                    pool_has_hints=_pool_hints,
                ):
                    continue
                cfg_kwargs = compute_config_kwargs(c)
                q, ivec, xvec = features_for_pair(
                    prob_kwargs, cfg_kwargs, arch, q_names, i_names, x_names
                )
                if q_vec is None:
                    q_vec = q
                i_rows.append(ivec)
                x_rows.append(xvec)
                us_list.append(us)
                sig_list.append(tuple(int(v) for v in _sig(c)))
            if not us_list or q_vec is None:
                continue
            winner_us[gi] = w
            n = len(us_list)
            q_arr = np.asarray([q_vec] * n, dtype=np.float32)
            i_arr = np.asarray(i_rows, dtype=np.float32)
            x_arr = np.asarray(x_rows, dtype=np.float32)
            qa, ia, xa = _apply_whiten(q_arr, i_arr, x_arr, norms)
            q_t = torch.tensor(qa, dtype=torch.float32, device=device)
            i_t = torch.tensor(ia, dtype=torch.float32, device=device)
            x_t = torch.tensor(xa, dtype=torch.float32, device=device)
            scores = student.score_pairs(q_t, i_t, x_t).cpu().numpy()
            for j in range(n):
                sig = sig_list[j]
                u = us_list[j]
                s = float(scores[j])
                # If the same sig was benched multiple times on this
                # GEMM (shouldn't happen for unique kernels), keep
                # the best us / first score.
                prev = sig_to_data[sig].get(gi)
                if prev is None or u < prev[0]:
                    sig_to_data[sig][gi] = (u, s)
    return sig_to_data, winner_us


def _greedy_student_aware(
    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]],
    winner_us: Dict[int, float],
    k: int,
) -> List[Tuple[int, ...]]:
    """Two-stage greedy K-list builder:

      Stage 1 (HARD coverage): until every cell GEMM has at least
        ONE feasible sig in `chosen`, pick the sig that covers the
        most still-uncovered GEMMs. Ties broken by regret gain.
        Guarantees that any in-distribution problem the deploy
        runtime sees has at least one K-list sig pass
        `is_kernel_feasible` -> fallback rate is 0 on the training
        distribution.

      Stage 2 (REGRET): for the remaining K - |stage1| slots, pick
        the sig that maximizes the average
        `log(winner_us / student_pick_us)` lift -- same objective
        the previous single-stage greedy used. These slots refine
        the student's picks WITHIN the already-covered GEMM set.

    Why two stages: the single-stage greedy had a silent bug --
    GEMMs whose feasibility constraints were unique (e.g. a TN
    cell with one K-unaligned GEMM requiring cha=chb=0 sigs among
    a sea of K-aligned GEMMs requiring chb=4) could be skipped
    entirely from the objective (`new_pick_us == inf -> continue`)
    while still claiming to "cover" the cell. Result: the K-list
    was high-regret across the majority stratum but had no sig
    feasible for the minority stratum -- deploy hit fallback on
    every minority GEMM. Stage 1 makes this impossible by
    construction.

    Cost: O(K * |sigs| * |gemms|), same as before.
    """
    if not winner_us or not sig_to_data:
        return []
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    chosen: List[Tuple[int, ...]] = []
    chosen_set: set = set()
    best_score_per_gemm: Dict[int, float] = {gi: -float("inf") for gi in winner_us}
    pick_us_per_gemm: Dict[int, float] = {gi: float("inf") for gi in winner_us}

    # ── Stage 1: hard coverage ─────────────────────────────────
    # Repeat until every GEMM is covered OR we ran out of K
    # OR no sig can cover anything new.
    while len(chosen) < int(k):
        n_uncovered = sum(1 for gi in winner_us if pick_us_per_gemm[gi] == float("inf"))
        if n_uncovered == 0:
            break
        best_sig: Optional[Tuple[int, ...]] = None
        best_cover = 0
        best_regret_tiebreak = -float("inf")
        for sig, gi_data in sig_to_data.items():
            if sig in chosen_set:
                continue
            new_cover = 0
            regret_delta = 0.0
            for gi in winner_us:
                cur_pick_us = pick_us_per_gemm[gi]
                cand = gi_data.get(gi)
                if cand is None:
                    continue
                cand_us, cand_score = cand
                if cur_pick_us == float("inf"):
                    new_cover += 1
                    regret_delta += log_win[gi] - math.log(cand_us)
                else:
                    if cand_score > best_score_per_gemm[gi]:
                        regret_delta += math.log(cur_pick_us) - math.log(cand_us)
            if new_cover == 0:
                continue
            if new_cover > best_cover or (
                new_cover == best_cover and regret_delta > best_regret_tiebreak
            ):
                best_cover = new_cover
                best_regret_tiebreak = regret_delta
                best_sig = sig
        if best_sig is None:
            # No remaining sig covers any new GEMM. Move to stage 2;
            # the un-coverable GEMMs aren't in any sig's training
            # data so the deploy can't help them either way.
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, (us, score) in sig_to_data[best_sig].items():
            if score > best_score_per_gemm[gi]:
                best_score_per_gemm[gi] = score
                pick_us_per_gemm[gi] = us

    # ── Stage 2: regret refinement on remaining slots ──────────
    while len(chosen) < int(k):
        best_sig = None
        best_obj = -float("inf")
        for sig, gi_data in sig_to_data.items():
            if sig in chosen_set:
                continue
            total_log = 0.0
            count = 0
            for gi, _w in winner_us.items():
                cur_score = best_score_per_gemm[gi]
                cur_pick_us = pick_us_per_gemm[gi]
                cand = gi_data.get(gi)
                if cand is not None:
                    cand_us, cand_score = cand
                    if cand_score > cur_score:
                        new_pick_us = cand_us
                    else:
                        new_pick_us = cur_pick_us
                else:
                    new_pick_us = cur_pick_us
                if new_pick_us == float("inf"):
                    continue
                total_log += log_win[gi] - math.log(new_pick_us)
                count += 1
            if count == 0:
                continue
            avg = total_log / count
            if avg > best_obj:
                best_obj = avg
                best_sig = sig
        if best_sig is None:
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, (us, score) in sig_to_data[best_sig].items():
            if score > best_score_per_gemm[gi]:
                best_score_per_gemm[gi] = score
                pick_us_per_gemm[gi] = us
    return chosen


# P2b: runtime-weighted smart_K knee. The legacy regret curve averages each
# GEMM EQUALLY, so the knee can shrink K when that barely moves the count-
# geomean even though it slows a few EXPENSIVE GEMMs that dominate wall-clock.
# With ML_SMARTK_RUNTIME_WEIGHT=1 each GEMM's log-ratio is weighted by its
# winner_us (its intrinsic runtime), so the knee preserves K wherever dropping
# a signature would cost real time. Knee-only: the stored sig list + C++ runtime
# are unchanged, so pick parity is unaffected (this only changes HOW MANY sigs
# are kept, biased toward protecting expensive GEMMs).
_SMARTK_RT_WEIGHT = os.environ.get("ML_SMARTK_RUNTIME_WEIGHT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _evaluate_kset_student_aware(
    sig_list: List[Tuple[int, ...]],
    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]],
    winner_us: Dict[int, float],
) -> Dict[str, Any]:
    """Compute deployed sel_eff if the student picks from
    `sig_list` only. Mirrors `evaluate_deployed_sel_eff`'s argmax
    but operates on the precomputed score table. Returns sel_eff,
    n_eval, and a per-K-size regret curve.

    When ML_SMARTK_RUNTIME_WEIGHT is set, the per-K sel_eff is a
    runtime-weighted geomean (weight = winner_us) instead of an
    equal-per-GEMM geomean, so the knee protects expensive GEMMs.
    """
    if not sig_list or not winner_us:
        return {"sel_eff": float("nan"), "n_eval": 0, "regret_curve": []}
    sig_set_progressive: List[set] = []
    cur: set = set()
    for s in sig_list:
        cur = cur | {s}
        sig_set_progressive.append(set(cur))
    # For each prefix length, compute the student-aware sel_eff.
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    rt_weight = _SMARTK_RT_WEIGHT
    regret_curve: List[Tuple[int, float, int]] = []
    for k_size, sk_set in enumerate(sig_set_progressive, start=1):
        total_log = 0.0
        wsum = 0.0
        count = 0
        for gi, _w in winner_us.items():
            best_score = -float("inf")
            best_us = float("inf")
            for sig in sk_set:
                cand = sig_to_data.get(sig, {}).get(gi)
                if cand is None:
                    continue
                u, s = cand
                if s > best_score:
                    best_score = s
                    best_us = u
            if best_us == float("inf"):
                continue
            w = float(_w) if rt_weight else 1.0
            total_log += w * (log_win[gi] - math.log(best_us))
            wsum += w
            count += 1
        if count == 0:
            regret_curve.append((k_size, float("nan"), 0))
            continue
        sel_eff = math.exp(total_log / wsum) if wsum else float("nan")
        regret_curve.append((k_size, sel_eff, count))
    final = regret_curve[-1] if regret_curve else (0, float("nan"), 0)
    return {"sel_eff": final[1], "n_eval": final[2], "regret_curve": regret_curve}


def _adaptive_knee_k(
    regret_curve: List[Tuple[int, float, int]], k_min: int, saturation_eps: float
) -> Tuple[int, Dict[str, Any]]:
    """Pick the per-cell K* from a student-aware regret curve.

    `regret_curve[j] = (k_size, sel_eff, n_covered)` for prefix length
    k_size = j+1 (output of `_evaluate_kset_student_aware`). We want
    the SMALLEST prefix that:
      (1) preserves full GEMM coverage -- `n_covered` equals the
          coverage at the FULL list length (so truncation never drops
          a GEMM the longer list could serve; dropping a covered GEMM
          would also spuriously inflate sel_eff by removing a hard
          case from the average), AND
      (2) is within `saturation_eps` of the BEST sel_eff anywhere on
          the curve (so we don't pay for signatures that add < eps).
    Clamped to >= k_min. This shrinks K for cells whose coverage
    saturates early (selection-time win) and grows it -- up to the
    k_max the curve was built at -- for cells that keep improving
    (the OOD whitelist-coverage win).

    Returns (k_star, info) where info carries the knee diagnostics.
    """
    if not regret_curve:
        return max(1, int(k_min)), {
            "reason": "empty_curve",
            "k_full_curve": 0,
            "k_star": max(1, int(k_min)),
            "best_sel_eff": float("nan"),
            "sel_eff_at_k_star": None,
            "full_coverage": 0,
            "saturation_eps": saturation_eps,
            "k_min": int(k_min),
        }
    k_full = regret_curve[-1][0]
    full_cover = max((c for (_ks, _se, c) in regret_curve), default=0)
    # The saturation target is relative to the best sel_eff among
    # FULL-COVERAGE prefixes only. A short prefix can post a higher
    # sel_eff purely by covering fewer GEMMs (it cherry-picks the easy
    # ones; the hard GEMMs drop out of the average), so the global max
    # would be an unreachable, misleading target. Anchoring to the
    # best full-coverage sel_eff makes the knee both achievable and
    # honest.
    full_sels = [
        se for (_ks, se, c) in regret_curve if c >= full_cover and se == se
    ]  # full-cov, non-NaN
    best_full_sel = max(full_sels) if full_sels else float("nan")
    target = best_full_sel - saturation_eps if best_full_sel == best_full_sel else None
    k_star = k_full  # default: keep everything
    for k_size, se, cov in regret_curve:
        cover_ok = cov >= full_cover
        sel_ok = (target is None) or (se == se and se >= target)
        if cover_ok and sel_ok:
            k_star = k_size
            break
    k_star = max(int(k_min), int(k_star))
    k_star = min(k_star, k_full)
    info = {
        "k_full_curve": k_full,
        "k_star": k_star,
        "best_sel_eff": best_full_sel,
        "sel_eff_at_k_star": next(
            (se for (ks, se, _c) in regret_curve if ks == k_star), None
        ),
        "full_coverage": full_cover,
        "saturation_eps": saturation_eps,
        "k_min": int(k_min),
    }
    return k_star, info


def _seed_oracle_winners(
    gemms: List[Dict[str, Any]], sigs: List[Tuple[int, ...]], n_seed: int, cap: int
) -> List[Tuple[int, ...]]:
    """Force the top-`n_seed` ORACLE max-coverage signatures into `sigs`
    if missing. The oracle greedy's first signatures are the kernels that
    win the most GEMMs in the cell (the dominant tile(s)), so seeding them
    guarantees the deployed student can pick the best kernel. Missing seeds
    are PREPENDED; the merged list is truncated to `cap` (drops the
    lowest-priority tail of the coverage-greedy student list)."""
    if n_seed <= 0:
        return sigs
    seeds = _select_smart_k_oracle(gemms, k=int(n_seed))
    if not seeds:
        return sigs
    have = set(sigs)
    add = [s for s in seeds if s not in have]
    if not add:
        return sigs
    merged = add + list(sigs)
    if cap and len(merged) > int(cap):
        merged = merged[: int(cap)]
    return merged


def select_smart_k_signatures(
    gemms: List[Dict[str, Any]],
    k: int = 10,
    *,
    student: Optional[Any] = None,
    q_names: Optional[List[str]] = None,
    i_names: Optional[List[str]] = None,
    x_names: Optional[List[str]] = None,
    arch: str = "gfx950",
    norms: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    return_metrics: bool = False,
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    saturation_eps: float = 0.005,
    oracle_seed: int = 0,
) -> Any:
    """Pick K kernel signatures for this cell.

    ADAPTIVE-K (opt-in via `k_max`): when `k_max` is given, the greedy
    runs out to `k_max` signatures, then the per-cell K* is chosen at
    the regret-curve knee (smallest prefix that keeps full coverage
    and lands within `saturation_eps` of the best sel_eff), clamped to
    >= `k_min`. This widens K only for cells whose coverage hasn't
    saturated (the OOD-coverage win) and shrinks it for cells that
    saturate early (a selection-time win). Parity-safe: the deployed
    runtime + parity check read whatever-length list is stored, so no
    C++ change is needed. When `k_max` is None, behaves exactly as
    before (greedy to `k`, no truncation).

    Default mode (when `student` is None) is ORACLE-based greedy
    max-coverage -- assumes the deploy student picks the
    oracle-best kernel from K. Backwards-compatible with callers
    that don't have a trained student yet (round-0 bootstrap).

    When `student` (plus feature norms + name lists) is provided,
    runs STUDENT-AWARE greedy max-coverage:
      1. Precompute student score for every (sig, gi) pair the
         deploy runtime would see (after is_kernel_feasible).
      2. Greedy adds the sig that maximizes the average
         `log(winner_us / student_pick_us)`, where
         student_pick_us = us of the kernel the student picks
         from `chosen ∪ {sig}` on each GEMM.
    The student-aware variant matches deploy semantics exactly
    -- it accounts for the student's actual ranking errors, so
    K slots aren't wasted on kernels the student would mis-rank.

    `return_metrics=True` returns
        {"sigs": [...], "metrics": {...}}
    where `metrics` carries the oracle vs student-aware sel_eff
    curves and the active-learning regret-vs-K-size sweep
    (smallest K where regret crosses the pass threshold).
    Default `False` returns just the sig list for back-compat.
    """
    if student is None or norms is None or q_names is None:
        sigs = _select_smart_k_oracle(gemms, k=k)
        if return_metrics:
            return {"sigs": sigs, "metrics": {"mode": "oracle"}}
        return sigs

    sig_to_data, winner_us = _student_score_cache(
        gemms, student, q_names, i_names or [], x_names or [], arch, norms, device
    )
    if not sig_to_data:
        sigs = _select_smart_k_oracle(gemms, k=k)
        if return_metrics:
            return {
                "sigs": sigs,
                "metrics": {"mode": "oracle_fallback", "reason": "empty_cache"},
            }
        return sigs

    # STUDENT-AWARE selection (the production path). When adaptive-K
    # is on, build the list out to `k_max` so the regret curve has
    # room to reveal whether this cell benefits from more than `k`
    # signatures; the knee then truncates back per cell.
    adaptive = k_max is not None
    k_build = int(k_max) if adaptive else int(k)
    eff_k_min = int(k_min) if k_min is not None else 1
    sigs_full = _greedy_student_aware(sig_to_data, winner_us, k_build)

    knee_info: Optional[Dict[str, Any]] = None
    if adaptive:
        # Regret curve over the FULL k_max-length list, then pick the
        # per-cell knee and truncate.
        full_curve = _evaluate_kset_student_aware(sigs_full, sig_to_data, winner_us)[
            "regret_curve"
        ]
        k_star, knee_info = _adaptive_knee_k(full_curve, eff_k_min, saturation_eps)
        sigs_student = sigs_full[:k_star]
    else:
        sigs_student = sigs_full

    # ── ORACLE-SEED (opt-in, default off) ──────────────────────
    # Guarantee the cell's dominant oracle-winning kernel(s) are in
    # the deployed shortlist so the student can always REACH the best
    # (e.g. large macro-tile) kernel, even if the coverage-greedy
    # student list didn't pick it. Large compute-bound cells regress
    # when the winning 256x256-class tile is absent from K. Prepended
    # (so the seed survives), then capped to the build length.
    if oracle_seed and int(oracle_seed) > 0:
        _cap = k_build if adaptive else int(k)
        sigs_student = _seed_oracle_winners(gemms, sigs_student, int(oracle_seed), _cap)

    if not return_metrics:
        return sigs_student

    # ── metrics for the per-cell comparison log ────────────────
    # ORACLE selection (same data) for side-by-side comparison.
    sigs_oracle = _select_smart_k_oracle(gemms, k=len(sigs_student) or k)

    # Student-aware sel_eff for each prefix size 1..K -- the
    # "regret curve". Smallest K where the curve crosses the pass
    # threshold is the data-driven cell K.
    student_eval = _evaluate_kset_student_aware(sigs_student, sig_to_data, winner_us)
    # Same curve for the oracle K-list, evaluated through the
    # student. Lets us see how much the oracle K wastes.
    oracle_eval = _evaluate_kset_student_aware(sigs_oracle, sig_to_data, winner_us)

    metrics = {
        "mode": "student_aware",
        "adaptive_k": bool(adaptive),
        "k_chosen": len(sigs_student),
        "k_build": k_build,
        "knee": knee_info,
        "n_sigs_student": len(sigs_student),
        "n_sigs_oracle": len(sigs_oracle),
        "student_aware_sel_eff": student_eval["sel_eff"],
        "oracle_sel_eff_under_student": oracle_eval["sel_eff"],
        "n_eval_gemms": student_eval["n_eval"],
        "student_aware_regret_curve": student_eval["regret_curve"],
        "oracle_regret_curve": oracle_eval["regret_curve"],
    }
    return {"sigs": sigs_student, "metrics": metrics}


def _round_module_weights_to_bf16_(module: Any) -> None:
    """In-place round every `.weight` parameter of `module` through
    bfloat16 (and back to float32). Used to make the Python deployed-
    eval / parity path see the EXACT SAME weights the C++ runtime sees
    when the deploy .bin stores bf16 weights (stage06
    `--weight-dtype bf16`).

    The round-trip is `w.to(bfloat16).to(float32)` -- identical to the
    writer's `w.to(bfloat16).view(uint16)` (same round-to-nearest-even)
    and to the C++ reader's `uint16 << 16` promotion. So with this
    applied, C++<->Python pick parity stays exact even with bf16
    weights. `.bias` params are left untouched (the .bin keeps them
    float32). No-op for fp32 deploys."""
    import torch  # local; stage05 already imports torch at module top

    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(".weight"):
                p.data = p.data.to(torch.bfloat16).to(torch.float32)


def _round_module_weights_to_int8_(module: Any) -> None:
    """In-place round every `.weight` parameter through the deployed
    per-tensor symmetric int8 grid: scale = max|w|/127, then
    w <- scale * clamp(round(w / scale), -127, 127). Identical to stage06's
    `_tensor_to_int8_bytes` and the C++ `read_weights_int8` dequant, so the
    Python parity / deployed-eval path computes with the EXACT weights the
    runtime loads from an int8 `.bin`. `.bias` left untouched. No-op-safe."""
    import torch

    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(".weight"):
                w = p.data.float()
                amax = float(w.abs().max())
                if amax > 0.0:
                    scale = amax / 127.0
                    p.data = (
                        torch.clamp(torch.round(w / scale), -127, 127) * scale
                    ).to(p.data.dtype)


def _round_module_weights_to_int4_(module: Any) -> None:
    """In-place round every `.weight` through a per-tensor symmetric int4 grid
    (scale = max|w|/7, clamp to [-7, 7]). Mirrors what an int4 `.bin` would
    store/dequant. Used to MEASURE int4's effect on picks before committing the
    C++ int4 reader. `.bias` untouched. No-op-safe."""
    import torch

    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(".weight"):
                w = p.data.float()
                amax = float(w.abs().max())
                if amax > 0.0:
                    scale = amax / 7.0
                    p.data = (torch.clamp(torch.round(w / scale), -7, 7) * scale).to(
                        p.data.dtype
                    )


def evaluate_deployed_sel_eff(
    student: Any,
    gemms: List[Dict[str, Any]],
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    arch: str,
    norms: Dict[str, Any],
    configs_by_key: Dict[Tuple, List[Any]],
    hw: Any,
    smart_k: int,
    device: str,
    smart_k_signatures: Optional[List[Tuple[int, ...]]] = None,
    weight_dtype: str = "fp32",
) -> Dict[str, Any]:
    """DEPLOYED selection efficiency.

    Mirrors the deploy ML-recommender path in
    `origami::ml_recommender::rank_configs` step-for-step:

      1. `check_lds_capacity` -- LDS budget.
      2. `is_kernel_feasible` -- short-circuit; rejects infeasible
         tiles AND enforces the per-problem NTA/NTB rule.
      3. `smart_k_signatures` filter -- restrict candidates to the
         cell's PER-CELL TRAINED smart_K list. The list is computed
         AFTER training via `select_smart_k_signatures` (greedy
         max-coverage on bench data) and persisted in models.pt /
         the deployed `.bin`. NOT sol_idx_global order, NOT
         Origami's analytical ranking.
      4. MLP scores the (at most K) survivors; argmax picks top-1.

    `smart_k_signatures` is the per-cell list of 8-tuples
    (mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, cha, chb). When None or
    empty, the function falls back to scoring every survivor of
    steps 1+2 -- useful for stage 5's same-cell sanity check
    BEFORE the K-list is computed, and as the round-0 bootstrap
    (no prior model -> no K-list yet).

    Returns geomean ratio + counters + per_gemm_log_ratio (an
    np.ndarray aligned with `gemms`: log(pick_us/winner_us) per
    GEMM, NaN for GEMMs we couldn't measure). Stage06 consumes the
    per-gemm log ratios as the CART criterion for split-axis
    selection.
    """
    n_gemms = len(gemms)
    per_gemm_log_ratio = np.full(n_gemms, float("nan"), dtype=np.float64)
    if hw is None or not configs_by_key:
        return {
            "sel_eff": float("nan"),
            "pick_us_geomean": float("nan"),
            "winner_us_geomean": float("nan"),
            "n_eval": 0,
            "n_no_match": 0,
            "n_no_configs": 0,
            "smart_k": smart_k,
            "per_gemm_log_ratio": per_gemm_log_ratio,
        }
    log_pick: List[float] = []
    log_win: List[float] = []
    n_no_match = n_no_configs = 0
    # Match the deployed weight precision. When the .bin ships bf16
    # weights, round a COPY of the student through bf16 so this
    # measurement reflects what the C++ runtime will actually compute
    # (and so stage 7's parity check stays exact). We copy rather than
    # mutate the caller's student object.
    # P3a: ML_EVAL_WEIGHT_DTYPE overrides the eval precision (fp32/bf16/
    # int8/int4) so we can probe a storage dtype's effect on picks WITHOUT
    # re-deploying -- e.g. score the OOB set under int4 vs int8.
    _wd = (os.environ.get("ML_EVAL_WEIGHT_DTYPE") or str(weight_dtype)).lower()
    if _wd in ("bf16", "int8", "int4"):
        import copy as _copy

        student = _copy.deepcopy(student)
        if _wd == "bf16":
            _round_module_weights_to_bf16_(student)
        elif _wd == "int4":
            _round_module_weights_to_int4_(student)
        else:
            _round_module_weights_to_int8_(student)
    student.eval()
    with torch.no_grad():
        for gi, g in enumerate(gemms):
            # Mirror the C++ deploy pipeline (two-pass filter):
            #   Pass 1: is_kernel_feasible + smart_K signature
            #     filter. The signature filter restricts to kernels
            #     in the cell's trained K-list.
            #   Pass 2 (fallback): if pass 1 found nothing for THIS
            #     problem (the K-list members all fail
            #     is_kernel_feasible for this specific GEMM --
            #     common when the K-list was selected from bench
            #     data whose feasibility constraints differ from
            #     this GEMM's), drop the signature filter and use
            #     all is_kernel_feasible survivors.
            # Without the fallback, GEMMs whose feasibility shape
            # doesn't match the K-list would silently report no
            # eligible candidates -- a divergence from the C++
            # runtime, which DOES fall back. Keeping them in lock-
            # step is what makes stage 7's parity check meaningful.
            from lib.origami import is_kernel_feasible as _is_feas
            from lib.origami import pool_has_cache_hints as _pool_has_hints
            from lib.dat import sig_from_row as _row_sig

            sk_set = (
                set(tuple(int(v) for v in s) for s in (smart_k_signatures or []))
                if smart_k_signatures
                else None
            )
            raw_candidates = list(g["candidates"])
            _pool_hints = _pool_has_hints(raw_candidates)

            def _scan(use_sig_filter: bool) -> List[Dict[str, Any]]:
                out_ = []
                for c in raw_candidates:
                    if not _is_feas(
                        g["m"],
                        g["n"],
                        g["k"],
                        g["batch_count"],
                        g["a_type"],
                        g["b_type"],
                        g["transA"],
                        g["transB"],
                        int(c.get("mt_m", 0) or 0),
                        int(c.get("mt_n", 0) or 0),
                        int(c.get("mt_k", 0) or 0),
                        int(c.get("mi_m", 0) or 0),
                        int(c.get("mi_n", 0) or 0),
                        int(c.get("mi_k", 0) or 0),
                        int(c.get("cache_hints_a", 0) or 0),
                        int(c.get("cache_hints_b", 0) or 0),
                        pool_has_hints=_pool_hints,
                    ):
                        continue
                    if use_sig_filter and sk_set is not None:
                        sig = tuple(int(v) for v in _row_sig(c))
                        if sig not in sk_set:
                            continue
                    out_.append(c)
                return out_

            candidates = _scan(use_sig_filter=True)
            if not candidates and sk_set is not None:
                candidates = _scan(use_sig_filter=False)
            if not candidates:
                n_no_configs += 1
                continue
            # Build (q, i_per_pair, x_per_pair) for these candidates.
            prob_kwargs = compute_problem_kwargs(
                {
                    "m": g["m"],
                    "n": g["n"],
                    "k": g["k"],
                    "batch_count": g["batch_count"],
                    "transA": g["transA"],
                    "transB": g["transB"],
                    "a_type": g["a_type"],
                    "b_type": g["b_type"],
                    "c_type": g["c_type"],
                    "d_type": g["d_type"],
                    "compute_type": g["compute_type"],
                    "lda": g["lda"],
                    "ldb": g["ldb"],
                    "ldc": g["ldc"],
                    "ldd": g["ldd"],
                }
            )
            i_rows, x_rows = [], []
            us_rows: List[float] = []
            q_vec = None
            for c in candidates:
                cfg_kwargs = compute_config_kwargs(c)
                q, ivec, xvec = features_for_pair(
                    prob_kwargs, cfg_kwargs, arch, q_names, i_names, x_names
                )
                if q_vec is None:
                    q_vec = q
                i_rows.append(ivec)
                x_rows.append(xvec)
                us_rows.append(float(c["us"]))
            q_arr = np.asarray([q_vec] * len(candidates), dtype=np.float32)
            i_arr = np.asarray(i_rows, dtype=np.float32)
            x_arr = np.asarray(x_rows, dtype=np.float32)
            qa, ia, xa = _apply_whiten(q_arr, i_arr, x_arr, norms)
            q_t = torch.tensor(qa, dtype=torch.float32, device=device)
            i_t = torch.tensor(ia, dtype=torch.float32, device=device)
            x_t = torch.tensor(xa, dtype=torch.float32, device=device)
            scores = student.score_pairs(q_t, i_t, x_t)
            # argmax over the FULL surviving candidate set.
            pick_idx = int(torch.argmax(scores).item())
            pick_us = us_rows[pick_idx]
            winner_us = float(g["winner_us"])
            if pick_us <= 0 or winner_us <= 0:
                continue
            log_pick.append(math.log(pick_us))
            log_win.append(math.log(winner_us))
            # Per-GEMM log_ratio = log(pick_us / winner_us). 0 means
            # the deployed pick equals the bench winner; > 0 means
            # the deployed pick is slower than the bench winner.
            per_gemm_log_ratio[gi] = math.log(pick_us) - math.log(winner_us)
    student.train()
    n_eval = len(log_pick)
    if n_eval == 0:
        return {
            "sel_eff": float("nan"),
            "pick_us_geomean": float("nan"),
            "winner_us_geomean": float("nan"),
            "n_eval": 0,
            "n_no_match": n_no_match,
            "n_no_configs": n_no_configs,
            "smart_k": smart_k,
            "per_gemm_log_ratio": per_gemm_log_ratio,
        }
    pick_g = math.exp(sum(log_pick) / n_eval)
    winner_g = math.exp(sum(log_win) / n_eval)
    return {
        "sel_eff": 1.0 / max(pick_g / max(winner_g, 1e-9), 1e-9),
        "pick_us_geomean": pick_g,
        "winner_us_geomean": winner_g,
        "n_eval": n_eval,
        "n_no_match": n_no_match,
        "n_no_configs": n_no_configs,
        "smart_k": smart_k,
        "per_gemm_log_ratio": per_gemm_log_ratio,
    }


def evaluate_per_cell_held_out_sel_eff(
    enriched_csv_dir: str,
    bundle: Dict[str, Any],
    split_tree: Dict[str, Any],
    configs_by_key: Dict[Tuple, List[Any]],
    hw: Any,
    arch: str,
    device: str = "cpu",
    smart_k: int = 10,
    min_cell_gemms: int = 20,
    quiet: bool = True,
    workload_filter: Optional[Dict[str, str]] = None,
    mx_block_size: int = 0,
) -> Dict[str, Any]:
    """For each leaf cell in `enriched_csv_dir`, compute:
        - the DEPLOYED-pipeline sel_eff of the trained model
          (Origami top-K + ML re-rank; mirrors what bench does at
          runtime, mirrors `evaluate_deployed_sel_eff`), and
        - the ORIGAMI baseline sel_eff (origami's analytical top-1
          on the same GEMM set; mirrors `compute_origami_sel_eff`).
    Returns ONE row per cell + a dataset-level summary.
    NO decision logic, NO retrain/split/pass classification. This is
    the helper stage 8 uses to answer "how does the deployed model
    do on this dataset?" without dragging in the active-learning
    decision machinery (which is stage 04b's job, not stage 8's).

    `workload_filter`: optional dict with any of
        {"a_type", "b_type", "c_type", "d_type",
         "compute_type", "transA", "transB"}.
    Drops every GEMM whose enriched-row fields don't match. Used by
    stage 8 to restrict a multi-dtype bench log to the
    workload-under-test (e.g. only BBS_TN with bf16/bf16). When
    None, every loaded GEMM is evaluated.
    """
    # `lib.subcells` / `lib.grid` are already on sys.path -- stage05
    # imports them at module load. We resolve cells via the leaf
    # labels already stamped onto each gemm by `load_enriched_chunks`,
    # so no extra path setup needed here.
    q_names = list(bundle["q_names"])
    i_names = list(bundle["i_names"])
    x_names = list(bundle["x_names"])
    # Whitening stats live PER-CELL under `m_entry["feature_norms"]`,
    # NOT at the top level. The top-level bundle has q_names/i_names/
    # x_names + the `models` dict; each model entry carries its own
    # `feature_norms`. We grab the per-cell norms below inside the
    # loop. (Earlier code mistakenly read `bundle.get("norms")` here,
    # which is always None -- the resulting empty dict crashed
    # `_apply_whiten` with KeyError: 'q_mean'.)
    models: Dict[str, Any] = bundle["models"]

    new_cells, _load_stats = load_enriched_chunks(
        str(enriched_csv_dir), quiet, split_tree=(split_tree or None)
    )

    # Workload filter: drop GEMMs whose dtype / transpose doesn't
    # match the workload-under-test. A multi-dtype bench log (e.g.
    # production traces from hipBLASLt) carries bf16 + fp16 + fp8 +
    # variant transposes mixed; the config we're validating only
    # trained models for ONE dtype/layout combo (e.g. BBS_TN =
    # bf16/bf16/transA=T/transB=N). Counting non-BBS_TN GEMMs into
    # the per-cell sel_eff would punish the model for picks on
    # workloads it never saw -- and inflate the cell GEMM count
    # past what the workload-under-test actually contains.
    n_before_filter = sum(len(v) for v in new_cells.values())
    n_dropped_by_filter = 0
    if workload_filter:
        # `compute_type` needs special handling: the bench yaml /
        # config writes it as `c_f32_r` (the hipBLASLt CLI form),
        # but stage 04 enrichment STRIPS the `c_` prefix when it
        # writes the CSV row -- so the column ends up `f32_r`. We
        # compare both sides after stripping the prefix so the two
        # spellings unify.
        def _norm(field: str, val: str) -> str:
            s = str(val).strip()
            if field == "compute_type" and s.lower().startswith("c_"):
                s = s[2:]
            return s

        wf = {
            k: _norm(k, v)
            for k, v in workload_filter.items()
            if v is not None and str(v) != ""
        }
        filtered: Dict[str, List[dict]] = {}
        for cell, gs in new_cells.items():
            keep: List[dict] = []
            for g in gs:
                ok = True
                for fk, fv in wf.items():
                    gv = _norm(fk, g.get(fk, ""))
                    if gv != fv:
                        ok = False
                        break
                if ok:
                    keep.append(g)
                else:
                    n_dropped_by_filter += 1
            if keep:
                filtered[cell] = keep
        new_cells = filtered
        if not quiet:
            n_after = sum(len(v) for v in new_cells.values())
            ui_lbl = "[load] "
            try:
                from lib import ui as _ui  # noqa: WPS433

                _ui.info(
                    "filter",
                    f"workload_filter={wf} -- "
                    f"GEMMs: {n_before_filter} -> {n_after} "
                    f"(dropped {n_dropped_by_filter}); "
                    f"populated cells: {len(new_cells)}/96",
                )
            except Exception:
                print(
                    f"{ui_lbl}workload_filter={wf} -- "
                    f"GEMMs: {n_before_filter} -> "
                    f"{n_after}",
                    flush=True,
                )

    def _resolve_model_cell(leaf: str) -> Optional[str]:
        """Same walk-up the deploy runtime does: leaf -> ancestor
        with a model, or None if none of leaf's ancestors are
        trained. Matches C++ `find_cell` in ml_recommender.cpp."""
        cur = leaf
        while cur not in models:
            j = cur.rfind("#")
            if j < 0:
                return None
            cur = cur[:j]
        return cur

    per_cell: List[Dict[str, Any]] = []
    cells_skipped_no_gemms: List[str] = []  # retained for schema; ~empty now
    cells_skipped_no_model: List[str] = []
    cells_via_fallback: List[str] = []
    student_cache: Dict[str, Any] = {}

    # Validation scores EVERY populated cell -- even a single GEMM is
    # signal, so there is NO min-GEMM skip here (min_cell_gemms is kept
    # only for the summary audit). Cells with no own/ancestor model that
    # are BATCHED (Bany) fall back to the same-M/N/K NON-batched (Bnone)
    # model, re-routing each GEMM through the Bnone split tree (batch:=1)
    # -- mirroring the deploy runtime's batch fallback. Those rows are
    # tagged `fallback=True` and carry the serving Bnone `model_cell`.
    from lib import subcells as sc  # noqa: WPS433

    eval_units: List[Tuple[str, str, List[dict], bool]] = []
    for cell_label in sorted(new_cells.keys()):
        gemms = new_cells.get(cell_label) or []
        if len(gemms) < 1:
            continue
        mc = _resolve_model_cell(cell_label)
        if mc is not None:
            eval_units.append((cell_label, mc, gemms, False))
            continue
        base = sc.base_cell(cell_label)
        if base.endswith("|Bany"):
            bnone_base = base[: -len("|Bany")] + "|Bnone"
            groups: Dict[Optional[str], List[dict]] = {}
            for g in gemms:
                leaf = sc.assign_subcell(
                    bnone_base,
                    int(g["m"]),
                    int(g["n"]),
                    int(g["k"]),
                    1,
                    split_tree or {},
                )
                groups.setdefault(_resolve_model_cell(leaf), []).append(g)
            for gmc, gs in groups.items():
                if gmc is None:
                    cells_skipped_no_model.append(cell_label)
                else:
                    eval_units.append((cell_label, gmc, gs, True))
                    if cell_label not in cells_via_fallback:
                        cells_via_fallback.append(cell_label)
        else:
            cells_skipped_no_model.append(cell_label)

    for cell_label, model_cell, gemms, is_fallback in eval_units:
        n_gemms = len(gemms)
        # m_entry always needed (we read feature_norms + smart_k from
        # it regardless of whether the student is already cached).
        m_entry = models[model_cell]
        student = student_cache.get(model_cell)
        if student is None:
            student = GenericTwoTower(
                q_dim=len(q_names),
                i_dim=len(i_names),
                x_dim=len(x_names),
                embed_dim=m_entry["embed_dim"],
                hidden_dim=m_entry["hidden_dim"],
                inter_hidden=m_entry["inter_hidden"],
            ).to(device)
            student.load_state_dict(m_entry["state_dict"])
            student.eval()
            student_cache[model_cell] = student
        norms = m_entry["feature_norms"]
        # Per-cell K-list comes from the trained bundle. Falls back
        # to None (no filter) if the bundle predates the
        # smart_K-as-list schema -- still functional, just doesn't
        # mirror deploy's cap.
        sk_sigs = list(m_entry.get("smart_k_signatures") or [])
        sk_sigs_tuples = [tuple(int(v) for v in s) for s in sk_sigs]
        ev = evaluate_deployed_sel_eff(
            student,
            gemms,
            q_names,
            i_names,
            x_names,
            arch,
            norms,
            configs_by_key,
            hw,
            smart_k=int(smart_k),
            device=device,
            smart_k_signatures=sk_sigs_tuples or None,
        )
        if hw is not None and configs_by_key:
            ob = compute_origami_sel_eff(gemms, configs_by_key, hw, mx_block_size)
        else:
            ob = {
                "sel_eff": float("nan"),
                "pick_us_geomean": float("nan"),
                "winner_us_geomean": float("nan"),
                "n_eval": 0,
            }
        per_cell.append(
            {
                "cell": cell_label,
                "model_cell": model_cell,
                "fallback": bool(is_fallback),
                "n_gemms": n_gemms,
                "model_sel_eff": float(ev.get("sel_eff", float("nan"))),
                "model_n_eval": int(ev.get("n_eval", 0) or 0),
                "model_pick_us_geomean": float(ev.get("pick_us_geomean", float("nan"))),
                "winner_us_geomean": float(ev.get("winner_us_geomean", float("nan"))),
                "origami_sel_eff": float(ob.get("sel_eff", float("nan"))),
                "origami_n_eval": int(ob.get("n_eval", 0) or 0),
                "origami_pick_us_geomean": float(
                    ob.get("pick_us_geomean", float("nan"))
                ),
            }
        )

    def _gm(xs: List[float]) -> float:
        xs = [
            v
            for v in xs
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
        ]
        if not xs:
            return float("nan")
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    def _gm_w(rows: List[Dict[str, Any]], key: str) -> float:
        """GEMM-count-weighted geomean: a 1-GEMM cell barely moves the
        aggregate, so a handful of tiny (often Bany-fallback) cells can't
        tank the headline the way the equal-weight per-cell geomean does."""
        num = 0.0
        den = 0.0
        for r in rows:
            v = r.get(key)
            w = float(r.get("n_gemms", 0) or 0)
            if (
                v is not None
                and not (isinstance(v, float) and math.isnan(v))
                and v > 0
                and w > 0
            ):
                num += w * math.log(v)
                den += w
        return math.exp(num / den) if den > 0 else float("nan")

    # Split primary (own/ancestor model) from Bany->Bnone fallback cells:
    # the fallback routes batched GEMMs to a Bnone model whose features
    # assume batch=1, so those cells are out-of-distribution and score
    # poorly -- reporting them blended into one geomean is misleading.
    prim = [r for r in per_cell if not r.get("fallback")]
    fb = [r for r in per_cell if r.get("fallback")]
    summary = {
        "n_cells_evaluated": len(per_cell),
        "n_cells_skipped_few_gemms": len(cells_skipped_no_gemms),
        "n_cells_skipped_no_model": len(cells_skipped_no_model),
        "n_cells_via_bnone_fallback": len(cells_via_fallback),
        "n_cells_primary": len(prim),
        "min_cell_gemms": int(min_cell_gemms),
        "smart_k": int(smart_k),
        # Workload filter audit -- how many GEMMs the enriched dir
        # provided vs how many survived the workload filter (the
        # dataset's a/b/c/d dtype + transA/transB constraints).
        "n_gemms_before_filter": int(n_before_filter),
        "n_gemms_dropped_by_filter": int(n_dropped_by_filter),
        "workload_filter": dict(workload_filter or {}),
        # Equal-weight per-cell geomean over ALL scored cells (back-compat).
        "model_geomean_sel_eff": _gm([r["model_sel_eff"] for r in per_cell]),
        "origami_geomean_sel_eff": _gm([r["origami_sel_eff"] for r in per_cell]),
        # Primary cells only (exclude Bany->Bnone fallback).
        "model_geomean_primary": _gm([r["model_sel_eff"] for r in prim]),
        "origami_geomean_primary": _gm([r["origami_sel_eff"] for r in prim]),
        # Bany->Bnone fallback cells only (diagnostic; batched OOD).
        "model_geomean_fallback": _gm([r["model_sel_eff"] for r in fb]),
        "origami_geomean_fallback": _gm([r["origami_sel_eff"] for r in fb]),
        # GEMM-count-weighted over ALL cells (most representative of how
        # the deployed model actually serves the workload).
        "model_geomean_gemm_weighted": _gm_w(per_cell, "model_sel_eff"),
        "origami_geomean_gemm_weighted": _gm_w(per_cell, "origami_sel_eff"),
    }
    return {
        "per_cell": per_cell,
        "summary": summary,
        "skipped_no_gemms": cells_skipped_no_gemms,
        "skipped_no_model": cells_skipped_no_model,
        "via_bnone_fallback": cells_via_fallback,
    }


def compute_deployed_top1_picks(
    problems: List[Dict[str, Any]],
    bundle: Dict[str, Any],
    split_tree: Dict[str, Any],
    configs_by_key: Dict[Tuple, List[Any]],
    hw: Any,
    arch: str,
    device: str,
    smart_k: int = 10,
    info_by_key: Optional[Dict[Tuple, List[Dict[str, Any]]]] = None,
    weight_dtype: str = "fp32",
    mx_block_size: int = 0,
) -> List[Dict[str, Any]]:
    """Python mirror of `origami::ml_recommender::rank_configs`: for
    each problem dict, run the deployed inference path end-to-end
    and return the top-1 kernel SIGNATURE.

    Used by stage 7 to verify -- per problem -- that the kernel the
    C++ deployed runtime picks (printed via `MOSAIC_PICK_LOG=1`) matches
    the kernel `models.pt` selects in Python. Mismatch means either
    (a) the deployed `.bin` wasn't loaded (origami fell back to its
    analytical scorer), (b) feature extraction drifted between C++
    and Python, or (c) the cmake build linked stale weights.

    Each problem dict must carry at least:
        m, n, k, batch_count, a_type, b_type, c_type, d_type,
        compute_type, transA, transB

    Pipeline: check_lds_capacity -> is_kernel_feasible -> smart_K
    cap (always on; default 10 matches the deploy default and the
    `train.smart_k` config knob). Walks `configs` in INPUT ORDER
    and keeps the first K survivors. NO Origami analytical
    latency is consulted to pick which K. Setting `smart_k=0`
    disables the cap (debug only).

    Returns one dict per input problem. Stable fields:
        m, n, k, batch       echoed from input
        reason               "ok" / "no_configs" / "no_model" /
                             "no_origami" / ...
    Plus on "ok":
        cell                 leaf-cell label after split-tree walk
        model_cell           ancestor cell whose model was actually
                             used (== `cell` unless the leaf has no
                             model and we walked up; mirrors C++
                             `find_cell` fallback)
        top1_sig             8-tuple (mt_m, mt_n, mt_k, mi_m, mi_n,
                             mi_k, cha, chb) -- matches the C++
                             `[MOSAIC_PICK] top1_sig=(...)` line
                             AND `lib.dat.sig_from_config_t`.
        top1_score           student logit for the picked config
        n_configs            candidates scored
    """
    from lib.origami import make_problem, rank as ori_rank, config_t_sig as _csig
    from lib.dat import sig_from_config_t
    from lib import grid as _grid
    from lib import subcells as _sc

    q_names = list(bundle["q_names"])
    i_names = list(bundle["i_names"])
    x_names = list(bundle["x_names"])
    models: Dict[str, Any] = bundle["models"]

    def _config_t_to_row(
        c: Any, info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Adapt an origami `config_t` to the row dict shape
        `compute_config_kwargs` expects (so we share feature
        extraction code with the CSV path). `occupancy=-1` (origami
        sentinel for 'unset') maps to 1 to match the row default.

        `info` (the matching `kernel_dat_info` dict) is accepted for
        call-site compatibility but no longer consulted: the only fields
        it carried beyond `config_t` were the removed config_ml_t params.
        """
        occ = int(c.occupancy)
        if occ < 0:
            occ = 1
        row = {
            "mt_m": int(c.mt.m),
            "mt_n": int(c.mt.n),
            "mt_k": int(c.mt.k),
            "mi_m": int(c.mi.m),
            "mi_n": int(c.mi.n),
            "mi_k": int(c.mi.k),
            "occupancy": occ,
            "cache_hints_a": int(c.cache_hints_a),
            "cache_hints_b": int(c.cache_hints_b),
            "grvw_a": int(c.grvw_a),
            "grvw_b": int(c.grvw_b),
            "gwvw_d": int(c.gwvw_d),
        }
        return row

    def _resolve_cell(m_: int, n_: int, k_: int, b_: int) -> Tuple[str, Optional[str]]:
        """(leaf_label, model_cell). model_cell is the nearest
        ancestor of leaf_label that has a trained entry in `models`;
        None if no ancestor (including the base 96-cell) does."""
        base = _grid.cell_key(m_, n_, k_, b_)
        leaf = _sc.assign_subcell(base, m_, n_, k_, b_, split_tree or {})
        cur = leaf
        while cur not in models:
            j = cur.rfind("#")
            if j < 0:
                return leaf, None
            cur = cur[:j]
        return leaf, cur

    # Cache student modules per cell (building one per problem is
    # wasteful when stage 7 runs hundreds of shapes in one yaml).
    student_cache: Dict[str, Any] = {}

    def _student_for(model_cell: str) -> Tuple[Any, Dict[str, Any]]:
        s = student_cache.get(model_cell)
        m_entry = models[model_cell]
        if s is None:
            s = GenericTwoTower(
                q_dim=len(q_names),
                i_dim=len(i_names),
                x_dim=len(x_names),
                embed_dim=m_entry["embed_dim"],
                hidden_dim=m_entry["hidden_dim"],
                inter_hidden=m_entry["inter_hidden"],
            ).to(device)
            s.load_state_dict(m_entry["state_dict"])
            # Match the deployed weight precision so this parity check
            # scores against the SAME rounded weights the C++ runtime
            # loads from the .bin. No-op for fp32 deploys.
            _wd = str(weight_dtype).lower()
            if _wd == "bf16":
                _round_module_weights_to_bf16_(s)
            elif _wd == "int8":
                _round_module_weights_to_int8_(s)
            elif _wd == "int4":
                _round_module_weights_to_int4_(s)
            s.eval()
            student_cache[model_cell] = s
        return s, m_entry

    out: List[Dict[str, Any]] = []
    for p in problems:
        ct = (p.get("compute_type") or "").strip()
        if ct and not ct.startswith("c_"):
            ct = "c_" + ct
        try:
            m_ = int(p["m"])
            n_ = int(p["n"])
            k_ = int(p["k"])
            b_ = int(p.get("batch_count", p.get("batch", 1)))
        except (KeyError, TypeError, ValueError) as e:
            out.append({"reason": f"bad_input:{e!r}", **p})
            continue

        cfg_key = (p["a_type"], p["d_type"], ct, p["transA"], p["transB"])
        configs = configs_by_key.get(cfg_key)
        # Optional per-kernel info dict (PGR/PLR/LSU/lds_pad/etc.) --
        # same shape `kernel_dat_info` produces. configs[i] aligns
        # 1:1 with info_list[i] (load_origami_configs_by_dtype builds
        # them in lock-step). When info_by_key is None or missing
        # this cfg_key, Python falls back to Tensile defaults
        # (PGR=1, PLR=1, LSU=1, NTC/D/E=0) which is the SOURCE of
        # parity drift on kernels whose actual values differ.
        info_list = (info_by_key or {}).get(cfg_key) if info_by_key else None
        if not configs:
            out.append({"m": m_, "n": n_, "k": k_, "batch": b_, "reason": "no_configs"})
            continue

        leaf_label, model_cell = _resolve_cell(m_, n_, k_, b_)
        if model_cell is None:
            out.append(
                {
                    "m": m_,
                    "n": n_,
                    "k": k_,
                    "batch": b_,
                    "cell": leaf_label,
                    "reason": "no_model",
                }
            )
            continue

        # Mirror the C++ deploy pipeline exactly:
        #   check_lds_capacity -> is_kernel_feasible -> smart_K
        # where smart_K is the cell's TRAINED kernel-signature
        # list (greedy max-coverage), NOT input order. The list
        # comes from `m_entry["smart_k_signatures"]` (saved by
        # stage 5 / loaded from `models.pt`). When the bundle
        # predates the smart_K-as-list schema, the list is empty and
        # we fall back to "no filter" (score every survivor of
        # the LDS + is_kernel_feasible gates).
        from lib.origami import is_kernel_feasible as _is_feas
        from lib.origami import pool_has_cache_hints as _pool_has_hints

        try:
            problem_t = make_problem(
                origami,
                m_,
                n_,
                k_,
                b_,
                p["a_type"],
                p["b_type"],
                p["d_type"],
                ct,
                p["transA"],
                p["transB"],
                mx_block_size=mx_block_size,
            )
        except Exception as e:
            out.append(
                {
                    "m": m_,
                    "n": n_,
                    "k": k_,
                    "batch": b_,
                    "cell": leaf_label,
                    "reason": f"no_origami:{e!r}",
                }
            )
            continue
        # Cell's K-list of kernel signatures. None / empty => no
        # filter (back-compat with the pre-smart_K-as-list schema).
        m_entry_resolved = models[model_cell]
        sk_sigs_raw = m_entry_resolved.get("smart_k_signatures") or []
        sk_set: Optional[set] = (
            set(tuple(int(v) for v in s) for s in sk_sigs_raw) if sk_sigs_raw else None
        )

        # Two-pass filter, mirroring the C++ deploy runtime
        # (`ml_recommender::rank_configs`):
        #
        #   Pass 1 -- LDS + is_kernel_feasible + smart_K signature
        #             filter. Same as the C++ "primary" path.
        #   Pass 2 -- if pass 1 found nothing, fall back to LDS +
        #             is_kernel_feasible ONLY (skip the signature
        #             filter). Same as the C++ "defensive fallback".
        #
        # The fallback handles cells where the cell-level K-list
        # was selected from bench data that doesn't satisfy
        # is_kernel_feasible for THIS problem -- e.g. the K-list
        # is all NTB=4 kernels, but THIS problem has K not aligned
        # to 128 bytes so is_kernel_feasible requires NTA=NTB=0.
        # Without the fallback the runtime would return no pick on
        # those problems; with it, the model scores the LDS+SC
        # survivors and picks the best one (degraded vs the K-list
        # path, but functional).
        _pool_hints = _pool_has_hints(configs)

        def _scan(use_sig_filter: bool) -> List[Tuple[int, Any]]:
            """Return (cfg_idx, config_t) for each survivor. cfg_idx
            is the index into the original `configs` list and matches
            `info_list[cfg_idx]` 1:1 (kept for recovery/back-compat;
            the config params now come entirely from `config_t`)."""
            out_: List[Tuple[int, Any]] = []
            for ci, cc in enumerate(configs):
                try:
                    if not origami.check_lds_capacity(
                        hw, cc.mt, problem_t.a_dtype, problem_t.b_dtype
                    ):
                        continue
                except Exception:
                    pass
                if not _is_feas(
                    m_,
                    n_,
                    k_,
                    b_,
                    p["a_type"],
                    p["b_type"],
                    p["transA"],
                    p["transB"],
                    int(cc.mt.m),
                    int(cc.mt.n),
                    int(cc.mt.k),
                    int(cc.mi.m),
                    int(cc.mi.n),
                    int(cc.mi.k),
                    int(cc.cache_hints_a),
                    int(cc.cache_hints_b),
                    pool_has_hints=_pool_hints,
                ):
                    continue
                if use_sig_filter and sk_set is not None:
                    sig = (
                        int(cc.mt.m),
                        int(cc.mt.n),
                        int(cc.mt.k),
                        int(cc.mi.m),
                        int(cc.mi.n),
                        int(cc.mi.k),
                        int(cc.cache_hints_a),
                        int(cc.cache_hints_b),
                    )
                    if sig not in sk_set:
                        continue
                out_.append((ci, cc))
            return out_

        candidates = _scan(use_sig_filter=True)
        used_fallback = False
        if not candidates and sk_set is not None:
            candidates = _scan(use_sig_filter=False)
            used_fallback = True

        # O6: opt-in per-kernel feasibility diagnostic. Set
        #   MOSAIC_FEAS_DIAG="m,n,k,b"   -> log gate breakdown for that exact
        #                                   shape, or
        #   MOSAIC_FEAS_DIAG="fallback"  -> log every shape that hit the
        #                                   pass-2 fallback.
        # Shows, of ALL library configs, how many each gate rejected, so a
        # fallback (K-list yielded nothing) is root-caused without guessing:
        #   lds_fail / feas_fail / klist_survivors / nonklist_survivors.
        _fdg = os.environ.get("MOSAIC_FEAS_DIAG")
        if _fdg:
            _hit = used_fallback if _fdg.strip().lower() in ("fallback", "fb") else None
            if _hit is None:
                try:
                    _tt = tuple(int(x) for x in _fdg.split(","))
                    _hit = len(_tt) >= 4 and _tt[:4] == (m_, n_, k_, b_)
                except Exception:
                    _hit = False
            if _hit:
                _n_lds = _n_feas = _n_klist = _n_surv = 0
                for _cc in configs:
                    try:
                        if not origami.check_lds_capacity(
                            hw, _cc.mt, problem_t.a_dtype, problem_t.b_dtype
                        ):
                            _n_lds += 1
                            continue
                    except Exception:
                        pass
                    if not _is_feas(
                        m_,
                        n_,
                        k_,
                        b_,
                        p["a_type"],
                        p["b_type"],
                        p["transA"],
                        p["transB"],
                        int(_cc.mt.m),
                        int(_cc.mt.n),
                        int(_cc.mt.k),
                        int(_cc.mi.m),
                        int(_cc.mi.n),
                        int(_cc.mi.k),
                        int(_cc.cache_hints_a),
                        int(_cc.cache_hints_b),
                        pool_has_hints=_pool_hints,
                    ):
                        _n_feas += 1
                        continue
                    _sig = (
                        int(_cc.mt.m),
                        int(_cc.mt.n),
                        int(_cc.mt.k),
                        int(_cc.mi.m),
                        int(_cc.mi.n),
                        int(_cc.mi.k),
                        int(_cc.cache_hints_a),
                        int(_cc.cache_hints_b),
                    )
                    if sk_set is not None and _sig in sk_set:
                        _n_klist += 1
                    else:
                        _n_surv += 1
                sys.stderr.write(
                    f"[feas_diag] m={m_} n={n_} k={k_} b={b_} "
                    f"cell={model_cell} fallback={used_fallback} | "
                    f"of {len(configs)} configs: lds_fail={_n_lds} "
                    f"feas_fail={_n_feas} klist_survivors={_n_klist} "
                    f"nonklist_survivors={_n_surv} "
                    f"(K-list size={len(sk_set or [])})\n"
                )
                sys.stderr.flush()

        # NO post-filter cap: the C++ runtime scores every survivor
        # of the LDS + is_kernel_feasible (+ optional K-list) filter
        # -- there's no `s_smart_k` cap on the final scoring set.
        # The `smart_k` argument is informational only at this
        # point (forwarded into the result dict for traceability).
        if not candidates:
            out.append(
                {
                    "m": m_,
                    "n": n_,
                    "k": k_,
                    "batch": b_,
                    "cell": leaf_label,
                    "reason": "no_survivors",
                }
            )
            continue

        student, m_entry = _student_for(model_cell)
        norms = m_entry["feature_norms"]

        prob_kwargs = compute_problem_kwargs(
            {
                "m": m_,
                "n": n_,
                "k": k_,
                "batch_count": b_,
                "transA": p["transA"],
                "transB": p["transB"],
                "a_type": p["a_type"],
                "b_type": p["b_type"],
                "c_type": p.get("c_type", p["d_type"]),
                "d_type": p["d_type"],
                "compute_type": p["compute_type"],
            }
        )

        q_vec = None
        i_rows: List[List[float]] = []
        x_rows: List[List[float]] = []
        for ci, c in candidates:
            info = (
                info_list[ci]
                if (info_list is not None and ci < len(info_list))
                else None
            )
            cfg_kwargs = compute_config_kwargs(_config_t_to_row(c, info))
            q, ivec, xvec = features_for_pair(
                prob_kwargs, cfg_kwargs, arch, q_names, i_names, x_names
            )
            if q_vec is None:
                q_vec = q
            i_rows.append(ivec)
            x_rows.append(xvec)
        if not i_rows:
            out.append(
                {
                    "m": m_,
                    "n": n_,
                    "k": k_,
                    "batch": b_,
                    "cell": leaf_label,
                    "reason": "no_features",
                }
            )
            continue

        q_arr = np.asarray([q_vec] * len(candidates), dtype=np.float32)
        i_arr = np.asarray(i_rows, dtype=np.float32)
        x_arr = np.asarray(x_rows, dtype=np.float32)
        qa, ia, xa = _apply_whiten(q_arr, i_arr, x_arr, norms)
        q_t = torch.tensor(qa, dtype=torch.float32, device=device)
        i_t = torch.tensor(ia, dtype=torch.float32, device=device)
        x_t = torch.tensor(xa, dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = student.score_pairs(q_t, i_t, x_t)
        pick_idx = int(torch.argmax(scores).item())
        out.append(
            {
                "m": m_,
                "n": n_,
                "k": k_,
                "batch": b_,
                # Carry transA / transB through so callers can join with
                # the C++ [MOSAIC_PICK] (which also emits them) without
                # collision on multi-layout yamls. A yaml that contains
                # both an (m,n,k,b,T,N) problem and its
                # (...,N,T) variant would alias on the (m,n,k,b) tuple
                # alone; including the transposes makes the key unique.
                "transA": p["transA"],
                "transB": p["transB"],
                "cell": leaf_label,
                "model_cell": model_cell,
                "top1_sig": tuple(
                    int(v) for v in sig_from_config_t(candidates[pick_idx][1])
                ),
                "top1_score": float(scores[pick_idx].item()),
                "n_configs": len(candidates),
                # True iff the smart_K signature filter rejected EVERY
                # config for this problem and we fell back to scoring
                # the LDS+is_kernel_feasible survivors. The C++ runtime
                # uses the same fallback. Stage 7 surfaces this so the
                # operator can see how often the K-list misses on real
                # problems (the K-selector is over-fit to the bench's
                # GEMM distribution; per-problem feasibility may require
                # different NTA/NTB than the K-list provides).
                "used_smart_k_fallback": used_fallback,
                "reason": "ok",
            }
        )
    return out


def _apply_whiten(
    q: np.ndarray, i: np.ndarray, x: np.ndarray, norms: Dict[str, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply train-time per-feature whitening (mean / std). Used by
    evaluate_deployed_sel_eff so the inference path matches what
    stage06_deploy will export."""
    q_mu = np.asarray(norms["q_mean"], dtype=np.float32)
    q_sd = np.asarray(norms["q_std"], dtype=np.float32)
    i_mu = np.asarray(norms["i_mean"], dtype=np.float32)
    i_sd = np.asarray(norms["i_std"], dtype=np.float32)
    x_mu = np.asarray(norms["x_mean"], dtype=np.float32)
    x_sd = np.asarray(norms["x_std"], dtype=np.float32)

    def _do(a, mu, sd):
        sd = np.where(sd < 1e-6, 1.0, sd)
        return ((a - mu) / sd).astype(np.float32)

    return _do(q, q_mu, q_sd), _do(i, i_mu, i_sd), _do(x, x_mu, x_sd)


def gemm_key(g: Dict[str, Any]) -> Tuple:
    """Stable per-GEMM identifier used to align stage05's train/val
    split with stage04b's snapshot-mode evaluation. MUST be computed
    from problem-definition fields only -- never from candidate
    list contents -- so the same key falls out in both stages.

    Stages downstream that need to identify a GEMM across processes
    (stage04b's snapshot mode, dashboards) call this same function.
    """
    return (
        int(g["m"]),
        int(g["n"]),
        int(g["k"]),
        int(g["batch_count"]),
        str(g.get("transA", "")),
        str(g.get("transB", "")),
        str(g.get("a_type", "")),
        str(g.get("b_type", "")),
        str(g.get("c_type", "")),
        str(g.get("d_type", "")),
        str(g.get("compute_type", "")),
    )


def _train_worker_init(threads_per_worker: int) -> None:
    """ProcessPool initializer: HARD-CAP every thread pool a training
    worker can spawn, so `W workers x t threads` stays under the core
    count. A naive parallel pool stalls because each torch worker
    defaults to ALL cores (OMP/MKL/OpenBLAS/LightGBM), oversubscribing
    the host many times over. With spawn, BLAS reads
    OMP_*_NUM_THREADS at import; the parent sets those env vars before the
    pool so workers INHERIT them, and here we also pin torch's intra-op
    pool at runtime (LightGBM is pinned via its `num_threads` param)."""
    t = max(1, int(threads_per_worker))
    for _v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_v] = str(t)
    try:
        torch.set_num_threads(t)
    except Exception:
        pass


def train_cell(
    cell: str,
    gemms: List[dict],
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    arch: str,
    # hyperparams
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    lam: float,
    T: float,
    alpha: float,
    target_tolerance: float,
    n_estimators: int,
    num_leaves: int,
    lgb_lr: float,
    seed: int,
    device: str,
    eval_sample_frac: float,
    use_teacher: bool,
    quiet: bool,
    training_log_writer: Any,
) -> Dict[str, Any]:
    """Train one cell. Returns a dict ready for inclusion in models.pt.

    Per the pipeline contract: stage05 trains the student (and the
    LightGBM teacher) on ALL of the cell's GEMMs -- there is no
    per-cell train/val hold-out inside stage05. The per-epoch
    `sel_eff` reported here is TRAINING-SET ACCURACY and exists only
    so the operator can spot non-convergence and pick the best epoch
    by training fit. The REAL selection-efficiency signal -- on
    GEMM shapes the model has never seen -- comes from stage04b of
    the NEXT round, where stage01's history-dedup guarantees the
    shapes are new.

    Weight initialization: a fresh `GenericTwoTower` is constructed
    every call, so retraining a cell ALWAYS starts from a
    randomly-initialized student -- no warm-start from the prior
    round's weights. This is intentional and load-bearing for the
    active-learning loop: training augmented data with fresh weights
    is what lets data augmentation move the model out of a stale
    local optimum.
    """
    t0 = time.time()
    n_gemms = len(gemms)

    # Seed the global torch RNG from THIS cell's seed so the fresh
    # GenericTwoTower weight init is deterministic per-cell -- previously
    # it consumed the process-global RNG, making results order-dependent
    # (fine sequentially, but unreproducible and unsafe to parallelize).
    # With this, training one cell per worker process is deterministic.
    torch.manual_seed(int(seed))

    (qvec_raw, ivec_raw, xvec_raw, gemm_idx, us, winner_us, rank_us) = (
        build_cell_tensors(gemms, arch, q_names, i_names, x_names)
    )
    qvec, ivec, xvec, norms = whiten(qvec_raw, ivec_raw, xvec_raw)

    if use_teacher:
        booster, teacher_scores = fit_lgb_teacher(
            qvec_raw,
            ivec_raw,
            xvec_raw,
            gemm_idx,
            us,
            winner_us,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            lr=lgb_lr,
            seed=seed,
            quiet=quiet,
        )
        teacher_scores_t = torch.tensor(
            teacher_scores, dtype=torch.float32, device=device
        )
    else:
        booster = None
        teacher_scores_t = None

    # Pre-compute soft targets for soft_CE component
    soft_targets_np = per_gemm_softmax_targets_np(
        us,
        winner_us,
        gemm_idx,
        alpha=alpha,
        target_tolerance=target_tolerance,
    )

    # Move to torch tensors
    q_t = torch.tensor(qvec, dtype=torch.float32, device=device)
    i_t = torch.tensor(ivec, dtype=torch.float32, device=device)
    x_t = torch.tensor(xvec, dtype=torch.float32, device=device)
    gemm_idx_t = torch.tensor(gemm_idx, dtype=torch.long, device=device)
    soft_targets_t = torch.tensor(soft_targets_np, dtype=torch.float32, device=device)
    us_t = torch.tensor(us, dtype=torch.float64, device=device)
    winner_us_t = torch.tensor(winner_us, dtype=torch.float64, device=device)

    # Per-epoch sel_eff reporting subsample. `eval_sample_frac=1.0`
    # (the configured default) scores ALL GEMMs each epoch, which is
    # what we want for stage05 -- the number is training-set accuracy
    # by design (see the docstring above); the real generalization
    # signal comes from the NEXT round's stage04b. Smaller fractions
    # are accepted only as a speed knob for very large cells.
    rng_eval = np.random.default_rng(seed)
    n_eval_sample = max(1, int(round(n_gemms * eval_sample_frac)))
    eval_idx_np = rng_eval.choice(n_gemms, size=n_eval_sample, replace=False)
    eval_idx_np.sort()
    eval_sample_mask = torch.zeros(n_gemms, dtype=torch.bool, device=device)
    eval_sample_mask[torch.tensor(eval_idx_np, dtype=torch.long, device=device)] = True

    # Adaptive model capacity. The OLD tiers (8/16/4 .. 32/64/16) badly
    # UNDERFIT: in a controlled experiment on a hard large-K cell the old
    # 16/32/8 topped out well below 64/160/96, which also GENERALIZED BETTER
    # (smaller in-sample->real gap) -- so the old tiers were underfitting,
    # not over-capacity. The interaction MLP width (ih) is
    # the ranker that resolves near-tied deep-K tiles, so it's grown most.
    # These tiers are ~4-8x the old ones; the soft-distillation loss (5%
    # tolerance) guards the small cells against overfitting.
    # CAPACITY MODE (ML_CAPACITY_MODE):
    #   "difficulty" (default, Option B): size each cell by how many DISTINCT
    #       kernels actually win across its GEMMs -- a pre-train proxy for the
    #       cell's smart-K / ranking difficulty. A cell where only a handful of
    #       kernels ever win needs little ranking capacity; only cells that must
    #       discriminate many near-tied winners get the big tiers. Shrinks the
    #       many EASY cells (the bulk of the binary) while preserving capacity
    #       where it matters -> a much smaller deployed .bin at about equal
    #       quality.
    #       (Insight: low K -> smaller model.)
    #   "n_gemms" (legacy): size purely by training-set GEMM count.
    _cap_mode = os.environ.get("ML_CAPACITY_MODE", "difficulty").strip().lower()
    if _cap_mode == "n_gemms":
        if n_gemms < 150:
            ed, hd, ih = 32, 96, 48
        elif n_gemms < 600:
            ed, hd, ih = 64, 160, 96
        elif n_gemms < 2000:
            ed, hd, ih = 80, 192, 112
        else:
            ed, hd, ih = 96, 256, 128
    else:
        # Distinct (near-)winning kernels: a per-pair `us` at/below its gemm's
        # winner us marks that pair as a (near-)winner; its item-feature row
        # identifies the kernel. `winner_us` is per-pair (carries each pair's
        # gemm winner us).
        _wmask = us <= (winner_us * 1.001)
        _wkeys = set()
        for _j in np.nonzero(_wmask)[0]:
            _wkeys.add(tuple(np.round(ivec_raw[_j], 3)))
        n_distinct_winners = max(1, len(_wkeys))
        if n_gemms < 60 or n_distinct_winners <= 2:
            ed, hd, ih = 16, 48, 24
        elif n_distinct_winners <= 6:
            ed, hd, ih = 24, 80, 40
        elif n_distinct_winners <= 14:
            ed, hd, ih = 40, 112, 56
        elif n_distinct_winners <= 28:
            ed, hd, ih = 64, 160, 96
        else:
            ed, hd, ih = 80, 192, 112
    # Experiment hook: ML_FORCE_CAPACITY="ed,hd,ih" overrides the
    # n_gemms-adaptive sizing (used to probe whether small cells are
    # capacity-limited). Unset => unchanged behavior.
    _force_cap = os.environ.get("ML_FORCE_CAPACITY")
    if _force_cap:
        try:
            ed, hd, ih = (int(v) for v in _force_cap.split(","))
        except Exception:
            pass
    # Big-regularized large-cell experiment: large compute-bound cells overfit
    # (near-perfect in-sample, held-out collapses), so give ONLY those cells a larger
    # capacity AND dropout (regularize the extra capacity). ML_LARGE_CELL_CAPACITY
    # ="ed,hd,ih" applies to cells whose base tier is Large M / Large N / LargeK;
    # ML_DROPOUT sets the student dropout (q/i towers + interaction MLP). Both
    # default off => unchanged behavior.
    from lib import subcells as sc  # noqa: WPS433 (local; matches other uses)

    _base = sc.base_cell(cell).split("|")
    _is_large = len(_base) >= 3 and (
        _base[0] == "Large" or _base[1] == "Large" or _base[2] == "LargeK"
    )
    _large_cap = os.environ.get("ML_LARGE_CELL_CAPACITY")
    if _large_cap and _is_large:
        try:
            ed, hd, ih = (int(v) for v in _large_cap.split(","))
        except Exception:
            pass
    _dropout = 0.0
    try:
        _dropout = float(os.environ.get("ML_DROPOUT", "0") or 0)
    except Exception:
        _dropout = 0.0
    # Apply dropout only where capacity is non-trivial (large/regularized cells);
    # tiny cells don't need it and the soft-distillation loss already guards them.
    _cell_dropout = (
        _dropout if (_dropout > 0 and (_is_large or _large_cap or _force_cap)) else 0.0
    )
    student = GenericTwoTower(
        q_dim=len(q_names),
        i_dim=len(i_names),
        x_dim=len(x_names),
        embed_dim=ed,
        hidden_dim=hd,
        inter_hidden=ih,
        dropout=_cell_dropout,
    )
    student.to(device)
    student.train()
    optim = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_sel_eff = -1.0  # higher is better; init to "any value beats this"
    best_pick_us_geomean: float = float("nan")
    best_epoch = -1
    best_state: Optional[Dict[str, Any]] = None
    per_epoch_metrics: List[Dict[str, Any]] = []
    n_groups = int(n_gemms)
    # winner_us geomean on the eval sample is a fixed baseline (depends
    # only on the data, not the model). Compute once and reuse for the
    # full-training output dict.
    eval_winner_us_np = winner_us[eval_idx_np]
    eval_winner_us_pos = eval_winner_us_np[eval_winner_us_np > 0]
    eval_winner_us_geomean = (
        float(math.exp(np.mean(np.log(eval_winner_us_pos))))
        if eval_winner_us_pos.size
        else float("nan")
    )
    for ep in range(epochs):
        optim.zero_grad()
        q_per_pair = q_t.index_select(0, gemm_idx_t)
        scores = student.score_pairs(q_per_pair, i_t, x_t)
        # Vectorized per-gemm log_softmax (was a Python loop over groups).
        log_s = per_gemm_log_softmax(scores, gemm_idx_t, n_groups)
        # Training loss is over ALL pairs (= ALL GEMMs) -- there is no
        # held-out subset inside stage05. The "is this model good?"
        # question is answered by stage04b of the NEXT round, where
        # the new shapes are by-construction unknown to this model.
        ce = -(soft_targets_t * log_s).sum() / float(n_gemms)
        if use_teacher and teacher_scores_t is not None:
            t_soft = per_gemm_softmax_T(
                teacher_scores_t, gemm_idx_t, T=T, n_groups=n_groups
            )
            s_soft = per_gemm_softmax_T(scores, gemm_idx_t, T=T, n_groups=n_groups)
            kl_pair = t_soft * (
                torch.log(t_soft.clamp(min=1e-12)) - torch.log(s_soft.clamp(min=1e-12))
            )
            kl = kl_pair.sum() / float(n_gemms) * (T**2)
            loss = lam * kl + (1.0 - lam) * ce
        else:
            loss = ce
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optim.step()
        train_loss = float(loss.item())

        ev = evaluate_sel_eff(
            student,
            q_t,
            i_t,
            x_t,
            gemm_idx_t,
            us_t,
            winner_us_t,
            sample_set_mask=eval_sample_mask,
            device=device,
        )
        sel_eff = ev["sel_eff"]
        n_eval = ev["n_eval"]
        per_epoch_metrics.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "sel_eff": sel_eff,
                "pick_us_geomean": ev["pick_us_geomean"],
                "n_eval_sample": n_eval,
            }
        )
        if training_log_writer is not None:
            training_log_writer.writerow(
                {
                    "cell": cell,
                    "epoch": ep,
                    "train_loss": f"{train_loss:.6f}",
                    "sel_eff": f"{sel_eff:.6f}",
                    "pick_us_geomean": f"{ev['pick_us_geomean']:.4f}",
                    "n_eval_sample": n_eval,
                    "n_gemms": n_gemms,
                }
            )
        if sel_eff > best_sel_eff:
            best_sel_eff = sel_eff
            best_pick_us_geomean = ev["pick_us_geomean"]
            best_epoch = ep
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
        if not quiet and (ep < 3 or ep % 10 == 0 or ep == epochs - 1):
            print(
                f"  {C.GREY}[{cell}]{C.ENDC} ep {ep:>3d}/{epochs-1}  "
                f"train_loss={train_loss:7.4f}  "
                f"sel_eff={sel_eff*100:.2f}%  "
                f"pick_us={ev['pick_us_geomean']:.2f}  "
                f"({n_eval} eval)",
                flush=True,
            )

    # Final state: restore best
    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    out = {
        "cell": cell,
        "n_gemms": n_gemms,
        "n_pairs": int(ivec_raw.shape[0]),
        "q_dim": len(q_names),
        "i_dim": len(i_names),
        "x_dim": len(x_names),
        "embed_dim": ed,
        "hidden_dim": hd,
        "inter_hidden": ih,
        "best_epoch": best_epoch,
        # `best_sel_eff` here is TRAINING-SET accuracy (per-epoch eval
        # over all GEMMs in this cell). The real generalization
        # number is reported by the next round's stage04b.
        "best_sel_eff": best_sel_eff,
        "best_pick_us_geomean": best_pick_us_geomean,
        "winner_us_geomean": eval_winner_us_geomean,
        "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "feature_norms": norms,
        "epoch_metrics": per_epoch_metrics,
        "elapsed_s": time.time() - t0,
    }
    return out


# ── origami baseline (optional, --origami-baseline) ──────────────────────────

# Imported, not re-declared: the private copy that used to live here was
# missing the 'f8_r' key (configs/CSVs never say 'fp8_r'), so every f8
# problem was built with bf16 dtypes and Origami's baseline ranking for
# the whole F8 family was wrong. One dict now, in lib/origami.py.
from lib.origami import _DTYPE_TO_ORIGAMI  # noqa: E402
from lib.origami import _rank_configs_compat  # noqa: E402


def _cand_sig(c: Dict[str, Any]) -> Tuple[int, ...]:
    """11-tuple identifier for a benched candidate: MT (m/n/k) + MI
    (m/n/k) + cache_hints_a/b + grvw_a/b + gwvw_d. Mirrors the schema
    in lib/dat.py and lib/origami.py."""
    return (
        int(c["mt_m"]),
        int(c["mt_n"]),
        int(c["mt_k"]),
        int(c["mi_m"]),
        int(c["mi_n"]),
        int(c["mi_k"]),
        int(c["cache_hints_a"]),
        int(c["cache_hints_b"]),
        int(c["grvw_a"]),
        int(c["grvw_b"]),
        int(c["gwvw_d"]),
    )


def _origami_cfg_sig(c: Any) -> Tuple[int, ...]:
    """Same 11-tuple, from an origami.config_t."""
    return (
        int(c.mt.m),
        int(c.mt.n),
        int(c.mt.k),
        int(c.mi.m),
        int(c.mi.n),
        int(c.mi.k),
        int(c.cache_hints_a),
        int(c.cache_hints_b),
        int(c.grvw_a),
        int(c.grvw_b),
        int(c.gwvw_d),
    )


def _make_origami_problem(g: Dict[str, Any], mx_block_size: int = 0) -> Any:
    p = origami.problem_t()
    p.size = origami.dim3_t(int(g["m"]), int(g["n"]), int(g["k"]))
    p.batch = int(g["batch_count"])
    p.a_transpose = (
        origami.transpose_t.T if g["transA"] == "T" else origami.transpose_t.N
    )
    p.b_transpose = (
        origami.transpose_t.T if g["transB"] == "T" else origami.transpose_t.N
    )
    a_dt = _DTYPE_TO_ORIGAMI.get(g["a_type"], "bf16")
    d_dt = _DTYPE_TO_ORIGAMI.get(g["d_type"], a_dt)
    p.a_dtype = origami.string_to_datatype(a_dt)
    p.b_dtype = origami.string_to_datatype(_DTYPE_TO_ORIGAMI.get(g["b_type"], a_dt))
    p.c_dtype = origami.string_to_datatype(d_dt)
    p.d_dtype = origami.string_to_datatype(d_dt)
    if "xf32" in (g.get("compute_type", "") or "").lower():
        p.mi_dtype = origami.string_to_datatype("xf32")
    else:
        p.mi_dtype = origami.string_to_datatype(a_dt)
    # 32 for MXFP8 (hipblaslt scaleA/scaleB == 3), 0 otherwise -- an MX
    # problem ranked with block_size 0 is scored as plain F8.
    p.a_mx_block_size = int(mx_block_size)
    p.b_mx_block_size = int(mx_block_size)
    return p


def compute_origami_sel_eff(
    gemms: List[Dict[str, Any]],
    configs_by_key: Dict[Tuple, List[Any]],
    hw: Any,
    mx_block_size: int = 0,
) -> Dict[str, Any]:
    """For each GEMM in the cell:
      1. rank_configs(problem, hw, configs)
      2. walk top-1 -> top-N, matching origami's pick sig against this
         GEMM's measured candidates
      3. take the first match's us, accumulate pick_us/winner_us
    Returns sel_eff + geomean(pick_us) + geomean(winner_us) + counters.
    No-match GEMMs are excluded from the geomeans."""
    log_pick: List[float] = []
    log_win: List[float] = []
    n_rank0 = 0
    n_no_match = 0
    n_no_configs = 0
    n_rank_err = 0
    for g in gemms:
        # configs_by_key is built from the .dat filenames which encode the
        # compute type with the c_-prefix ('c_f32_r'); the bench log uses
        # the un-prefixed form ('f32_r'). Normalize before lookup.
        ct = (g.get("compute_type") or "").strip()
        if ct and not ct.startswith("c_"):
            ct = "c_" + ct
        cfg_key = (g["a_type"], g["d_type"], ct, g["transA"], g["transB"])
        configs = configs_by_key.get(cfg_key)
        if not configs:
            n_no_configs += 1
            continue
        cand_us_by_sig: Dict[Tuple[int, ...], float] = {}
        for c in g["candidates"]:
            sig = _cand_sig(c)
            if sig not in cand_us_by_sig or c["us"] < cand_us_by_sig[sig]:
                cand_us_by_sig[sig] = float(c["us"])
        try:
            ranked = _rank_configs_compat(
                origami, _make_origami_problem(g, mx_block_size), hw, configs
            )
        except Exception:
            n_rank_err += 1
            continue
        picked_us = None
        for rk, r in enumerate(ranked):
            us_v = cand_us_by_sig.get(_origami_cfg_sig(r.config))
            if us_v is not None and us_v > 0:
                picked_us = us_v
                if rk == 0:
                    n_rank0 += 1
                break
        if picked_us is None:
            n_no_match += 1
            continue
        winner = float(g["winner_us"])
        if winner <= 0:
            continue
        log_pick.append(math.log(picked_us))
        log_win.append(math.log(winner))
    n_eval = len(log_pick)
    if n_eval == 0:
        return {
            "sel_eff": float("nan"),
            "pick_us_geomean": float("nan"),
            "winner_us_geomean": float("nan"),
            "n_eval": 0,
            "n_rank0_hit": 0,
            "n_no_match": n_no_match,
            "n_no_configs": n_no_configs,
            "n_rank_err": n_rank_err,
        }
    pick_g = math.exp(sum(log_pick) / n_eval)
    winner_g = math.exp(sum(log_win) / n_eval)
    return {
        "sel_eff": 1.0 / max(pick_g / max(winner_g, 1e-9), 1e-9),
        "pick_us_geomean": pick_g,
        "winner_us_geomean": winner_g,
        "n_eval": n_eval,
        "n_rank0_hit": n_rank0,
        "n_no_match": n_no_match,
        "n_no_configs": n_no_configs,
        "n_rank_err": n_rank_err,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pipeline_v3 stage05: grid-based per-cell distill "
        "trainer (LightGBM teacher + GenericTwoTower student)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--enriched-csv-dir",
        required=True,
        action="append",
        help="folder of chunk_*.csv. REPEATABLE: pass once per "
        "round to train on the UNION of every round's "
        "enriched data (cross-round data augmentation).",
    )
    ap.add_argument(
        "--prior-round-dir",
        "--prior-run-dir",
        type=Path,
        action="append",
        default=None,
        dest="prior_round_dir",
        help="repeatable; each entry MUST point at a "
        "`<run>/round_<i>/` ROUND directory. Used to "
        "load every prior round's stage04b/splits.json "
        "so GEMMs route to their leaf sub-cell instead "
        "of just their base 96-cell label. "
        "(`--prior-run-dir` is kept as a deprecated "
        "alias.)",
    )
    ap.add_argument(
        "--current-round-dir",
        type=Path,
        default=None,
        help="the CURRENT round's directory. Used to ALSO "
        "load this round's stage04b/splits.json into "
        "the cumulative split tree so GEMMs in this "
        "round's stage04 chunks get routed to the "
        "child labels stage04b JUST decided -- "
        "without this, round_R/stage05 sees the new "
        "child labels as empty (the GEMMs are still "
        "labeled with the parent) and `to_train=0`. "
        "Active-mode orchestrator passes this; "
        "round_0 doesn't (no stage04b in round 0).",
    )
    ap.add_argument(
        "--smart-k",
        type=int,
        default=10,
        help="ALWAYS-ON candidate cap after the "
        "check_lds_capacity + is_kernel_feasible "
        "pre-filter (the same gates the deployed "
        "C++ runtime applies). Walks survivors in "
        "INPUT ORDER (sol_idx_global at training "
        "time; Tensile's solution registration "
        "order at deploy time) and keeps the "
        "first K. NO Origami analytical latency "
        "is consulted to choose which K. Default "
        "10 matches the deploy default. K=0 means "
        "no cap (debug only).",
    )
    # Adaptive-K (opt-in). When --smart-k-max is set, the per-cell
    # signature list is grown to k_max, then truncated at the
    # regret-curve knee within [k_min, k_max]: cells whose coverage
    # saturates early keep a SHORT list (selection-time win); cells
    # still improving keep a LONGER list up to k_max (the OOD
    # whitelist-coverage win). Parity-safe -- the deployed runtime
    # and the stage 7 parity check read whatever-length list is
    # stored, so no C++ change is required. Leave --smart-k-max
    # unset to reproduce the legacy fixed-K behavior exactly.
    ap.add_argument(
        "--smart-k-max",
        type=int,
        default=None,
        help="(adaptive-K, opt-in) max signatures per cell. "
        "When set, enables the regret-curve knee "
        "selection in [--smart-k-min, --smart-k-max]. "
        "Unset = legacy fixed --smart-k.",
    )
    ap.add_argument(
        "--smart-k-min",
        type=int,
        default=4,
        help="(adaptive-K) floor on per-cell signatures.",
    )
    ap.add_argument(
        "--smart-k-saturation-eps",
        type=float,
        default=0.005,
        help="(adaptive-K) stop growing K once the next "
        "signature adds < this much sel_eff (0.005 = "
        "0.5 percentage points). The knee is the "
        "smallest K within eps of the best sel_eff that "
        "ALSO keeps full GEMM coverage.",
    )
    ap.add_argument(
        "--smartk-fallback-delta",
        type=float,
        default=0.0,
        help="QUALITY-BASED smart_K fallback. When a cell's "
        "in-sample deployed sel_eff is this many points "
        "BELOW the full-ranking model (k_delta = deployed - "
        "model <= this NEGATIVE threshold), the cell's "
        "smart_K whitelist cannot represent its winning "
        "kernels, so we CLEAR the whitelist -> the runtime "
        "ranks ALL feasible candidates (full ranking) for "
        "that cell, recovering the lost sel_eff at the cost "
        "of selection time. 0 (default) = OFF. Typical: "
        "-0.02 (fall back when deployed is >=2 percentage "
        "points below model).",
    )
    ap.add_argument(
        "--oracle-seed",
        type=int,
        default=0,
        help="force the top-N oracle max-coverage signatures "
        "(the cell's dominant winning kernels) into the "
        "deployed shortlist so the student can always reach "
        "the best tile. 0 (default) = off, baseline "
        "coverage-greedy behavior unchanged.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="where to write models.pt + metrics + training_log.csv",
    )
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="distillation weight: loss = lam*KL + (1-lam)*soft_CE",
    )
    ap.add_argument("--T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument(
        "--alpha",
        type=float,
        default=12.0,
        help="soft-CE concentration around the winner",
    )
    ap.add_argument(
        "--target-tolerance",
        type=float,
        default=0.05,
        help="log-ratio tolerance for soft-CE (within this much "
        "of the winner -> same target relevance)",
    )
    ap.add_argument(
        "--n-estimators", type=int, default=300, help="LightGBM teacher boosting rounds"
    )
    ap.add_argument(
        "--num-leaves", type=int, default=31, help="LightGBM teacher num_leaves"
    )
    ap.add_argument(
        "--lgb-lr", type=float, default=0.05, help="LightGBM teacher learning rate"
    )
    ap.add_argument(
        "--no-teacher",
        action="store_true",
        help="skip LightGBM teacher; train student with soft-CE only",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device", default="cpu", help="torch device (cpu / cuda / cuda:0 / hip / ...)"
    )
    ap.add_argument(
        "--train-workers",
        type=int,
        default=1,
        help="parallel per-cell TRAINING workers (spawn process "
        "pool over train_cell). 1 = sequential (default). "
        ">1 trains cells concurrently with hard per-worker "
        "thread caps (W*threads_per_worker <= cores) to "
        "avoid the oversubscription that stalls naive "
        "parallelism. Origami baseline / smart_K / deployed "
        "eval stay serial in the parent.",
    )
    ap.add_argument(
        "--train-threads-per-worker",
        type=int,
        default=0,
        help="torch/BLAS/LightGBM threads per training worker. "
        "0 = AUTO = cpu_count // actual_workers (use the "
        "whole machine). Explicit value pins it.",
    )
    # --eval-sample-frac: per-epoch sel_eff reporting subsample. The
    # student trains on ALL of the cell's GEMMs (no hold-out inside
    # stage05); this knob only controls how many GEMMs the per-epoch
    # `sel_eff` is computed over. 1.0 = all GEMMs (the configured
    # default); set lower only as a speed knob on very large cells.
    # The number IS training-set accuracy by design -- real
    # generalization is measured by the next round's stage04b against
    # shapes stage01 history-dedup'd against every prior round.
    # `--val-frac` is the deprecated alias (kept so older configs /
    # invocations don't break).
    ap.add_argument(
        "--eval-sample-frac",
        "--val-frac",
        dest="eval_sample_frac",
        type=float,
        default=1.0,
        help="fraction of each cell's GEMMs scored each "
        "epoch for the per-epoch sel_eff REPORT "
        "(training-set accuracy; reference only). "
        "Default 1.0 = all GEMMs. The student trains "
        "on every GEMM regardless of this value.",
    )
    ap.add_argument(
        "--only-cells",
        default=None,
        help="comma-separated cell labels to train (default: all)",
    )
    ap.add_argument(
        "--skip-cells", default=None, help="comma-separated cell labels to skip"
    )
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=50,
        help="cells with fewer GEMMs are skipped",
    )
    # Origami baseline is always-on: every cell gets a per-cell line
    # reporting Origami's selection efficiency for direct comparison. If
    # origami fails to import or the library dir is missing, training
    # continues with a warning (baseline is skipped, not fatal).
    ap.add_argument(
        "--origami-library-dir",
        default=os.path.join(
            REPO_ROOT, "projects/hipblaslt/build/release", "Tensile/library/gfx950"
        ),
        help="hipblaslt library .dat directory used for the " "Origami baseline.",
    )
    ap.add_argument(
        "--origami-python-path",
        default=os.path.join(REPO_ROOT, "shared/origami/python/src"),
        help="prepended to sys.path so `import origami` resolves.",
    )
    # None == no filter == today's behavior. See lib/dat.py: on gfx1250
    # the MX and non-MX F8 TN libraries decode to the SAME dtype/layout
    # key, so the baseline pool has to be narrowed by scale mode or the
    # two kernel pools merge.
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA/scaleB mode of this run "
        "(0=none, 1=Scalar, 3=Block_32_UE8M0/MX); "
        "filters which .dat libraries feed the Origami "
        "baseline pool and sets the MX block size",
    )
    ap.add_argument(
        "--origami-device",
        type=int,
        default=0,
        help="GPU index passed to origami.get_hardware_for_device().",
    )
    ap.add_argument(
        "--no-origami-baseline",
        action="store_true",
        help="(escape-hatch) disable the Origami per-cell "
        "baseline. Default behaviour is always-on.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Origami's python binding path needs to be on sys.path BEFORE any
    # `lib.features` function that touches `get_hw(...)` runs (the
    # hardware constants are queried live from the binding -- there's
    # no JSON fallback). We do this unconditionally; the per-cell
    # Origami baseline (a separate concern) gates only the heavier
    # `import origami` + `load_origami_configs_by_dtype` step further
    # down.
    if args.origami_python_path and args.origami_python_path not in sys.path:
        sys.path.insert(0, args.origami_python_path)

    req_path = os.path.normpath(
        os.path.join(THIS_DIR, "..", "..", "training_pipeline/requirements.txt")
    )
    if not _TORCH_OK:
        _err("err", f"torch is not importable; run `pip install -r {req_path}`")
        raise SystemExit(1)
    if not _LGB_OK and not args.no_teacher:
        _err(
            "err",
            f"lightgbm is not importable; run `pip install -r "
            f"{req_path}` or pass --no-teacher to skip distillation.",
        )
        raise SystemExit(1)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.quiet:
        _banner("grid-based per-cell distill_v3 trainer", C.HEADER)
        _info(
            "cfg",
            f"enriched_csv_dir : {args.enriched_csv_dir}  "
            f"({len(args.enriched_csv_dir)} dirs unioned)",
        )
        _info("cfg", f"output_dir       : {args.output_dir}")
        _info("cfg", f"arch / device    : {args.arch} / {args.device}")
        _info("cfg", f"epochs           : {args.epochs}")
        _info(
            "cfg",
            f"lr / wd / clip   : {args.lr} / {args.weight_decay} / {args.grad_clip_norm}",
        )
        _info(
            "cfg",
            f"distill lam/T/α  : {args.lam} / {args.T} / {args.alpha}  (tol={args.target_tolerance})",
        )
        _info(
            "cfg",
            f"LightGBM teacher : "
            f"{'ENABLED' if not args.no_teacher else 'DISABLED (--no-teacher)'}  "
            f"n_est={args.n_estimators} leaves={args.num_leaves} lgb_lr={args.lgb_lr}",
        )
        _info(
            "cfg",
            f"eval_sample_frac : {args.eval_sample_frac}  "
            f"(per-epoch reporting subsample; training-set acc)",
        )
        _info("cfg", f"seed             : {args.seed}")

    # Stage 1: feature inventory + cell loading -------------------------------
    if not args.quiet:
        _banner("Stage 1/3  Loading data + feature inventory", C.BLUE)
    q_names, i_names, x_names = _build_feature_lists()
    if not args.quiet:
        _ok(
            "feat",
            f"feature dims: Q={len(q_names)}  I={len(i_names)}  "
            f"X={len(x_names)}  (total = {len(q_names)+len(i_names)+len(x_names)})",
        )

    from lib import subcells as sc

    split_tree = sc.load_cumulative_split_tree(args.prior_round_dir or [])
    # ALSO merge in THIS round's stage04b/splits.json (which the
    # orchestrator passes via --current-round-dir). In an active
    # round, stage04b runs JUST BEFORE stage05 and may have decided
    # new splits whose children labels do not yet exist in any prior
    # round's tree. Without merging this round's splits in, the GEMMs
    # in `round_R/stage04` would still get assigned to the parent
    # label and the new children would appear empty (`to_train=0`).
    if args.current_round_dir is not None:
        current_splits = sc.load_cumulative_split_tree([Path(args.current_round_dir)])
        n_added = 0
        for cell, rule in current_splits.items():
            if cell not in split_tree:
                n_added += 1
            split_tree[cell] = rule
        if n_added and not args.quiet:
            _ok(
                "splits",
                f"merged {n_added} splits from THIS round's "
                f"stage04b/splits.json (so the new child labels see "
                f"their GEMMs)",
            )
    if split_tree and not args.quiet:
        _ok(
            "splits",
            f"loaded cumulative split tree: "
            f"{len(split_tree)} parent cells, "
            f"{sum(len(sc.leaves_under(c, split_tree))-1 for c in split_tree)}"
            f" extra sub-cells in this round",
        )
        for line in sc.format_split_tree(split_tree).splitlines():
            print(f"        {line}", flush=True)

    cells, load_stats = load_enriched_chunks(
        args.enriched_csv_dir, args.quiet, split_tree
    )

    def _split_csv_cells(s: Optional[str]) -> Optional[set]:
        if not s:
            return None
        return {c.strip() for c in s.split(",") if c.strip()}

    only = _split_csv_cells(args.only_cells)
    skip = _split_csv_cells(args.skip_cells) or set()
    cells_to_train: List[str] = []
    cells_skipped: List[Tuple[str, str]] = []
    # Iteration order: base 96-cell order first, then any sub-cells
    # (sorted by depth then label) so output is stable across runs.
    base_order = all_cell_labels()
    base_set = set(base_order)
    subcell_labels = sorted(
        [c for c in cells.keys() if c not in base_set],
        key=lambda c: (sc.split_depth(c), c),
    )
    cell_order = [c for c in base_order if c in cells] + subcell_labels
    for label in cell_order:
        if only is not None and label not in only:
            cells_skipped.append((label, "not-in-only"))
            continue
        if label in skip:
            cells_skipped.append((label, "in-skip"))
            continue
        if len(cells[label]) < args.min_cell_gemms:
            cells_skipped.append((label, f"too few gemms ({len(cells[label])})"))
            continue
        cells_to_train.append(label)

    # Base 96-cell coverage report (sub-cells are an opt-in expansion;
    # the "unpopulated" warning is about the 96-cell baseline so the
    # operator knows which problem regions have no model at all).
    full_grid = all_cell_labels()
    cells_unpopulated = [
        lbl
        for lbl in full_grid
        if lbl not in cells and not any(c.startswith(lbl + "#") for c in cells)
    ]

    def _why_unpopulated(lbl: str) -> str:
        mt, nt, kt, bt = lbl.split("|")
        # Mirrors phase_8_grid_seed.is_rotating512_safe: any batched shape
        # with max(M,N) > 256 OR K > 512 was filtered for bench-tool OOM.
        if bt == "Bany" and ("Large" in (mt, nt) or kt == "LargeK"):
            return "OOB filter is_rotating512_safe: B>1 & (max(M,N)>256 or K>512)"
        return "no shape sampled (not in phase_8 OOB grid)"

    if not args.quiet:
        _ok(
            "cells",
            f"populated={len(cells)}/{len(full_grid)}  "
            f"to_train={len(cells_to_train)}  "
            f"too_small_or_filtered={len(cells_skipped)}  "
            f"unpopulated={len(cells_unpopulated)}  "
            f"(min_cell_gemms={args.min_cell_gemms})",
        )
        for label, why in cells_skipped[:6]:
            _warn("cells", f"  skip {label}: {why}")
        if len(cells_skipped) > 6:
            _info("cells", f"  ... ({len(cells_skipped)-6} more skipped)")
        if cells_unpopulated:
            _warn(
                "cells",
                f"NO MODEL for these {len(cells_unpopulated)} "
                f"untrained cells (inference must fall back, e.g. to "
                f"plain origami):",
            )
            # Group by reason for compact display
            from collections import defaultdict as _dd

            by_reason: Dict[str, List[str]] = _dd(list)
            for lbl in cells_unpopulated:
                by_reason[_why_unpopulated(lbl)].append(lbl)
            for reason in sorted(by_reason):
                lbls = sorted(by_reason[reason])
                _warn("cells", f"  ({len(lbls)} cells) reason: {reason}")
                for lbl in lbls:
                    _warn("cells", f"     - {lbl}")

    # Always-on Origami per-cell baseline: load configs + hardware. If
    # any step fails the rest of training proceeds without baseline (with
    # a clear warning).
    origami_configs: Optional[Dict[Tuple, List[Any]]] = None
    origami_hw: Any = None
    # Elements per block scale for the baseline's problem_t (32 for MXFP8 /
    # --scale-mode 3, 0 for everything else) -- see lib/dat.py.
    from lib.dat import mx_block_size_for_scale_mode as _mx_bs

    _mx_block = _mx_bs(args.scale_mode)
    if not args.no_origami_baseline:
        global origami, _ORIGAMI_OK
        if args.origami_python_path and args.origami_python_path not in sys.path:
            sys.path.insert(0, args.origami_python_path)
        try:
            import origami as _origami_mod  # type: ignore

            origami = _origami_mod
            _ORIGAMI_OK = True
        except ImportError as e:
            _warn("origami", f"import failed ({e}); per-cell baseline disabled")
        if _ORIGAMI_OK:
            try:
                sys.path.insert(0, THIS_DIR)
                # The dat-walker + origami_hardware helpers live in
                # stage04 (port of bench_logs_to_chunks.py). Import them
                # by file path to keep stage05 dependency-free.
                from stage04_convert_to_enriched_dataset import (  # type: ignore
                    load_origami_configs_by_dtype as _load_cfgs,
                    _origami_hardware as _ori_hw,
                )

                # _load_cfgs now returns (configs_by_dtype, dat_info_by_dtype);
                # we only need the configs dict for the baseline.
                _loaded = _load_cfgs(
                    args.origami_library_dir,
                    quiet=args.quiet,
                    scale_mode=args.scale_mode,
                )
                origami_configs = _loaded[0] if isinstance(_loaded, tuple) else _loaded
                # arch= matters only on the FALLBACK path, but it matters a lot
                # there: without it this defaults to "gfx950" (the `arch`
                # default of stage04's _origami_hardware()), so a gfx1250 host
                # whose device probe fails would silently get gfx950 hardware
                # constants for the Origami baseline. stage04b already passes it.
                origami_hw = _ori_hw(
                    args.origami_device, quiet=args.quiet, arch=args.arch
                )
                if origami_hw is None:
                    _warn(
                        "origami",
                        "no hardware descriptor; per-cell " "baseline disabled",
                    )
                    origami_configs = None
                elif not args.quiet:
                    _ok(
                        "origami",
                        f"per-cell baseline ENABLED  "
                        f"(configs by dtype/layout: {len(origami_configs)} keys)",
                    )
            except Exception as e:
                _warn("origami", f"setup failed ({e}); per-cell baseline disabled")
                origami_configs = None
                origami_hw = None

    # Stage 2: per-cell training ----------------------------------------------
    if not args.quiet:
        _banner("Stage 2/3  Per-cell training", C.BLUE)

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as logf:
        log_writer = csv.DictWriter(
            logf,
            fieldnames=[
                "cell",
                "epoch",
                "train_loss",
                "sel_eff",
                "pick_us_geomean",
                "n_eval_sample",
                "n_gemms",
            ],
        )
        log_writer.writeheader()
        all_results: List[Dict[str, Any]] = []
        cells_failed: List[str] = []

        # ── Phase 1 (optional): PARALLEL per-cell training ──────────────
        # With --train-workers > 1, train cells concurrently in a
        # thread-capped `spawn` process pool over the pure `train_cell()`.
        # The enriched data is loaded ONCE in this parent; each worker gets
        # only its cell's `gemms`. `_train_worker_init` + the inherited
        # OMP/MKL/OpenBLAS caps (and LightGBM's pinned num_threads) prevent
        # the thread oversubscription that made a naive multi-job attempt
        # SLOWER than sequential. The cheap origami-dependent post-steps
        # (baseline / smart_K / deployed eval) still run serially in the
        # loop below, reading each cell's result from `train_results`.
        # Seed is keyed to the cell's position in `cells_to_train` so a
        # cell trains identically regardless of worker count.
        train_results: Dict[str, Dict[str, Any]] = {}
        # Cap workers at the number of cells; auto-size threads/worker so
        # workers x threads ~= cpu_count (use the whole machine) unless an
        # explicit --train-threads-per-worker was given.
        _tw = min(int(getattr(args, "train_workers", 1) or 1), len(cells_to_train))
        if _tw > 1 and len(cells_to_train) > 1:
            import concurrent.futures as _cf
            import multiprocessing as _mp

            _cores = os.cpu_count() or 8
            _tpw = int(getattr(args, "train_threads_per_worker", 0) or 0)
            if _tpw <= 0:
                _tpw = max(2, _cores // _tw)
            # Spawned workers inherit these (BLAS reads them once at import).
            for _v in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ[_v] = str(_tpw)
            _ci_of = {lbl: i for i, lbl in enumerate(cells_to_train, 1)}
            # Largest cells first so the longest trains don't strand the tail.
            _order = sorted(cells_to_train, key=lambda c: len(cells[c]), reverse=True)
            if not args.quiet:
                _info(
                    "parallel",
                    f"Phase 1: parallel-training {len(_order)} cells on "
                    f"{_tw} workers x {_tpw} threads "
                    f"(cores={os.cpu_count()})",
                )
            _tpar = time.time()
            _ctx = _mp.get_context("spawn")
            with _cf.ProcessPoolExecutor(
                max_workers=_tw,
                mp_context=_ctx,
                initializer=_train_worker_init,
                initargs=(_tpw,),
            ) as _ex:
                # NOTE: positional args MUST match train_cell's signature
                # (cell, gemms, q_names, i_names, x_names, arch, epochs, lr,
                #  weight_decay, grad_clip_norm, lam, T, alpha,
                #  target_tolerance, n_estimators, num_leaves, lgb_lr, seed,
                #  device, eval_sample_frac, use_teacher, quiet,
                #  training_log_writer).
                _f2l = {
                    _ex.submit(
                        train_cell,
                        label,
                        cells[label],
                        q_names,
                        i_names,
                        x_names,
                        args.arch,
                        args.epochs,
                        args.lr,
                        args.weight_decay,
                        args.grad_clip_norm,
                        args.lam,
                        args.T,
                        args.alpha,
                        args.target_tolerance,
                        args.n_estimators,
                        args.num_leaves,
                        args.lgb_lr,
                        args.seed + 7 * _ci_of[label],
                        args.device,
                        args.eval_sample_frac,
                        (not args.no_teacher),
                        True,
                        None,
                    ): label
                    for label in _order
                }
                _nd = 0
                for _fut in _cf.as_completed(_f2l):
                    _lbl = _f2l[_fut]
                    _nd += 1
                    try:
                        train_results[_lbl] = _fut.result()
                    except Exception as _e:
                        _warn(
                            "parallel",
                            f"{_lbl}: parallel train failed "
                            f"({_e!r}); will retry serially",
                        )
                    if not args.quiet and (_nd % 10 == 0 or _nd == len(_order)):
                        _info(
                            "parallel",
                            f"  {_nd}/{len(_order)} cells trained "
                            f"({time.time() - _tpar:.0f}s)",
                        )
            if not args.quiet:
                _info(
                    "parallel",
                    f"Phase 1 complete: {len(train_results)}/{len(_order)} "
                    f"cells in {time.time() - _tpar:.0f}s; "
                    f"post-steps (origami/smart_K/deployed) run serially",
                )

        for ci, label in enumerate(cells_to_train, 1):
            n_gemms = len(cells[label])
            if not args.quiet:
                print(
                    f"\n{C.BLUE}[cell {ci:>2d}/{len(cells_to_train)}]"
                    f"{C.ENDC} {C.BOLD}{label}{C.ENDC}  "
                    f"n_gemms={_fmt_int(n_gemms)}",
                    flush=True,
                )
            try:
                # Phase 1 may have already trained this cell in parallel.
                # Otherwise (sequential mode, or a worker that errored) train
                # it here serially as a fallback.
                r = train_results.get(label)
                if r is None:
                    r = train_cell(
                        cell=label,
                        gemms=cells[label],
                        q_names=q_names,
                        i_names=i_names,
                        x_names=x_names,
                        arch=args.arch,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        grad_clip_norm=args.grad_clip_norm,
                        lam=args.lam,
                        T=args.T,
                        alpha=args.alpha,
                        target_tolerance=args.target_tolerance,
                        n_estimators=args.n_estimators,
                        num_leaves=args.num_leaves,
                        lgb_lr=args.lgb_lr,
                        seed=args.seed + 7 * ci,
                        device=args.device,
                        eval_sample_frac=args.eval_sample_frac,
                        use_teacher=(not args.no_teacher),
                        quiet=args.quiet,
                        training_log_writer=log_writer,
                    )
                logf.flush()
                if origami_configs is not None and origami_hw is not None:
                    ob = compute_origami_sel_eff(
                        cells[label], origami_configs, origami_hw, _mx_block
                    )
                    r["origami"] = ob
                    # Per-cell smart_K LIST: STUDENT-AWARE greedy
                    # max-coverage on the cell's bench data, with
                    # the trained student in the loop. The student
                    # picks one kernel from K at deploy time, so K
                    # is chosen against the student's ACTUAL pick
                    # (not the oracle's argmin) -- slots aren't
                    # wasted on kernels the student would mis-rank.
                    # Reload best-epoch student so K-selection +
                    # deployed eval see the exact bundle that ships.
                    student_d = GenericTwoTower(
                        q_dim=len(q_names),
                        i_dim=len(i_names),
                        x_dim=len(x_names),
                        embed_dim=r["embed_dim"],
                        hidden_dim=r["hidden_dim"],
                        inter_hidden=r["inter_hidden"],
                    ).to(args.device)
                    student_d.load_state_dict(r["state_dict"])
                    # Adaptive-K is opt-in: only engages when
                    # `--smart-k-max` was given (args.smart_k_max not
                    # None). Otherwise k_max=None reproduces the
                    # legacy fixed-K behavior exactly.
                    sk_result = select_smart_k_signatures(
                        cells[label],
                        k=int(args.smart_k),
                        student=student_d,
                        q_names=q_names,
                        i_names=i_names,
                        x_names=x_names,
                        arch=args.arch,
                        norms=r["feature_norms"],
                        device=args.device,
                        return_metrics=True,
                        k_min=(
                            int(args.smart_k_min)
                            if args.smart_k_max is not None
                            else None
                        ),
                        k_max=(
                            int(args.smart_k_max)
                            if args.smart_k_max is not None
                            else None
                        ),
                        saturation_eps=float(args.smart_k_saturation_eps),
                        oracle_seed=int(getattr(args, "oracle_seed", 0) or 0),
                    )
                    smart_k_sigs = sk_result["sigs"]
                    sk_metrics = sk_result["metrics"]
                    r["smart_k_signatures"] = smart_k_sigs
                    r["smart_k_metrics"] = sk_metrics
                    if not args.quiet:
                        if sk_metrics.get("mode") == "student_aware":
                            sa = sk_metrics["student_aware_sel_eff"]
                            ou = sk_metrics["oracle_sel_eff_under_student"]
                            _ad = (
                                f"  [adaptive: knee@{len(smart_k_sigs)}"
                                f"/{sk_metrics.get('k_build')}]"
                                if sk_metrics.get("adaptive_k")
                                else ""
                            )
                            _info(
                                "smart_k",
                                f"{label}  K={len(smart_k_sigs)} "
                                f"(target={int(args.smart_k)})  "
                                f"student_aware_sel_eff={sa:.4f}  "
                                f"oracle_under_student={ou:.4f}  "
                                f"gain={sa - ou:+.4f}{_ad}",
                            )
                        else:
                            _info(
                                "smart_k",
                                f"{label}  K={len(smart_k_sigs)} "
                                f"(target={int(args.smart_k)})  "
                                f"mode={sk_metrics.get('mode')}",
                            )
                    # DEPLOYED sel_eff: same student picks 1 from
                    # the K-list it co-selected. This number is now
                    # honest -- not an oracle upper-bound.
                    try:
                        dep = evaluate_deployed_sel_eff(
                            student_d,
                            cells[label],
                            q_names,
                            i_names,
                            x_names,
                            args.arch,
                            r["feature_norms"],
                            origami_configs,
                            origami_hw,
                            args.smart_k,
                            args.device,
                            smart_k_signatures=smart_k_sigs,
                        )
                        r["deployed"] = dep
                        # QUALITY-BASED smart_K fallback: if the whitelist
                        # costs too much in-sample vs the full-ranking model
                        # (deployed << model), the signature set structurally
                        # cannot express this cell's winners (confirmed: even
                        # the oracle-over-signatures caps far below model).
                        # Clear the whitelist -> the runtime ranks ALL feasible
                        # candidates for this cell. Both the Python eval and the
                        # C++ runtime treat an empty whitelist as "no filter".
                        _fb = float(getattr(args, "smartk_fallback_delta", 0.0) or 0.0)
                        _mse = float(r.get("best_sel_eff") or 0.0)
                        _dse = float(dep.get("sel_eff") or 0.0)
                        if _fb < 0 and _dse > 0 and _mse > 0 and (_dse - _mse) <= _fb:
                            r["smart_k_signatures"] = []  # -> full ranking
                            try:
                                dep_full = evaluate_deployed_sel_eff(
                                    student_d,
                                    cells[label],
                                    q_names,
                                    i_names,
                                    x_names,
                                    args.arch,
                                    r["feature_norms"],
                                    origami_configs,
                                    origami_hw,
                                    args.smart_k,
                                    args.device,
                                    smart_k_signatures=[],
                                )
                                r["deployed"] = dep_full
                                (r.setdefault("smart_k_metrics", {}))[
                                    "quality_fallback"
                                ] = {
                                    "fired": True,
                                    "k_delta_before": _dse - _mse,
                                    "sel_eff_before": _dse,
                                    "sel_eff_after": dep_full.get("sel_eff"),
                                    "n_sigs_dropped": len(smart_k_sigs),
                                }
                                if not args.quiet:
                                    _info(
                                        "smart_k",
                                        f"{label}  QUALITY-FALLBACK -> full "
                                        f"ranking (k_delta={_dse-_mse:+.4f} "
                                        f"<= {_fb}); deployed "
                                        f"{_dse*100:.2f}% -> "
                                        f"{(dep_full.get('sel_eff') or 0)*100:.2f}%",
                                    )
                            except Exception as e2:
                                _warn(
                                    "deploy", f"{label}: fallback re-eval failed ({e2})"
                                )
                    except Exception as e:
                        _warn("deploy", f"{label}: deployed eval failed ({e})")
                        r["deployed"] = {
                            "sel_eff": float("nan"),
                            "pick_us_geomean": float("nan"),
                            "winner_us_geomean": float("nan"),
                            "n_eval": 0,
                        }
                all_results.append(r)
                if not args.quiet:
                    win_g = r.get("winner_us_geomean", float("nan"))
                    best_pick = r.get("best_pick_us_geomean", float("nan"))
                    ob = r.get("origami") or {}
                    dep = r.get("deployed") or {}

                    # Compact per-cell line: sel_eff% (latency_us) for
                    # MODEL / ORIGAMI / DEPLOYED, plus the winner-us
                    # geomean baseline. NaN-safe formatting.
                    def _fmt_se(d: Dict[str, Any]) -> str:
                        if (
                            not d
                            or d.get("n_eval", 0) <= 0
                            or math.isnan(d.get("sel_eff", float("nan")))
                        ):
                            return "  n/a"
                        return (
                            f"{d['sel_eff']*100:6.2f}% "
                            f"({d['pick_us_geomean']:7.2f}us)"
                        )

                    model_str = (
                        f"  model={best_sel_eff_pct(r):6.2f}% " f"({best_pick:7.2f}us)"
                    )
                    _ok(
                        "cell",
                        f"{label}  best_ep={r['best_epoch']}  "
                        f"winner_us={win_g:7.2f}us"
                        f"{model_str}"
                        f"  origami={_fmt_se(ob)}"
                        f"  deployed={_fmt_se(dep)}"
                        f"  ({_fmt_dur(r['elapsed_s'])})",
                    )
            except Exception as e:
                import traceback

                _err("cell", f"{label} FAILED: {e}")
                _err("cell", traceback.format_exc().rstrip())
                cells_failed.append(label)
                continue

    # Stage 3: aggregate + save -----------------------------------------------
    if not args.quiet:
        _banner("Stage 3/3  Aggregate + save", C.BLUE)

    # Per-cell summary table, worst deployed sel_eff first, so the operator
    # can scan every cell at a glance. The cell-name column is sized to the
    # longest label this round so the numeric columns stay aligned no matter
    # how deep the adaptive splits made the labels.
    if not args.quiet and all_results:

        def _se_pct(d: Optional[Dict[str, Any]]) -> float:
            v = (d or {}).get("sel_eff", float("nan"))
            return v * 100.0 if v is not None else float("nan")

        def _pctcol(x: float) -> str:
            return "    n/a" if math.isnan(x) else f"{x:7.2f}"

        _w = min(max(max((len(r["cell"]) for r in all_results), default=24), 24), 80)
        _rows = sorted(
            all_results,
            key=lambda r: (
                math.isnan(_se_pct(r.get("deployed"))),
                _se_pct(r.get("deployed")),
            ),
        )
        _hdr = (
            f"  {'cell':<{_w}}  {'n_gemms':>8}  {'model%':>7}  "
            f"{'orig%':>7}  {'deploy%':>7}  {'K':>3}  "
            f"{'win_us':>9}  {'train':>7}"
        )
        print(_hdr, flush=True)
        print("  " + "-" * (len(_hdr) - 2), flush=True)
        for r in _rows:
            _k = len(r.get("smart_k_signatures") or [])
            _win = r.get("winner_us_geomean", float("nan"))
            _win_s = (
                "      n/a" if (_win is None or math.isnan(_win)) else f"{_win:9.2f}"
            )
            print(
                f"  {r['cell']:<{_w}}  {_fmt_int(r['n_gemms']):>8}  "
                f"{_pctcol(best_sel_eff_pct(r))}  "
                f"{_pctcol(_se_pct(r.get('origami')))}  "
                f"{_pctcol(_se_pct(r.get('deployed')))}  "
                f"{_k:>3}  {_win_s}  "
                f"{_fmt_dur(r.get('elapsed_s', 0) or 0):>7}",
                flush=True,
            )
        print("", flush=True)

    # Globals are computed as the n-weighted log-mean of each cell's
    # per-method geomean (sel_eff AND the corresponding pick_us / winner_us
    # geomeans). This is equivalent to a flat geomean across every
    # evaluated GEMM if you assume per-cell evaluators all touch all
    # their GEMMs; with sub-sampling it's the closest honest aggregate.
    def _wlog_mean(pairs: List[Tuple[float, int]]) -> float:
        """pairs: list of (value, weight). Returns exp(weighted mean of
        log(value)) over positive (value, weight)."""
        ls = 0.0
        ws = 0.0
        for v, w in pairs:
            if v is None or (isinstance(v, float) and (math.isnan(v) or v <= 0)):
                continue
            if w <= 0:
                continue
            ls += math.log(v) * w
            ws += w
        return math.exp(ls / ws) if ws > 0 else float("nan")

    # Model (best epoch).
    model_pairs_se = [
        (r.get("best_sel_eff", float("nan")), r["n_gemms"]) for r in all_results
    ]
    model_pairs_pick = [
        (r.get("best_pick_us_geomean", float("nan")), r["n_gemms"]) for r in all_results
    ]
    win_pairs = [
        (r.get("winner_us_geomean", float("nan")), r["n_gemms"]) for r in all_results
    ]
    global_sel_eff = _wlog_mean(model_pairs_se)
    global_model_pick_us = _wlog_mean(model_pairs_pick)
    global_winner_us = _wlog_mean(win_pairs)

    # Origami baseline.
    ori_pairs_se = [
        (((r.get("origami") or {}).get("sel_eff")), r["n_gemms"]) for r in all_results
    ]
    ori_pairs_pick = [
        (((r.get("origami") or {}).get("pick_us_geomean")), r["n_gemms"])
        for r in all_results
    ]
    global_origami_sel_eff = _wlog_mean(ori_pairs_se)
    global_origami_pick_us = _wlog_mean(ori_pairs_pick)

    # Deployed (smart-K + model).
    dep_pairs_se = [
        (((r.get("deployed") or {}).get("sel_eff")), r["n_gemms"]) for r in all_results
    ]
    dep_pairs_pick = [
        (((r.get("deployed") or {}).get("pick_us_geomean")), r["n_gemms"])
        for r in all_results
    ]
    global_deployed_sel_eff = _wlog_mean(dep_pairs_se)
    global_deployed_pick_us = _wlog_mean(dep_pairs_pick)

    def _cell_entry(r: Dict[str, Any]) -> Dict[str, Any]:
        lbl = r["cell"]

        # `r["deployed"]` carries `per_gemm_log_ratio` (np.ndarray, one
        # entry per GEMM in this cell). It's used INSIDE this process
        # (and by stage04b's CART split criterion, which calls
        # evaluate_deployed_sel_eff itself), so it has no business in
        # cells.json -- it isn't JSON-serializable and would bloat the
        # summary file with kilo-row arrays for no consumer. Strip it
        # (and any other ndarray-valued fields just in case) before
        # serializing.
        def _json_scalars(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not isinstance(d, dict):
                return d
            return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

        # Smart_K kernel signatures (8-tuples). The deploy runtime
        # reads this list per cell from the .bin -- it's the
        # candidate set the MLP gets to score at inference.
        sk_sigs = r.get("smart_k_signatures") or []
        sk_serial = [list(int(v) for v in s) for s in sk_sigs]

        # JSON-friendly view of the K-selection metrics (regret
        # curve etc.) for offline analysis. The regret_curve is a
        # list of (k_size, sel_eff, n_eval) tuples; serialize as
        # list-of-lists.
        def _metrics_to_json(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not isinstance(m, dict):
                return None
            out_m = {}
            for kk, vv in m.items():
                if kk.endswith("regret_curve") and isinstance(vv, list):
                    out_m[kk] = [
                        [
                            int(p[0]),
                            (float(p[1]) if isinstance(p[1], float) else p[1]),
                            int(p[2]),
                        ]
                        for p in vv
                    ]
                else:
                    out_m[kk] = vv
            return out_m

        entry = {
            "label": lbl,
            "n_gemms": r["n_gemms"],
            "n_pairs": r["n_pairs"],
            "best_epoch": r["best_epoch"],
            "best_sel_eff": r["best_sel_eff"],
            "best_pick_us_geomean": r.get("best_pick_us_geomean"),
            "winner_us_geomean": r.get("winner_us_geomean"),
            "embed_dim": r["embed_dim"],
            "hidden_dim": r["hidden_dim"],
            "inter_hidden": r["inter_hidden"],
            "elapsed_s": r["elapsed_s"],
            "origami": _json_scalars(r.get("origami")),
            "deployed": _json_scalars(r.get("deployed")),
            "smart_k_signatures": sk_serial,
            "smart_k_metrics": _metrics_to_json(r.get("smart_k_metrics")),
        }
        if sc.is_subcell(lbl):
            entry["base_cell"] = sc.base_cell(lbl)
            entry["split_depth"] = sc.split_depth(lbl)
        return entry

    cells_json = {
        "schema_version": "pipeline-v3-2026-05-27",
        "feature_version": fs.FEATURES_VERSION,
        "feature_names_hash": fs.feature_names_hash(),
        "q_names": q_names,
        "i_names": i_names,
        "x_names": x_names,
        "n_cells_populated": len(cells),
        "n_cells_trained": len(all_results),
        "n_subcells_trained": sum(1 for r in all_results if sc.is_subcell(r["cell"])),
        "cells": [_cell_entry(r) for r in all_results],
    }
    with open(os.path.join(args.output_dir, "cells.json"), "w") as f:
        json.dump(cells_json, f, indent=2)

    # Save models.pt.
    # `models` is the dict of {cell_label: model_entry}. In ACTIVE
    # mode (--only-cells set), we MERGE the just-trained models into
    # the PRIOR round's models.pt so cells that didn't fail this
    # round (PASS cells, plus already-trained sub-cells from earlier
    # splits) keep their models -- without this merge, every round
    # would drop the un-retrained cells and the next round's stage04b
    # would mark them as `no_prior_model`, leaving the cumulative split
    # tree stuck at depth 1.
    # Cells that were SPLIT by the orchestrator (i.e., no longer
    # leaves of the cumulative split tree) are EXCLUDED from carry-
    # forward; their two child labels are trained fresh this round.
    new_models: Dict[str, Any] = {}
    for r in all_results:
        new_models[r["cell"]] = {
            "state_dict": r["state_dict"],
            "feature_norms": r["feature_norms"],
            "embed_dim": r["embed_dim"],
            "hidden_dim": r["hidden_dim"],
            "inter_hidden": r["inter_hidden"],
            "best_epoch": r["best_epoch"],
            # The K-list IS part of the model artifact -- the
            # deploy runtime reads it per cell from the .bin and
            # restricts MLP scoring to these K kernels.
            "smart_k_signatures": r.get("smart_k_signatures") or [],
        }

    # Carry forward un-retrained models from the most recent prior
    # round's models.pt, if any. The "most recent prior" is the LAST
    # entry in --prior-round-dir; the orchestrator passes them in
    # chronological order.
    carried_n = 0
    if args.prior_round_dir:
        prior_models_pt = Path(args.prior_round_dir[-1]) / "stage05" / "models.pt"
        if prior_models_pt.exists():
            try:
                prior_bundle = torch.load(
                    str(prior_models_pt), map_location="cpu", weights_only=False
                )
            except Exception as e:
                _warn(
                    "carry",
                    f"failed to load prior models.pt "
                    f"({prior_models_pt}): {e}; skipping carry-forward",
                )
                prior_bundle = None
            if prior_bundle is not None:
                # A cell is "still a leaf" iff it's not a split parent
                # in the cumulative tree (split parents have been
                # replaced by their children).
                still_leaf = lambda c: c not in split_tree
                for cell_label, model_entry in (
                    prior_bundle.get("models") or {}
                ).items():
                    if cell_label in new_models:
                        continue  # this round overrides
                    if not still_leaf(cell_label):
                        continue  # parent of a split
                    new_models[cell_label] = model_entry
                    carried_n += 1
                if not args.quiet and carried_n > 0:
                    _ok(
                        "carry",
                        f"carried {carried_n} un-retrained models "
                        f"forward from {prior_models_pt}",
                    )

    torch.save(
        {
            "schema_version": cells_json["schema_version"],
            "feature_version": fs.FEATURES_VERSION,
            "q_dim": len(q_names),
            "i_dim": len(i_names),
            "x_dim": len(x_names),
            "q_names": q_names,
            "i_names": i_names,
            "x_names": x_names,
            "models": new_models,
            # Metadata: how many models this bundle holds + provenance.
            "n_models_trained_this_round": len(all_results),
            "n_models_carried_forward": carried_n,
        },
        os.path.join(args.output_dir, "models.pt"),
    )

    def _per_cell_method_dict(get) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in all_results:
            try:
                v = get(r)
            except Exception:
                v = None
            out[r["cell"]] = v
        return out

    metrics = {
        # Global sel_eff (n_gemms-weighted geomean over cells) for every
        # method we evaluate. winner_us_geomean is the empirical-best
        # latency baseline; for context the model / origami / deployed
        # pick_us_geomeans show what each picker's geomean latency would
        # have been.
        "global_sel_eff": global_sel_eff,
        "global_origami_sel_eff": global_origami_sel_eff,
        "global_deployed_sel_eff": global_deployed_sel_eff,
        "global_winner_us_geomean": global_winner_us,
        "global_model_pick_us_geomean": global_model_pick_us,
        "global_origami_pick_us_geomean": global_origami_pick_us,
        "global_deployed_pick_us_geomean": global_deployed_pick_us,
        "n_cells_trained": len(all_results),
        "n_gemms_trained": sum(r["n_gemms"] for r in all_results),
        "load_stats": load_stats,
        # PosixPath instances from argparse (`type=Path`) aren't
        # JSON-serializable; coerce the whole namespace recursively.
        "args": _jsonable(vars(args)),
        "feature_version": fs.FEATURES_VERSION,
        "elapsed_total_s": time.time() - t0,
        # Per-cell drilldown.
        "per_cell_sel_eff": _per_cell_method_dict(lambda r: r["best_sel_eff"]),
        "per_cell_origami_sel_eff": _per_cell_method_dict(
            lambda r: (
                (r["origami"] or {}).get("sel_eff")
                if (r.get("origami") and r["origami"].get("n_eval", 0) > 0)
                else None
            )
        ),
        "per_cell_deployed_sel_eff": _per_cell_method_dict(
            lambda r: (
                (r["deployed"] or {}).get("sel_eff")
                if (r.get("deployed") and r["deployed"].get("n_eval", 0) > 0)
                else None
            )
        ),
        "per_cell_winner_us_geomean": _per_cell_method_dict(
            lambda r: r.get("winner_us_geomean")
        ),
        "per_cell_model_pick_us_geomean": _per_cell_method_dict(
            lambda r: r.get("best_pick_us_geomean")
        ),
        "per_cell_origami_pick_us_geomean": _per_cell_method_dict(
            lambda r: (
                (r["origami"] or {}).get("pick_us_geomean")
                if (r.get("origami") and r["origami"].get("n_eval", 0) > 0)
                else None
            )
        ),
        "per_cell_deployed_pick_us_geomean": _per_cell_method_dict(
            lambda r: (
                (r["deployed"] or {}).get("pick_us_geomean")
                if (r.get("deployed") and r["deployed"].get("n_eval", 0) > 0)
                else None
            )
        ),
        "smart_k": args.smart_k,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if not args.quiet:
        _banner("Done", C.GREEN)

        # Fixed-width method block: sel_eff% (latency_us).
        def _fmt_global(name: str, se: float, pick: float) -> str:
            color = C.BOLD
            if math.isnan(se):
                return f"  {color}{name:32s}:{C.ENDC} n/a"
            return (
                f"  {color}{name:32s}:{C.ENDC} "
                f"{se*100:6.2f}%  pick_us_geomean={pick:8.2f}us"
            )

        print(
            f"  {C.BOLD}{'winner_us_geomean':32s}:{C.ENDC} "
            f"{global_winner_us:8.2f}us  (empirical-best baseline)",
            flush=True,
        )
        print(
            _fmt_global("global_sel_eff (model)", global_sel_eff, global_model_pick_us),
            flush=True,
        )
        print(
            _fmt_global(
                "global_origami_sel_eff", global_origami_sel_eff, global_origami_pick_us
            ),
            flush=True,
        )
        print(
            _fmt_global(
                "global_deployed_sel_eff",
                global_deployed_sel_eff,
                global_deployed_pick_us,
            ),
            flush=True,
        )
        print(
            f"  {C.BOLD}{'n_cells_trained':32s}:{C.ENDC} " f"{len(all_results)}",
            flush=True,
        )
        print(
            f"  {C.BOLD}{'n_gemms_trained':32s}:{C.ENDC} "
            f"{_fmt_int(metrics['n_gemms_trained'])}",
            flush=True,
        )
        if cells_failed:
            print(
                f"  {C.RED}{'n_cells_failed':32s}:{C.ENDC} "
                f"{len(cells_failed)}  "
                f"({', '.join(cells_failed[:5])}"
                f"{'...' if len(cells_failed) > 5 else ''})",
                flush=True,
            )
        print(
            f"  {C.BOLD}{'elapsed':32s}:{C.ENDC} "
            f"{_fmt_dur(metrics['elapsed_total_s'])}",
            flush=True,
        )
        print(
            f"  {C.BOLD}{'outputs':32s}:{C.ENDC} {args.output_dir}/  "
            f"({{cells.json, models.pt, training_log.csv, "
            f"metrics.json}})",
            flush=True,
        )


if __name__ == "__main__":
    main()
