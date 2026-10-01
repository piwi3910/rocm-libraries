# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage04b_iterate_active_learning -- VALIDATE the prior round's
per-cell models against THIS round's NEW unknown data, then decide
which cells need {pass / retrain / split}.

Conceptually "stage 4.5" (hence the `b` suffix on `04`): it sits
BETWEEN data collection (stage04 enrich) and training (stage05
retrain) -- the active-learning loop validates the PRIOR round's
model on this round's new data, decides which cells need work, THEN
retrains.

This stage ONLY runs in active rounds (round R >= 1). In round 0
there is no prior model to validate and no held-out data
(everything is training data), so the orchestrator skips stage04b
entirely; round 0 ends after stage05 produces the initial
models.pt.

Active-round sequence (where stage04b fits):

    round_R/
      stage01    # gen NEW shapes (dedup vs every prior round)
      stage02    # probe
      stage03    # offline tuning (bench)
      stage04    # enrich
   -> stage04b   # YOU ARE HERE: validate prior model on new data
                 #   - reads round_(R-1)/stage05/models.pt
                 #   - reads round_R/stage04 enriched chunks
                 #   - emits retrain_cells.txt + splits.json
      stage05    # retrain ONLY the cells stage04b marked failing
                 #   (cross-round data augmentation; fresh weights)

The `sel_eff` stage04b reports is the DEPLOYED selection
efficiency: for each problem, Origami ranks every compatible
kernel and we keep its top `smart_k`; the student picks top-1
among those K (i.e. exactly the inference path hipBLASLt uses
at runtime). So the number is a faithful estimate of the
deployed selector's generalization on unknown shapes -- NOT a
more permissive "ML picks from every benched candidate" number.

What stage04b does:

  1. Load the prior round's `<prior>/stage05/models.pt` + `cells.json`.
  2. Load this round's `<round>/stage04` enriched chunks.
  3. For every cell that has BOTH a prior model AND new GEMMs,
     reconstruct the GenericTwoTower student, score every candidate,
     take the top-1 pick per GEMM, and compute the cell's geomean
     sel_eff = winner_us / pick_us on the new shapes.
  4. Also compute the plain-Origami sel_eff on the same GEMMs for
     comparison.
     5. For each cell, decide from this round's `sel_eff_new` (the
     unknown-dataset measurement) PLUS prior-round history:
        - sel_eff_new >= --sel-eff-threshold              -> "pass"
        - sel_eff_new <  --sel-eff-threshold  AND
          prior_retrain_attempts >= --split-after-attempts  AND
          (sel_eff_new did NOT improve over the IMPROVEMENT BASELINE
           by >= --split-min-improvement  OR  sel_eff_new <
           --split-floor)                                 -> "split"
        - everything else                                 -> "retrain"
        - no prior model OR < --min-cell-gemms new GEMMs  -> "retrain"
     The whole below-threshold band is splittable (NOT just below
     split_floor): with no Origami fallback, every cell must be driven
     to >= threshold by augmentation or carving. SPLIT still waits for
     augmentation to run out of road (attempts + not-improving);
     split_floor is now the "hopeless -> carve without waiting on the
     improvement check" fast-path.
     The IMPROVEMENT BASELINE has two sources, in preference order:
        (i)  if --prior-models points at round_(R-2)'s models.pt,
             we evaluate THAT bundle's model for this cell on THIS
             round's GEMMs -- "sel_eff_prior_model_same_data". Same
             unknown dataset, prior model state; the delta isolates
             the model-change effect cleanly.
        (ii) fallback: prior_sel_eff_new from --prior-decisions.
             ALSO measured on unknown data, just on a DIFFERENT
             unknown dataset (the prior round's new shapes), so
             the delta carries an extra data-shift noise term.
     The attempts counter is the cumulative count of "retrain"
     decisions across all prior rounds for the cell.
  6. Write three outputs:
        decisions.json   -- full per-cell metadata
        retrain_cells.txt-- comma-separated retrain list (for stage05
                            --only-cells)
        splits.json      -- {cell: SplitRule} for every cell with
                            decision == "split"; read by stage01 of
                            round R+1 (sub-cell shape synthesis) and
                            stage05 of round R+1 (sub-cell training).

Example (standalone):

    python stage04b_iterate_active_learning.py \\
        --models-dir       <run>/round_0/stage05 \\
        --enriched-csv-dir <run>/round_1/stage04 \\
        --out-dir          <run>/round_1/stage04b \\
        --sel-eff-threshold 0.95 \\
        --round-idx 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
ML_RECOMMENDER_ROOT = THIS_DIR.parent
# training_pipeline/ lives at <rocm-libraries>/shared/tilewright/training_pipeline.
REPO_ROOT = ML_RECOMMENDER_ROOT.parents[2]
if str(ML_RECOMMENDER_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_RECOMMENDER_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# Reuse stage05_train helpers wholesale -- they're already battle-tested
# for the loader, feature builder, and the GenericTwoTower student
# definition.
import stage05_train as _train  # noqa: E402

# hipblaslt scaleA mode -> origami `*_mx_block_size` (32 for MXFP8).
from lib.dat import (  # noqa: E402
    mx_block_size_for_scale_mode as _mx_block_size_for_scale_mode,
)


# ── UI ────────────────────────────────────────────────────────────────────────


class _C:
    _tty = sys.stdout.isatty()
    H = "\033[95m" if _tty else ""
    B = "\033[94m" if _tty else ""
    G = "\033[92m" if _tty else ""
    Y = "\033[93m" if _tty else ""
    R = "\033[91m" if _tty else ""
    GR = "\033[90m" if _tty else ""
    E = "\033[0m" if _tty else ""
    BD = "\033[1m" if _tty else ""


def _banner(title: str, color: str = _C.H) -> None:
    print(f"\n{color}{'='*80}{_C.E}", flush=True)
    print(f"{color}{_C.BD}  {title}{_C.E}", flush=True)
    print(f"{color}{'='*80}{_C.E}", flush=True)


def _ok(tag, msg):
    print(f"{_C.G}[{tag}]{_C.E} {msg}", flush=True)


def _info(tag, msg):
    print(f"{_C.B}[{tag}]{_C.E} {msg}", flush=True)


def _warn(tag, msg):
    print(f"{_C.Y}[{tag}]{_C.E} {msg}", flush=True)


def _err(tag, msg):
    print(f"{_C.R}[{tag}]{_C.E} {msg}", flush=True)


# ── per-cell evaluation ───────────────────────────────────────────────────────


def _eval_cell_with_student(
    student: Any,
    q_t: torch.Tensor,
    i_t: torch.Tensor,
    x_t: torch.Tensor,
    gemm_idx_t: torch.Tensor,
    us_t: torch.Tensor,
    winner_us_t: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """Runs the prior-round student on ALL of this round's GEMMs (no
    sampling -- this is held-out validation). Returns a dict with:
      sel_eff           geomean(winner_us / pick_us)  in (0, 1]
      n_eval            # of GEMMs that contributed
      per_gemm_log_ratio  np.ndarray of shape (n_gemms,), one float per
                          gemm (NaN for gemms with no valid pick); the
                          ratio is log(pick_us / winner_us). Used by
                          stage04b's split-decision CART criterion.

    Tie-break on top-1: lowest pair index wins (deterministic)."""
    n_gemms = q_t.shape[0]
    n_pairs = gemm_idx_t.shape[0]
    student.eval()
    with torch.no_grad():
        q_per_pair = q_t.index_select(0, gemm_idx_t)
        scores = student.score_pairs(q_per_pair, i_t, x_t)
        group_max = torch.full(
            (n_gemms,), float("-inf"), device=device, dtype=scores.dtype
        )
        group_max.scatter_reduce_(
            0, gemm_idx_t, scores, reduce="amax", include_self=False
        )
        is_pick = scores == group_max.index_select(0, gemm_idx_t)
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
        has_pick = (winner_pair_idx < sentinel) & ~torch.isnan(winner_us_per_gemm)
        pick_us_all = us_t[winner_pair_idx.clamp(max=n_pairs - 1)]
        log_ratios_all = torch.full(
            (n_gemms,), float("nan"), device=device, dtype=us_t.dtype
        )
        log_ratios_all[has_pick] = torch.log(
            pick_us_all[has_pick].clamp(min=1e-9)
            / winner_us_per_gemm[has_pick].clamp(min=1e-9)
        )
    valid = log_ratios_all[~torch.isnan(log_ratios_all)]
    n = int(valid.numel())
    if n == 0:
        return {
            "sel_eff": float("nan"),
            "n_eval": 0,
            "per_gemm_log_ratio": np.full(n_gemms, float("nan"), dtype=np.float64),
        }
    sel_eff = 1.0 / max(float(torch.exp(valid.mean()).item()), 1e-9)
    return {
        "sel_eff": sel_eff,
        "n_eval": n,
        "per_gemm_log_ratio": log_ratios_all.detach().cpu().numpy(),
    }


def _resolve_threshold(
    spec: str, origami_sel_eff: Optional[float], fallback: float
) -> float:
    """A threshold spec can be a numeric string ("0.95") or the literal
    "origami" (use this cell's measured Origami sel_eff). When the
    spec is "origami" but the cell has no Origami baseline this round,
    fall back to `fallback`."""
    s = str(spec).strip().lower()
    if s == "origami":
        if origami_sel_eff is None or math.isnan(origami_sel_eff):
            return fallback
        return float(origami_sel_eff)
    try:
        return float(s)
    except (TypeError, ValueError):
        return fallback


def _whiten_with_norms(
    q: np.ndarray, i: np.ndarray, x: np.ndarray, norms: Dict[str, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply train-time per-feature whitening (mean/std) saved by train.py."""
    q_mu = np.asarray(norms["q_mean"], dtype=np.float32)
    q_sd = np.asarray(norms["q_std"], dtype=np.float32)
    i_mu = np.asarray(norms["i_mean"], dtype=np.float32)
    i_sd = np.asarray(norms["i_std"], dtype=np.float32)
    x_mu = np.asarray(norms["x_mean"], dtype=np.float32)
    x_sd = np.asarray(norms["x_std"], dtype=np.float32)
    return (
        ((q - q_mu) / np.where(q_sd < 1e-6, 1.0, q_sd)).astype(np.float32),
        ((i - i_mu) / np.where(i_sd < 1e-6, 1.0, i_sd)).astype(np.float32),
        ((x - x_mu) / np.where(x_sd < 1e-6, 1.0, x_sd)).astype(np.float32),
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="pipeline_v3 stage04b: validate the prior round's "
        "trained per-cell models on this round's NEW enriched "
        "data and emit per-cell pass/retrain decisions."
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="prior round's stage05 output dir "
        "(contains models.pt + cells.json + metrics.json).",
    )
    ap.add_argument(
        "--enriched-csv-dir",
        type=Path,
        required=True,
        help="folder of this round's chunk_*.csv files " "(stage04 output).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="where to write decisions.json + validation.log.",
    )
    ap.add_argument(
        "--sel-eff-threshold",
        default="0.95",
        help="cells whose sel_eff_new >= this PASS. Accepts "
        "a float in (0, 1] OR the literal string "
        '"origami", in which case the per-cell '
        "PASS threshold = the cell's measured Origami "
        "sel_eff on this round's data (\"the model "
        'must do at least as well as plain Origami").',
    )
    ap.add_argument(
        "--split-floor",
        default="0.85",
        help="cells whose sel_eff_new < this are candidates "
        "for SPLIT (subject to --split-after-attempts). "
        'Accepts a float OR "origami" (per-cell '
        "floor = origami_sel_eff_new). 0 disables the "
        "below-floor trigger.",
    )
    ap.add_argument(
        "--split-min-improvement",
        type=float,
        default=0.02,
        help="A below-floor cell is SPLIT only when its "
        "sel_eff_new did NOT improve over the prior "
        "round's sel_eff by at least this delta -- "
        "i.e. 'data augmentation has stopped helping, "
        "carve the cell'. Combined with "
        "--split-after-attempts (which guarantees N "
        "rounds of augmentation are tried first), "
        "this prevents the orchestrator from "
        "carving up the grid every round just "
        "because a cell is below floor. Set to 0 "
        "to disable the not-improving check (then "
        "SPLIT triggers purely on attempts + floor).",
    )
    ap.add_argument(
        "--split-after-attempts",
        type=int,
        default=2,
        help="how many times a cell must have been "
        "RETRAINED before it's eligible to SPLIT. "
        "Default 2: the cell gets two more rounds of "
        "data augmentation; only if it still trips a "
        "split trigger after the 2nd retrain is it "
        "carved in two. Set 0 to split immediately.",
    )
    ap.add_argument(
        "--max-split-depth",
        type=int,
        default=6,
        help="hard cap on the split-tree depth of any "
        "single base 96-grid cell. A cell at this "
        "depth (e.g. 6 = 64 leaves under one base "
        "cell) is never split further regardless of "
        "the CART gain.",
    )
    ap.add_argument(
        "--structural-split-min-octaves",
        type=float,
        default=0.0,
        help="STRUCTURAL width-split trigger (0 = OFF). When "
        ">0, any cell whose widest splittable axis "
        "(Large M/N or LargeK) still spans more than "
        "this many octaves (log2 magnitude range, "
        "tightened by prior splits) is carved REGARDLESS "
        "of its synthetic sel_eff / pass status / "
        "attempts. This is a robustness guarantee: a leaf "
        "the deploy runtime routes a ~10-octave range to "
        "can mis-PASS the bar on same-distribution "
        "synthetic shapes yet fail on real OOD data; "
        "magnitude carving fixes that independent of the "
        "(optimistic) proxy signal. Stops once leaves are "
        "below the cap; bounded by --max-split-depth.",
    )
    ap.add_argument(
        "--structural-split-max-depth",
        type=int,
        default=-1,
        help="cap (split-tree depth) on how deep the STRUCTURAL "
        "width-split (above) may carve a base cell, "
        "INDEPENDENT of --max-split-depth (which still "
        "bounds CART/sel_eff splits). -1 (default) = use "
        "--max-split-depth (no extra cap). Set lower (e.g. "
        "2-3) to stop over-carving Large/LargeK tiers into "
        "thin, data-starved leaves that then get tiny "
        "per-cell models and overfit on the few benched "
        "large GEMMs.",
    )
    ap.add_argument(
        "--smart-k",
        type=int,
        default=10,
        help="ALWAYS-ON candidate cap after the "
        "check_lds_capacity + is_kernel_feasible "
        "pre-filter. Walks survivors in INPUT "
        "ORDER and keeps the first K. NO Origami "
        "analytical latency is consulted. Match "
        "this to the deploy runtime's ML_SMART_K "
        "(default 10).",
    )
    ap.add_argument(
        "--prior-decisions",
        type=Path,
        action="append",
        default=None,
        help="repeatable. Each entry MUST be a "
        "`<round>/stage04b/decisions.json` from an "
        "earlier round. Used to count how many times "
        "each cell has already been retrained, which "
        "gates --split-after-attempts (i.e. give a "
        "cell N rounds of data augmentation before "
        "carving it). The per-cell prior sel_eff is "
        "ALSO read and saved in decisions.json for "
        "history.",
    )
    ap.add_argument(
        "--prior-round-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable; each entry is a `<run>/round_<i>/` "
        "dir. Used to load the cumulative split tree "
        "(round_*/stage04b/splits.json) so this round's "
        "NEW gemms are routed to the SAME leaf sub-cells "
        "the bundle was trained on. If omitted, the tree "
        "is auto-discovered from the run dir inferred "
        "from --models-dir.",
    )
    ap.add_argument(
        "--prior-models",
        type=Path,
        default=None,
        help="path to the round_(R-2) stage05/models.pt "
        "(i.e. the model state immediately BEFORE "
        "the round_(R-1) retraining that produced "
        "--models-dir). When provided, the "
        "'is augmentation helping?' check in the "
        "SPLIT trigger compares THIS round's "
        "sel_eff against the PRIOR model's sel_eff "
        "ON THE SAME UNKNOWN GEMMs -- so the delta "
        "isolates the model-change effect from the "
        "(noisy) data-distribution shift between "
        "rounds. Falls back to the cross-round "
        "comparison (prior_sel_eff_new from "
        "--prior-decisions) when this bundle isn't "
        "available or doesn't carry the cell.",
    )
    ap.add_argument(
        "--round-idx",
        type=int,
        default=0,
        help="round index this stage04b belongs to (used in " "splits.json metadata).",
    )
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--device", default="cpu", help="torch device (cpu / cuda / hip).")
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=20,
        help="if a cell has fewer than this many new GEMMs we "
        "skip validation and conservatively mark it retrain.",
    )
    ap.add_argument(
        "--with-origami-baseline",
        action="store_true",
        default=True,
        help="compute Origami's sel_eff on the same GEMMs "
        "for comparison (always on by default).",
    )
    ap.add_argument(
        "--origami-library-dir",
        default=str(
            REPO_ROOT / "projects/hipblaslt/build/release" "/Tensile/library/gfx950"
        ),
    )
    ap.add_argument(
        "--origami-python-path", default=str(REPO_ROOT / "shared/origami/python/src")
    )
    ap.add_argument("--origami-device", type=int, default=0)
    # None == no filter == today's behavior; see lib/dat.py for why the
    # gfx1250 MX / non-MX F8 libraries need this to stay separate pools.
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA/scaleB mode of this run "
        "(0=none, 1=Scalar, 3=Block_32_UE8M0/MX); filters "
        "the .dat libraries feeding the Origami baseline",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--final-round",
        action="store_true",
        help="On the FINAL round there is no subsequent round to "
        "synthesize + TRAIN a split's children, so a split "
        "here would leave untrained leaves in the deployed "
        "split tree (orphaned GEMM regions with no matching "
        "trained cell). With this flag, SPLIT decisions are "
        "demoted to RETRAIN so every deployed leaf stays a "
        "trained cell.",
    )
    args = ap.parse_args()

    if not _TORCH_OK:
        _err(
            "err",
            "torch is not importable; cannot validate. Run "
            "`pip install -r requirements.txt`.",
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.quiet:
        _banner("Per-cell model validator", _C.H)
        _info("cfg", f"models_dir       : {args.models_dir}")
        _info("cfg", f"enriched_csv_dir : {args.enriched_csv_dir}")
        _info("cfg", f"out_dir          : {args.out_dir}")
        _info(
            "cfg",
            f"sel-eff threshold: {args.sel_eff_threshold}  "
            f"split_floor: {args.split_floor}",
        )
        _info("cfg", f"arch / device    : {args.arch} / {args.device}")
        _info("cfg", f"min_cell_gemms   : {args.min_cell_gemms}")

    # ── load model bundle ────────────────────────────────────────────────────
    models_pt = args.models_dir / "models.pt"
    cells_json = args.models_dir / "cells.json"
    if not models_pt.exists():
        _err("err", f"missing models.pt at {models_pt}")
        return 1
    if not cells_json.exists():
        _err("err", f"missing cells.json at {cells_json}")
        return 1

    # Stage04b is only called from active rounds (round R>=1) where
    # --models-dir points at the PRIOR round's stage05 and
    # --enriched-csv-dir points at THIS round's stage04 -- so the
    # GEMMs are by-construction held-out (stage01's history-dedup
    # excluded every prior round's shapes). The orchestrator skips
    # stage04b entirely in round 0, where there's no prior model and
    # no unknown data. The `validation_kind` field is recorded in
    # decisions.json so downstream consumers can branch on it; the
    # value is always "held_out_new_data" today.
    validation_kind = "held_out_new_data"

    if not args.quiet:
        _banner("Stage 1/4  Load models", _C.B)
    bundle = torch.load(models_pt, map_location="cpu", weights_only=False)
    q_names = bundle["q_names"]
    i_names = bundle["i_names"]
    x_names = bundle["x_names"]
    models = bundle["models"]  # cell -> {state_dict, feature_norms, dims}
    if not args.quiet:
        _ok(
            "models",
            f"loaded {len(models)} per-cell models  "
            f"(Q={len(q_names)} I={len(i_names)} X={len(x_names)})",
        )
        # Tell the operator about carry-forward provenance so it's
        # obvious when round_R's models.pt is a MERGE of this round's
        # retrain + prior PASS cells (the design that lets the
        # cumulative split tree grow past depth 1).
        n_this = int(bundle.get("n_models_trained_this_round") or len(models))
        n_carry = int(bundle.get("n_models_carried_forward") or 0)
        if n_carry > 0:
            _ok(
                "models",
                f"  (trained this round={n_this}; "
                f"carried forward from prior round={n_carry})",
            )

    # ── PRIOR model bundle (for same-data improvement check) ──────────
    # In active-learning rounds R>=2 the orchestrator passes a path to
    # round_(R-2)/stage05/models.pt -- i.e. the model STATE that the
    # round_(R-1) retraining changed. We score that bundle on THIS
    # round's GEMMs to get a strict, same-dataset baseline for the
    # SPLIT trigger's "did augmentation help?" check. Without this
    # bundle (round 1, or stale runs) we fall back to comparing this
    # round's sel_eff to the prior decisions.json's sel_eff -- which
    # is also on unknown data, just on a DIFFERENT unknown dataset
    # (the prior round's new shapes), so the delta carries an extra
    # data-distribution-shift noise term.
    prior_models: Dict[str, Dict[str, Any]] = {}
    if args.prior_models is not None:
        try:
            prior_bundle = torch.load(
                str(args.prior_models), map_location="cpu", weights_only=False
            )
            prior_models = prior_bundle.get("models") or {}
            if not args.quiet:
                _ok(
                    "prior",
                    f"loaded prior model bundle: "
                    f"{len(prior_models)} cells (same-data "
                    f"improvement check will use this)",
                )
        except Exception as e:
            _warn(
                "prior",
                f"failed to load --prior-models "
                f"({args.prior_models}): {e}; falling back to "
                f"cross-round sel_eff comparison",
            )
            prior_models = {}

    # ── cumulative split tree: route NEW gemms to the SAME leaf
    #    sub-cells the bundle was trained on ───────────────────────────────
    # WITHOUT this, gemms for a split cell are binned under their (now
    # model-less) BASE label, so the split children see 0 new gemms
    # ("no_change" forever) while the base label gets "retrain
    # (no_prior_model)" -- silently UNDOING every split. The bundle's
    # labels (with `#axis...` segments) are leaves of this tree, so the
    # routing here must match. Prefer explicit --prior-round-dir; else
    # auto-discover round_*/stage04b/splits.json under the run dir
    # inferred from --models-dir (= <run>/round_<R-1>/stage05).
    from lib import subcells as sc  # local import: `sc` is (re)imported

    # later in main() too, which makes it function-local; import here so
    # it's bound before this first use.
    split_tree: Dict[str, Any] = {}
    prior_round_dirs = list(args.prior_round_dir or [])
    if not prior_round_dirs:
        try:
            run_dir = Path(args.models_dir).resolve().parent.parent
            prior_round_dirs = sorted(
                (
                    d
                    for d in run_dir.glob("round_*")
                    if (d / "stage04b" / "splits.json").exists()
                ),
                key=lambda p: p.name,
            )
        except Exception:
            prior_round_dirs = []
    if prior_round_dirs:
        try:
            split_tree = sc.load_cumulative_split_tree(prior_round_dirs)
        except Exception as e:
            _warn(
                "splits",
                f"failed to load cumulative split tree "
                f"({e}); routing NEW gemms by BASE cell (split "
                f"children would starve)",
            )
            split_tree = {}
    if not args.quiet and split_tree:
        _ok(
            "splits",
            f"cumulative split tree: {len(split_tree)} parent "
            f"cell(s) -> routing NEW gemms to leaf sub-cells",
        )

    # ── load new enriched chunks ────────────────────────────────────────────
    if not args.quiet:
        _banner("Stage 2/4  Load new enriched chunks", _C.B)
    new_cells, load_stats = _train.load_enriched_chunks(
        str(args.enriched_csv_dir), args.quiet, split_tree=(split_tree or None)
    )
    if not args.quiet:
        _ok("data", f"new GEMMs by cell: {len(new_cells)} populated")

    # ── optional: prepare Origami baseline ─────────────────────────────────
    origami_configs: Optional[Dict[Tuple, List[Any]]] = None
    origami_hw: Any = None
    if args.with_origami_baseline:
        if not args.quiet:
            _banner("Stage 3/4  Load Origami baseline", _C.B)
        if args.origami_python_path and args.origami_python_path not in sys.path:
            sys.path.insert(0, args.origami_python_path)
        try:
            import origami as _ori_mod  # type: ignore

            _train.origami = _ori_mod  # train.compute_origami_sel_eff uses
            _train._ORIGAMI_OK = True
        except ImportError as e:
            _warn("origami", f"import failed ({e}); skipping baseline")
            args.with_origami_baseline = False
        if args.with_origami_baseline:
            try:
                from stage04_convert_to_enriched_dataset import (  # type: ignore
                    load_origami_configs_by_dtype as _load_cfgs,
                    _origami_hardware as _ori_hw,
                )

                _loaded = _load_cfgs(
                    args.origami_library_dir,
                    quiet=args.quiet,
                    scale_mode=args.scale_mode,
                )
                origami_configs = _loaded[0] if isinstance(_loaded, tuple) else _loaded
                origami_hw = _ori_hw(
                    args.origami_device, quiet=args.quiet, arch=args.arch
                )
                if origami_hw is None:
                    _warn("origami", "no hardware descriptor; baseline disabled")
                    origami_configs = None
                elif not args.quiet:
                    _ok("origami", "Origami baseline ENABLED")
            except Exception as e:
                _warn("origami", f"setup failed ({e}); baseline disabled")
                origami_configs = None
                origami_hw = None

    # ── per-cell evaluation ────────────────────────────────────────────────
    if not args.quiet:
        _banner("Stage 4/4  Per-cell sel_eff on NEW (unknown) data", _C.B)

    # Prior decisions (repeatable). Used purely to count cumulative
    # retrain attempts per cell (gates --split-after-attempts) and to
    # record the previous round's sel_eff in this round's
    # decisions.json (informational). Prior decisions DO NOT drive
    # the pass/retrain/split logic -- that's based purely on THIS
    # round's measurements against THIS run's thresholds + the
    # same-data improvement baseline from --prior-models.
    prior_sel_eff_by_cell: Dict[str, Optional[float]] = {}
    prior_origami_by_cell: Dict[str, Optional[float]] = {}
    # Last-measured low<70% GEMM count (and its n_eval denominator)
    # per cell, carried forward exactly like `prior_sel_eff_by_cell`.
    # A cell that passed in round R and gets 0 new GEMMs in rounds
    # R+1.. has no fresh per-GEMM data to recompute the low-sel_eff
    # tail from, but the tail measured when it last passed should
    # stay VISIBLE -- so we echo the carried value on the
    # carry-forward line instead of printing nothing.
    prior_low_seleff_by_cell: Dict[str, Optional[int]] = {}
    prior_low_n_eval_by_cell: Dict[str, Optional[int]] = {}
    last_decision_by_cell: Dict[str, str] = {}
    retrain_attempts_by_cell: Dict[str, int] = {}
    # `prior_paths` MUST be in chronological order (round_0,
    # round_1, ..., round_(R-1)); the orchestrator passes them this
    # way. `last_decision_by_cell` ends up holding each cell's
    # MOST RECENT non-(no_change) decision, which is what
    # `insufficient_new_data` cells fall back to.
    prior_paths = list(args.prior_decisions or [])
    for i, pth in enumerate(prior_paths):
        if not pth.exists():
            _warn("prior", f"missing prior decisions: {pth}")
            continue
        try:
            with pth.open() as f:
                pj = json.load(f)
        except Exception as e:
            _warn("prior", f"failed to load {pth}: {e}")
            continue
        for r in pj.get("per_cell", []):
            cell = r["cell"]
            # `prior_sel_eff_by_cell` is informational -- decision
            # logic ignores it. We keep it so the per-cell row in
            # this round's decisions.json carries the most recent
            # measured sel_eff for human inspection.
            if r.get("sel_eff_new") is not None:
                prior_sel_eff_by_cell[cell] = r.get("sel_eff_new")
            # Carry the prior round's Origami sel_eff too, so carried/
            # no_change rows can show it (like RETRN rows do).
            if r.get("origami_sel_eff_new") is not None:
                prior_origami_by_cell[cell] = r.get("origami_sel_eff_new")
            # Carry forward the low-sel_eff tail too. Only update when
            # this prior row actually evaluated GEMMs (n_eval > 0);
            # carry-forward rows themselves re-stamp these so the
            # most recent real measurement always wins.
            _pne = r.get("n_eval")
            if _pne is not None and int(_pne or 0) > 0:
                prior_low_seleff_by_cell[cell] = r.get("n_low_seleff_lt_0p70")
                prior_low_n_eval_by_cell[cell] = int(_pne)
            dec = r.get("decision")
            # A round where this cell got 0 new GEMMs (and thus
            # decision="no_change") is NOT a real signal -- skip
            # it so the carry-forward logic looks past these and
            # finds the actual last pass/retrain/split decision.
            if dec and dec != "no_change":
                last_decision_by_cell[cell] = dec
            if dec == "retrain":
                retrain_attempts_by_cell[cell] = (
                    retrain_attempts_by_cell.get(cell, 0) + 1
                )
    if prior_paths and not args.quiet:
        _ok(
            "prior",
            f"loaded {len(prior_paths)} prior decisions.json; "
            f"{len(prior_sel_eff_by_cell)} cells have a prior sel_eff_new; "
            f"{sum(retrain_attempts_by_cell.values())} retrain attempts "
            f"across {len(retrain_attempts_by_cell)} cells",
        )

    # subcells helper.
    sys.path.insert(0, str(ML_RECOMMENDER_ROOT))
    from lib import subcells as sc  # noqa: E402

    def _fmt_se(se: Optional[float]) -> str:
        """Render a prior sel_eff like the live sel_eff, width-6 so the
        percent column lines up across rows: ' NN.NN%'."""
        return f"{se*100:6.2f}%" if se is not None else "    --"

    def _display_prior_se(label: str) -> Optional[float]:
        """Prior sel_eff for the per-cell LOG line only. A freshly-split
        child has no own prior decision, so fall back to its nearest
        ANCESTOR's prior sel_eff (= the cell's sel_eff BEFORE it was
        split) -- so both children of a split show the SAME pre-split
        number. The SPLIT/improvement DECISION still uses the cell's OWN
        prior_se (None for a new child), unchanged."""
        se = prior_sel_eff_by_cell.get(label)
        cur = label
        while se is None and "#" in cur:
            cur = sc.parent_of(cur)
            if not cur:
                break
            se = prior_sel_eff_by_cell.get(cur)
        return se

    def _display_prior_origami(label: str) -> Optional[float]:
        """Carried Origami sel_eff for the per-cell LOG line (same
        ancestor-fallback as _display_prior_se for freshly-split kids)."""
        oe = prior_origami_by_cell.get(label)
        cur = label
        while oe is None and "#" in cur:
            cur = sc.parent_of(cur)
            if not cur:
                break
            oe = prior_origami_by_cell.get(cur)
        return oe

    results: List[Dict[str, Any]] = []
    cells_pass: List[str] = []
    cells_pass_carry: List[str] = []  # passed in a prior round; no new data this round
    cells_no_change: List[str] = []  # no new data AND no prior pass to carry forward
    cells_retrain: List[str] = []
    cells_split: List[str] = []
    cells_no_data: List[str] = []  # union: cells_pass_carry + cells_no_change
    cells_no_model: List[str] = []
    split_rules: List["sc.SplitRule"] = []

    # Union of {cells with a model} and {cells with new data}.
    all_labels = sorted(set(models.keys()) | set(new_cells.keys()))
    # Fixed cell-name column width so the sel_eff column lines up across
    # every per-cell row, regardless of how deep the adaptive splits made
    # the labels. Clamped so short-name rounds don't get absurd padding
    # and very deep names don't blow the line width out.
    _lbl_w = min(max(max((len(l) for l in all_labels), default=24), 24), 80)
    for label in all_labels:
        has_model = label in models
        new_gemms = new_cells.get(label, [])
        n_new = len(new_gemms)
        # Per-cell validation kind == run-wide kind. Always
        # "held_out_new_data" today because the orchestrator never
        # calls stage04b in round 0.
        cell_validation_kind = validation_kind

        if not has_model:
            cells_no_model.append(label)
            results.append(
                {
                    "cell": label,
                    "decision": "retrain",
                    "reason": "no_prior_model",
                    "n_new_gemms": n_new,
                    "sel_eff_new": None,
                    "origami_sel_eff_new": None,
                }
            )
            if not args.quiet:
                _warn(
                    "eval",
                    f"{label:{_lbl_w}s}  decision=retrain "
                    f"(no_prior_model, {n_new} new gemms)",
                )
            continue
        if n_new < args.min_cell_gemms:
            # Carry forward the cell's MOST RECENT real decision.
            # Active learning intentionally stops probing cells that
            # passed in earlier rounds (no incremental information
            # to gain), so a "0 new gemms this round" outcome for
            # those cells is EXPECTED -- it means "still passing",
            # not "needs retrain". Conversely, a cell that's never
            # been evaluated yet gets `no_change` so it doesn't
            # pollute the retrain-attempts counter.
            cells_no_data.append(label)
            prior_dec = last_decision_by_cell.get(label)
            prior_se = prior_sel_eff_by_cell.get(label)
            prior_low = prior_low_seleff_by_cell.get(label)
            prior_low_n = prior_low_n_eval_by_cell.get(label)
            if prior_dec == "pass":
                decision = "pass"
                reason = "carried_forward_no_new_data"
                cells_pass_carry.append(label)
            else:
                decision = "no_change"
                reason = "no_new_data"
                cells_no_change.append(label)
            results.append(
                {
                    "cell": label,
                    "decision": decision,
                    "reason": reason,
                    "n_new_gemms": n_new,
                    "n_eval": 0,
                    "sel_eff_new": prior_se,
                    # Re-stamp the carried low-sel_eff tail so it keeps
                    # propagating to subsequent rounds (the prior-decisions
                    # loader only picks up rows with n_eval>0, so we store
                    # the carried n_eval under a separate key to preserve
                    # provenance without re-triggering the >0 gate).
                    "n_low_seleff_lt_0p70": prior_low,
                    "carried_low_n_eval": prior_low_n,
                    "origami_sel_eff_new": None,
                    "prior_decision": prior_dec,
                }
            )
            if not args.quiet:
                # SAME aligned column layout as the evaluated
                # [RETRN]/[PASS ]/[SPLIT] rows below; all values are
                # CARRIED from the cell's last real evaluation (n_eval=0
                # this round), labeled (carried). Tags:
                #   [CARRY] = passed a prior round, no new data now
                #   [NOCHG] = never passed AND no new data now
                pse = _display_prior_se(label)
                ore = _display_prior_origami(label)
                ori_str = (
                    f"  origami={_fmt_se(ore)}(carried)" if ore is not None else ""
                )
                low_str = ""
                if prior_low is not None and prior_low_n:
                    low_str = (
                        f"  low<70%={prior_low:>3d}/{prior_low_n:>3d}" f"(carried)"
                    )
                tag, color = ("CARRY", _C.G) if decision == "pass" else ("NOCHG", _C.Y)
                print(
                    f"  {color}[{tag}]{_C.E} {label:{_lbl_w}s}  "
                    f"sel_eff={_fmt_se(pse)}(carried)  "
                    f"n_eval={0:>3d}/{n_new:>3d}{ori_str}{low_str}",
                    flush=True,
                )
            continue

        # Build the student for THIS round's evaluation. The model
        # state_dict, dims, and per-feature whitening stats come
        # from this round's cell entry in `models` (which is the
        # PRIOR round's stage05 output -- carried-forward or
        # freshly retrained as of round_(R-1)).
        m = models[label]
        norms = m["feature_norms"]
        student = _train.GenericTwoTower(
            q_dim=len(q_names),
            i_dim=len(i_names),
            x_dim=len(x_names),
            embed_dim=m["embed_dim"],
            hidden_dim=m["hidden_dim"],
            inter_hidden=m["inter_hidden"],
        ).to(args.device)
        student.load_state_dict(m["state_dict"])

        # ── DEPLOYED sel_eff (the only metric stage04b reports) ────
        # `evaluate_deployed_sel_eff` mirrors the production
        # inference path: origami.rank_configs() ranks every
        # compatible kernel for the problem; we keep the top
        # `smart_k`; the student picks top-1 among those K.
        # That's the EXACT model that ships -- sel_eff_new here is
        # the held-out generalization measurement of the deployed
        # selector, NOT a more permissive 'pick from every kernel'
        # number. Per-GEMM log-ratios from this call also drive
        # the CART axis-selection inside stage04b's split decision,
        # so 'what we measured' and 'what we optimize' are the
        # same picks.
        ev = _train.evaluate_deployed_sel_eff(
            student,
            new_gemms,
            q_names,
            i_names,
            x_names,
            args.arch,
            norms,
            origami_configs or {},
            origami_hw,
            args.smart_k,
            args.device,
        )
        sel_eff_new = ev["sel_eff"]
        n_eval = ev["n_eval"]
        per_gemm_log_ratio = ev["per_gemm_log_ratio"]
        # GEMM-level sel_eff distribution. `per_gemm_log_ratio[gi]`
        # = log(pick_us / winner_us); per-GEMM sel_eff is its
        # negative-exp. Count how many GEMMs have sel_eff < 0.70
        # so the operator can see which cells have a long tail of
        # bad picks (the aggregate sel_eff_new hides this -- e.g. a
        # cell could average 80% with 30% of its GEMMs below 70%
        # AND the other 70% above 95%).
        _LOW_THRESH = 0.70
        _LOW_LOG = -math.log(_LOW_THRESH)  # ≈ 0.3567
        _low_mask = per_gemm_log_ratio > _LOW_LOG
        n_low_seleff = int(np.nansum(_low_mask))
        # Defensive: NaN log_ratio means the GEMM was excluded
        # from the eval (no candidates / no winner); don't count.

        # ── SAME-DATA improvement baseline (also deployed pipeline) ──
        # Evaluate the round_(R-2) version of this cell on the EXACT
        # SAME `new_gemms` we just scored the current model on, via
        # the SAME deployed-pipeline path. The delta
        # `sel_eff_new - sel_eff_prior_model_same_data` then
        # isolates "did round_(R-1)'s retraining help?" without
        # any data-distribution shift or pipeline mismatch.
        sel_eff_prior_model_same_data: Optional[float] = None
        if label in prior_models:
            pm = prior_models[label]
            try:
                p_student = _train.GenericTwoTower(
                    q_dim=len(q_names),
                    i_dim=len(i_names),
                    x_dim=len(x_names),
                    embed_dim=pm["embed_dim"],
                    hidden_dim=pm["hidden_dim"],
                    inter_hidden=pm["inter_hidden"],
                ).to(args.device)
                p_student.load_state_dict(pm["state_dict"])
                p_ev = _train.evaluate_deployed_sel_eff(
                    p_student,
                    new_gemms,
                    q_names,
                    i_names,
                    x_names,
                    args.arch,
                    pm["feature_norms"],
                    origami_configs or {},
                    origami_hw,
                    args.smart_k,
                    args.device,
                )
                if not math.isnan(p_ev["sel_eff"]):
                    sel_eff_prior_model_same_data = float(p_ev["sel_eff"])
            except Exception as e:
                if not args.quiet:
                    _warn(
                        "prior",
                        f"{label}: same-data prior eval "
                        f"failed ({e}); falling back to cross-round "
                        f"prior_sel_eff_new for the improvement "
                        f"check",
                    )

        # Origami baseline on the same gemms.
        ori_sel = None
        ori_n_eval = 0
        ob: Optional[Dict[str, Any]] = None
        if origami_configs is not None and origami_hw is not None:
            ob = _train.compute_origami_sel_eff(
                new_gemms,
                origami_configs,
                origami_hw,
                _mx_block_size_for_scale_mode(args.scale_mode),
            )
            if ob and ob.get("n_eval", 0) > 0:
                ori_sel = ob["sel_eff"]
                ori_n_eval = ob["n_eval"]

        prior_se = prior_sel_eff_by_cell.get(label)
        prior_retrain_attempts = retrain_attempts_by_cell.get(label, 0)

        # Per-cell thresholds. "origami" means: use this cell's measured
        # origami sel_eff (so the model has to do at least as well as
        # plain Origami to PASS, and worse than origami to be a split
        # candidate). If the origami baseline didn't measure this cell,
        # the threshold falls back to a safe default (0.95 / 0.85).
        sel_eff_threshold = _resolve_threshold(
            args.sel_eff_threshold, ori_sel, fallback=0.95
        )
        split_floor = _resolve_threshold(args.split_floor, ori_sel, fallback=0.85)

        # Decision policy: SPLIT a cell ONLY when
        # data augmentation has run out of road. Concretely, all three
        # conditions must hold simultaneously:
        #
        #   (a) sel_eff_new < split_floor
        #       (this round's measurement on unknown shapes is bad)
        #   (b) prior_retrain_attempts >= split_after_attempts
        #       (the cell has ALREADY received N rounds of augmented
        #        training data -- no more headroom from "just give it
        #        more data")
        #   (c) sel_eff_new did NOT improve over the prior round's
        #       sel_eff by at least split_min_improvement
        #       (augmentation isn't helping -- the model has plateaued)
        #
        # Otherwise the cell is RETRAINED (with cross-round data
        # augmentation; fresh weights). The rationale: don't split a
        # cell every round just because it's below floor -- only split
        # once more data has stopped helping.
        # PASS is independent: sel_eff_new >= sel_eff_threshold.
        # ── Structural width-split (opt-in; robust to a misleading
        #    synthetic sel_eff) ───────────────────────────────────────
        # The sel_eff trigger below relies on the synthetic held-out
        # being a faithful proxy for real OOD data. For a leaf the deploy
        # runtime routes a 9-10 octave magnitude range to (e.g.
        # Large|Large|LargeK), that proxy is optimistic -- the cell can
        # score >= threshold on same-distribution synthetic shapes yet be
        # far worse on real workloads, and then PASS-and-freeze forever.
        # When --structural-split-min-octaves > 0 we carve ANY cell whose
        # widest splittable axis still spans more than that many octaves,
        # REGARDLESS of sel_eff / attempts / improvement, until its leaves
        # are narrow enough. This guarantees the worst offenders get
        # carved by magnitude even if they (mis-)clear the bar -- a
        # structural guarantee, not a function of the proxy signal.
        structural_octaves = 0.0
        structural_axis = None
        if getattr(args, "structural_split_min_octaves", 0.0) > 0:
            _oct_by_axis = sc.cell_axis_octaves(label)
            if _oct_by_axis:
                structural_axis, structural_octaves = max(
                    _oct_by_axis.items(), key=lambda kv: kv[1]
                )
        _struct_max_depth = getattr(args, "structural_split_max_depth", -1)
        if _struct_max_depth is None or _struct_max_depth < 0:
            _struct_max_depth = args.max_split_depth
        _struct_depth_cap = min(args.max_split_depth, int(_struct_max_depth))
        force_structural = (
            structural_octaves >= getattr(args, "structural_split_min_octaves", 0.0) > 0
            and sc.split_depth(label) < _struct_depth_cap
        )

        if math.isnan(sel_eff_new):
            decision = "retrain"
            reason = "sel_eff_nan_on_new_data"
        elif force_structural:
            decision = "split"
            reason = (
                f"structural_width({structural_axis}="
                f"{structural_octaves:.1f}oct>="
                f"{getattr(args, 'structural_split_min_octaves', 0.0):.1f}"
                f"; sel_eff={sel_eff_new:.3f})"
            )
        elif sel_eff_new >= sel_eff_threshold:
            decision = "pass"
            reason = f"above_threshold(th={sel_eff_threshold:.3f})"
        else:
            # BELOW THRESHOLD. With no Origami fallback, EVERY cell must
            # be driven to >= threshold by data augmentation OR carving --
            # so the whole band [split_floor, threshold) is splittable,
            # not just below split_floor. A cell becomes a SPLIT candidate
            # once augmentation has run out of road:
            #   (a) prior_retrain_attempts >= split_after_attempts, AND
            #   (b) NOT improving by >= split_min_improvement, OR it's
            #       below split_floor (hopeless -- carve without waiting
            #       on the improvement check).
            # Otherwise RETRAIN (more data / fresh init).
            # Improvement-baseline preference: (1) round_(R-2) bundle on
            # the SAME unknown GEMMs (isolates the model-change effect),
            # else (2) cross-round prior decisions.json sel_eff (noisier).
            attempts_ok = prior_retrain_attempts >= args.split_after_attempts
            if sel_eff_prior_model_same_data is not None:
                improvement_baseline = sel_eff_prior_model_same_data
                improvement_source = "same_data_prior_model"
            else:
                improvement_baseline = prior_se
                improvement_source = "cross_round_prior_decisions"
            not_improving = (
                improvement_baseline is None
                or args.split_min_improvement <= 0
                or (sel_eff_new - improvement_baseline) < args.split_min_improvement
            )
            below_floor = split_floor > 0 and sel_eff_new < split_floor
            if attempts_ok and (not_improving or below_floor):
                decision = "split"
                base = (
                    f"below_floor({split_floor:.3f})"
                    if below_floor
                    else f"below_threshold({sel_eff_threshold:.3f})"
                )
                if improvement_baseline is None:
                    reason = (
                        f"{base}__attempts={prior_retrain_attempts}"
                        f">={args.split_after_attempts}__no_prior"
                    )
                else:
                    reason = (
                        f"{base}__attempts={prior_retrain_attempts}"
                        f">={args.split_after_attempts}__"
                        f"delta={(sel_eff_new - improvement_baseline):+.3f}"
                        f"[{improvement_source}]"
                    )
            elif not attempts_ok:
                decision = "retrain"
                reason = (
                    f"below_threshold__augment_more_first("
                    f"{prior_retrain_attempts+1}/"
                    f"{args.split_after_attempts})"
                )
            else:  # improving
                decision = "retrain"
                reason = (
                    f"below_threshold__augment_helping(delta="
                    f"{(sel_eff_new - (improvement_baseline or 0.0)):+.3f}"
                    f">={args.split_min_improvement:.3f}"
                    f"[{improvement_source}])"
                )

        # (The decision logic above already gates SPLIT on
        # `prior_retrain_attempts >= split_after_attempts` AND
        # `not improving`, so no extra demotion is needed here.)

        # FINAL-ROUND guard: a split decided here has no subsequent round to
        # synthesize + TRAIN its children, so it would leave UNTRAINED leaves
        # in the deployed split tree (orphaned GEMM regions). Demote to
        # retrain so every deployed leaf is a trained cell.
        if getattr(args, "final_round", False) and decision == "split":
            decision = "retrain"
            reason = f"{reason}__final_round_no_split"

        # If we still picked "split" but the cell can't be subdivided
        # (every axis is too narrow, child counts too small, or the
        # cell has hit max_split_depth), fall back to retrain.
        split_axis = split_threshold = None
        if decision == "split":
            # Filter NaN log_ratios; keep gemms+ratios aligned.
            log_ratios_list: List[float] = []
            gemms_for_split: List[Dict[str, Any]] = []
            for g, lr in zip(new_gemms, per_gemm_log_ratio):
                if not math.isnan(float(lr)):
                    gemms_for_split.append(g)
                    log_ratios_list.append(float(lr))
            picked = sc.choose_split_for_cell(
                label,
                gemms_for_split,
                log_ratios=log_ratios_list if log_ratios_list else None,
                # BOTH children must be independently trainable, not just
                # non-empty. With `max(1, min_cell_gemms // 2)` a split could
                # proceed when one side held a single GEMM; that side then drew
                # no shapes in the next round, never got a trained cell, and the
                # parent -- which HAD covered the whole region -- was gone.
                # Such a split silently drops a sibling, leaving an orphaned
                # region whose GEMMs get no prediction at all (they end up
                # on an unranked fallback kernel). Requiring the full
                # min_cell_gemms on each side keeps the parent whenever a clean
                # two-child split is not supportable, which preserves 100%
                # coverage of the GEMM space.
                min_split_count=max(1, args.min_cell_gemms),
                max_split_depth=args.max_split_depth,
            )
            if picked is None:
                decision = "retrain"
                reason = f"{reason}__unsplittable"
            else:
                split_axis, split_threshold = picked
                split_rules.append(
                    sc.build_split_rule(
                        cell_label=label,
                        axis=split_axis,
                        threshold=split_threshold,
                        reason=reason,
                        prior_sel_eff=prior_se,
                        this_sel_eff=sel_eff_new,
                    )
                )

        results.append(
            {
                "cell": label,
                "decision": decision,
                "reason": reason,
                "validation_kind": cell_validation_kind,
                "n_new_gemms": n_new,
                "n_eval": n_eval,
                "sel_eff_new": sel_eff_new,
                # Geomean pick / winner us from `evaluate_deployed_sel_eff`.
                # Saved alongside `sel_eff_new` so the end-of-run summary
                # can compute weighted-log-mean aggregates the same way
                # stage 5's "Done" block does (winner_us_geomean +
                # global_sel_eff (model) + global_origami_sel_eff lines).
                "pick_us_geomean": ev.get("pick_us_geomean"),
                "winner_us_geomean": ev.get("winner_us_geomean"),
                # Per-GEMM sel_eff distribution proxy: how many of the
                # n_eval GEMMs landed BELOW the 0.70 threshold. A high
                # count here flags cells where the model is uneven --
                # large variance between "perfect" picks and "broken"
                # picks. Currently informational; the training-side fix
                # is to up-weight low-sel_eff GEMMs during the next
                # round's pair sampling.
                "n_low_seleff_lt_0p70": int(n_low_seleff),
                # Same-data improvement baseline: prior model on THIS
                # round's GEMMs. None when --prior-models wasn't passed
                # (e.g. round 1) or didn't carry this cell. When present,
                # the SPLIT trigger's improvement check uses
                # (sel_eff_new - sel_eff_prior_model_same_data); when
                # absent, it falls back to prior_sel_eff_new (the
                # cross-round, different-dataset comparison).
                "sel_eff_prior_model_same_data": sel_eff_prior_model_same_data,
                "prior_sel_eff_new": prior_se,
                "prior_retrain_attempts": prior_retrain_attempts,
                "origami_sel_eff_new": ori_sel,
                "origami_n_eval": ori_n_eval,
                "origami_pick_us_geomean": (
                    ob.get("pick_us_geomean") if (ob and ori_sel is not None) else None
                ),
                "split_axis": split_axis,
                "split_threshold": split_threshold,
            }
        )
        if decision == "pass":
            cells_pass.append(label)
        elif decision == "split":
            cells_split.append(label)
        else:
            cells_retrain.append(label)

        if not args.quiet:
            # `sel_eff` is THIS round's measurement on a dataset that
            # is unknown to the model under test (round_R's new
            # shapes in active mode; the cell's held-out split in
            # snapshot mode). `origami` is Origami's own sel_eff on
            # the SAME dataset, computed inline above so the two are
            # directly comparable. We ALSO print the prior round's
            # sel_eff (`prior=`, with delta `d`) -- it plays no role in
            # the decision, but it makes the round-over-round OOB trend
            # visible at a glance.
            ori_str = f"  origami={ori_sel*100:6.2f}%" if ori_sel is not None else ""
            # prior_sel_eff: this cell's sel_eff on the PRIOR round's
            # held-out shapes -- a DIFFERENT unknown dataset, so the
            # delta `d` is noisier than a same-data comparison. Shown to
            # eyeball the round-over-round OOB trend (good enough for
            # tracking OOB progress).
            # Display prior = own prior, or (freshly-split child) the
            # pre-split ANCESTOR's sel_eff -- so both split halves show
            # the SAME pre-split number. Decision logic still uses the
            # own prior_se above.
            pse_disp = _display_prior_se(label)
            # Fixed-width prior block so the n_eval column lines up across
            # all evaluated rows (active branch and the no-prior fallback
            # are both 26 chars wide).
            if pse_disp is not None and not math.isnan(sel_eff_new):
                prior_str = (
                    f"  prior={pse_disp*100:6.2f}%"
                    f"(d={(sel_eff_new - pse_disp)*100:+7.2f})"
                )
            else:
                prior_str = "  prior=" + "--".rjust(18)
            split_str = ""
            if decision == "split":
                split_str = (
                    f"  SPLIT[{split_axis}={split_threshold}]" f"  reason={reason}"
                )
            color = (
                _C.G if decision == "pass" else _C.R if decision == "split" else _C.Y
            )
            tag = (
                "PASS "
                if decision == "pass"
                else "SPLIT" if decision == "split" else "RETRN"
            )
            low_str = f"  low<70%={n_low_seleff:>3d}/{n_eval:>3d}" if n_eval > 0 else ""
            print(
                f"  {color}[{tag}]{_C.E} {label:{_lbl_w}s}  "
                f"sel_eff={sel_eff_new*100:6.2f}%{prior_str}  "
                f"n_eval={n_eval:>3d}/{n_new:>3d}{ori_str}{low_str}{split_str}",
                flush=True,
            )

    # ── aggregate sel_eff across all evaluated cells ───────────────────────
    # Same shape as stage 5's "Done" block: weighted-log-mean across
    # cells, weighted by `n_eval` (the per-cell GEMM count that
    # actually contributed to the cell's geomean). Carried-forward
    # and no_change cells have n_eval=0 and drop out of the
    # aggregate naturally -- they didn't get evaluated this round, so
    # they shouldn't sway "how well did the round go?" The aggregate
    # is also saved to `decisions.json` under "aggregate" so the
    # orchestrator / dashboards can plot the cross-round series.
    def _wlog_mean(pairs: List[Tuple[Optional[float], int]]) -> float:
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

    evaluated_rows = [
        r
        for r in results
        if (r.get("n_eval", 0) or 0) > 0 and r.get("sel_eff_new") is not None
    ]
    n_gemms_evaluated = sum(int(r["n_eval"] or 0) for r in evaluated_rows)
    n_cells_evaluated = len(evaluated_rows)
    n_low_seleff_total = sum(
        int(r.get("n_low_seleff_lt_0p70", 0) or 0) for r in evaluated_rows
    )
    g_sel_eff = _wlog_mean(
        [(r.get("sel_eff_new"), r["n_eval"]) for r in evaluated_rows]
    )
    g_model_pick_us = _wlog_mean(
        [(r.get("pick_us_geomean"), r["n_eval"]) for r in evaluated_rows]
    )
    g_winner_us = _wlog_mean(
        [(r.get("winner_us_geomean"), r["n_eval"]) for r in evaluated_rows]
    )
    g_ori_sel_eff = _wlog_mean(
        [
            (r.get("origami_sel_eff_new"), r.get("origami_n_eval", 0) or 0)
            for r in evaluated_rows
        ]
    )
    g_ori_pick_us = _wlog_mean(
        [
            (r.get("origami_pick_us_geomean"), r.get("origami_n_eval", 0) or 0)
            for r in evaluated_rows
        ]
    )

    def _json_float(v: float) -> Optional[float]:
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)

    aggregate = {
        "n_cells_evaluated": int(n_cells_evaluated),
        "n_gemms_evaluated": int(n_gemms_evaluated),
        "n_low_seleff_lt_0p70_total": int(n_low_seleff_total),
        "winner_us_geomean": _json_float(g_winner_us),
        "global_sel_eff_new": _json_float(g_sel_eff),
        "global_model_pick_us_geomean": _json_float(g_model_pick_us),
        "global_origami_sel_eff_new": _json_float(g_ori_sel_eff),
        "global_origami_pick_us_geomean": _json_float(g_ori_pick_us),
    }

    # ── summary ─────────────────────────────────────────────────────────────
    decisions_json = {
        "schema_version": "pipeline-v3-2026-05-27",
        "round_idx": int(args.round_idx),
        "models_dir": str(args.models_dir),
        "enriched_csv_dir": str(args.enriched_csv_dir),
        # `validation_kind` flags whether the per-cell sel_eff in
        # `per_cell` reflects true held-out generalization
        # ("held_out_new_data") or just training-set accuracy
        # ("training_snapshot"). Consumers (orchestrator,
        # stage07/08, dashboards) can branch on this; today it's
        # always "held_out_new_data" since stage04b never runs in
        # round 0 anymore.
        "validation_kind": validation_kind,
        "sel_eff_threshold": args.sel_eff_threshold,
        "split_floor": args.split_floor,
        "split_min_improvement": args.split_min_improvement,
        "split_after_attempts": args.split_after_attempts,
        "elapsed_s": time.time() - t0,
        "summary": {
            "pass": len(cells_pass),
            "pass_carried_forward": len(cells_pass_carry),
            "no_change": len(cells_no_change),
            "retrain": len(cells_retrain),
            "split": len(cells_split),
            "no_model": len(cells_no_model),
            "no_data": len(cells_no_data),
            "total_evaluated": len(all_labels),
        },
        "aggregate": aggregate,
        "cells_pass": cells_pass,
        "cells_pass_carried_forward": cells_pass_carry,
        "cells_no_change": cells_no_change,
        "cells_retrain": cells_retrain,
        "cells_split": cells_split,
        "cells_no_model": cells_no_model,
        "cells_no_data": cells_no_data,
        "per_cell": results,
    }
    out_path = args.out_dir / "decisions.json"
    with out_path.open("w") as f:
        json.dump(decisions_json, f, indent=2)

    # retrain_cells.txt feeds stage05's --only-cells. For "retrain"
    # decisions the parent label IS the cell that gets retrained. For
    # "split" decisions the parent disappears in THIS round's cumulative
    # split tree and is replaced by its two children -- those are the
    # labels stage05 will see THIS round (the orchestrator passes
    # `--current-round-dir <round>` so stage05's split tree includes
    # the just-written splits.json), so we write the children here.
    retrain_targets: List[str] = list(cells_retrain)
    for rule in split_rules:
        retrain_targets.append(rule.lo_label)
        retrain_targets.append(rule.hi_label)
    out_only_cells = args.out_dir / "retrain_cells.txt"
    with out_only_cells.open("w") as f:
        f.write(",".join(retrain_targets))

    # splits.json (always written; empty `splits: {}` if no cell split
    # this round).
    splits_path = args.out_dir / "splits.json"
    sc.write_splits_json(splits_path, args.round_idx, split_rules)

    if not args.quiet:
        _banner("Done", _C.G)

        # Aggregate-sel_eff block FIRST (mirrors stage 5's layout).
        def _fmt_global(name: str, se: float, pick: float) -> str:
            if isinstance(se, float) and math.isnan(se):
                return f"  {_C.BD}{name:32s}:{_C.E} n/a"
            return (
                f"  {_C.BD}{name:32s}:{_C.E} "
                f"{se*100:6.2f}%  pick_us_geomean={pick:8.2f}us"
            )

        print(
            f"  {_C.BD}{'winner_us_geomean':32s}:{_C.E} "
            f"{g_winner_us:8.2f}us  (empirical-best baseline)",
            flush=True,
        )
        print(
            _fmt_global("global_sel_eff_new (model)", g_sel_eff, g_model_pick_us),
            flush=True,
        )
        print(
            _fmt_global("global_origami_sel_eff_new", g_ori_sel_eff, g_ori_pick_us),
            flush=True,
        )
        print(
            f"  {_C.BD}{'n_cells_evaluated':32s}:{_C.E} " f"{n_cells_evaluated}",
            flush=True,
        )
        print(
            f"  {_C.BD}{'n_gemms_evaluated':32s}:{_C.E} " f"{n_gemms_evaluated:,}",
            flush=True,
        )
        pct_low = 100.0 * n_low_seleff_total / max(n_gemms_evaluated, 1)
        print(
            f"  {_C.BD}{'n_gemms_low_seleff_lt_0p70':32s}:{_C.E} "
            f"{n_low_seleff_total:,}  ({pct_low:.1f}% of evaluated)",
            flush=True,
        )
        print(flush=True)
        # Decision breakdown.
        _ok("done", f"validation_kind   : {validation_kind}")
        _ok(
            "done",
            f"pass              : {len(cells_pass):3d} cells "
            f"(evaluated against new data; met threshold)",
        )
        _ok(
            "done",
            f"pass (carried)    : {len(cells_pass_carry):3d} cells "
            f"(prior pass; no new data this round)",
        )
        _ok(
            "done",
            f"retrain           : {len(cells_retrain):3d} cells "
            f"(evaluated against new data; below threshold)",
        )
        _ok("done", f"split             : {len(cells_split):3d} cells")
        _ok(
            "done",
            f"no_change         : {len(cells_no_change):3d} cells "
            f"(no new data; no prior pass to carry forward)",
        )
        _ok("done", f"no_model          : {len(cells_no_model):3d} cells")
        _ok("done", f"decisions.json     -> {out_path}")
        _ok("done", f"retrain_cells.txt  -> {out_only_cells}")
        _ok("done", f"splits.json        -> {splits_path}")
        if split_rules:
            _info(
                "tree",
                "splits decided this round (children are "
                "trained THIS round by stage05 -- the orchestrator "
                "passes --current-round-dir so stage05's cumulative "
                "split tree includes the splits.json above):",
            )
            this_round_tree = {r.cell: r for r in split_rules}
            for line in sc.format_split_tree(this_round_tree).splitlines():
                print(f"        {line}", flush=True)
        if cells_retrain or cells_split:
            _info(
                "hint",
                "THIS round's stage05 will retrain these "
                "cells (split cells: their two child sub-cells, "
                "routed via this round's splits.json).",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
