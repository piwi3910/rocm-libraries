# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage01_generate_OOB_shapes -- OOB shape generation.

Two modes:

  --mode initial   First-round seed (full grid; uniform target-per-cell
                   across the populated 96-cell grid).

  --mode active    Active-learning round. EVERY cell with a model in the
                   prior round gets the SAME shape budget; the prior
                   round's per-cell sel_eff is carried as informational
                   metadata only (the pass/retrain decision is made by
                   stage04b from the MEASURED sel_eff on the new shapes).

Both modes:

  - cell-uniform synthesis via log-uniform sampling per axis, with a
    30% chance of macro-tile-boundary snapping per axis
  - `is_rotating512_safe` filter excludes B>1 with max(M,N)>256 OR
    K>512 (bench-tool OOM safety; see lib/shapes.py)
  - HISTORY DEDUP across rounds: pass `--prior-round-dir <round_R>/`
    for every earlier round (auto-walks `<round>/stage01/shapes.yaml`,
    `<round>/stage04/`, `<round>/stage05/cells.json`); every (M,N,K,B)
    found in those is excluded so round_N never re-tests a shape from
    rounds 0..N-1. Additional explicit history sources are accepted
    via --prior-shapes-yaml and --prior-enriched-csv-dir.

Outputs in `--out-dir`:

  shapes.yaml          hipblaslt-bench yaml with default knobs
                       (iters=50, cold_iters=50, rotating:512). The
                       per-shape calibrated knobs come from stage02.
  categorization.json  per-cell budget + filter stats + provenance
  per_cell_plan.json   ACTIVE MODE ONLY: per-cell prior sel_eff + plan
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_V3 = THIS_DIR.parent
if str(PIPELINE_V3) not in sys.path:
    sys.path.insert(0, str(PIPELINE_V3))

import yaml  # noqa: E402

from lib import ui  # noqa: E402
from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml  # noqa: E402
from lib.fs import read_json, write_json  # noqa: E402
from lib.grid import TIER_RANGES_K, TIER_RANGES_MN, all_cell_labels  # noqa: E402
from lib.shapes import (
    Shape,
    _BPE,
    hbm_safe_bytes,  # noqa: E402
    is_rotating512_safe,
    shape_hash,
    synthesize_cell_shapes,
)
from lib import subcells as sc  # noqa: E402


# ── budget builders ──────────────────────────────────────────────────────────


def _build_initial_budget(target_per_cell: int) -> Dict[str, int]:
    """Uniform per-cell budget across the full 96-cell grid."""
    return {lbl: target_per_cell for lbl in all_cell_labels()}


def _build_active_plan(
    prior_cells_json: Path, shapes_per_cell: int, split_tree: Dict[str, "sc.SplitRule"]
) -> Dict[str, Dict[str, Any]]:
    """Per-cell active-learning plan.

    Inputs:
      prior_cells_json   The LATEST prior round's stage05/cells.json.
                         Its `cells[].label` are the leaves of the
                         split tree as of the prior round.
      shapes_per_cell    Budget per leaf sub-cell.
      split_tree         Cumulative split tree (all prior rounds'
                         stage04b/splits.json merged). Any cell in the
                         prior leaves that this tree splits is EXPANDED
                         into its leaves and each child gets the full
                         per-cell budget (=> 2x the data on the bad
                         cell, half-and-half between the two halves).

    Returns plan[leaf_label] = {has_prior_model, sel_eff_prev_round,
                                n_target}."""
    cells_json = read_json(prior_cells_json)
    prior_by_cell: Dict[str, float] = {}
    for c in cells_json.get("cells", []):
        prior_by_cell[c["label"]] = float(c.get("best_sel_eff", float("nan")))

    plan: Dict[str, Dict[str, Any]] = {}

    # Step A: every base 96-cell that has no prior model -> budget 0.
    # (Only base cells appear if no prior round has run; sub-cell
    # labels are introduced by split_tree.)
    for lbl in all_cell_labels():
        if lbl not in prior_by_cell and lbl not in split_tree:
            plan[lbl] = {
                "sel_eff_prev_round": None,
                "n_target": 0,
                "has_prior_model": False,
            }

    # Step B: walk every prior-leaf and expand through the cumulative
    # tree. Each FINAL leaf gets one budget unit.
    for parent in prior_by_cell:
        for leaf in sc.leaves_under(parent, split_tree):
            plan[leaf] = {
                "sel_eff_prev_round": (
                    prior_by_cell[parent] if leaf == parent else None
                ),
                "n_target": shapes_per_cell,
                "has_prior_model": True,
                "parent_cell": parent if leaf != parent else None,
            }
    return plan


# ── history harvesting (prior-shapes dedup) ──────────────────────────────────


def _harvest_enriched_dir(path: Path) -> List[Tuple[int, int, int, int]]:
    seen: set = set()
    if not path.exists():
        return []
    for fn in sorted(os.listdir(path)):
        if not (fn.startswith("chunk_") and fn.endswith(".csv")):
            continue
        with (path / fn).open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    seen.add(
                        (
                            int(row["m"]),
                            int(row["n"]),
                            int(row["k"]),
                            int(row["batch_count"]),
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
    return list(seen)


_ROUND_RE = re.compile(r"/(round_\d+)/")


def _round_tag(p: Path) -> str:
    """Extract the `round_<idx>` segment from a path inside a pipeline_v3
    run dir; returns `?` if not found."""
    s = str(p.resolve())
    m = _ROUND_RE.search(s)
    return m.group(1) if m else "?"


def collect_history_hashes(
    prior_shapes_yamls: List[Path], prior_enriched_dirs: List[Path], quiet: bool
) -> set:
    history: set = set()
    for p in prior_shapes_yamls:
        n0 = len(history)
        for s in parse_shapes_yaml(p):
            history.add(shape_hash(*s))
        if not quiet:
            ui.info(
                "dedup",
                f"prior {_round_tag(p)} yaml {p.name}: "
                f"+{len(history)-n0} shapes "
                f"(cumulative {len(history):,})",
            )
    for d in prior_enriched_dirs:
        n0 = len(history)
        for s in _harvest_enriched_dir(d):
            history.add(shape_hash(*s))
        if not quiet:
            ui.info(
                "dedup",
                f"prior {_round_tag(d)} enriched {d.name}: "
                f"+{len(history)-n0} shapes "
                f"(cumulative {len(history):,})",
            )
    return history


# ── per-cell sampler ─────────────────────────────────────────────────────────


def sample_shapes(
    per_cell_target: Dict[str, int],
    rng: random.Random,
    history_hashes: set,
    n_strata: int = 1,
    bpe: int = 2,
    min_per_cell: int = 0,
    arch: Optional[str] = None,
    scale_a: int = 0,
    scale_b: int = 0,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, int], int]:
    """For each cell, synthesize up to its target. Cells may be base
    96-grid labels OR sub-cell labels carrying `#<axis><op><threshold>`
    segments; sub-cells synthesize within the tighter axis box. History
    dedup is against the (M,N,K,batch) hash set."""
    final: List[Tuple[int, int, int, int]] = []
    actual_per_cell: Dict[str, int] = {}
    seen: set = set()
    n_hist_drop = 0
    for cell, target in per_cell_target.items():
        if target <= 0:
            actual_per_cell[cell] = 0
            continue
        base = sc.base_cell(cell)
        axis_bounds = None
        if cell != base:
            axis_bounds = {}
            # Tier extents for each axis -- restrict_axis_bounds tightens
            # them with the split segments stamped into the label.
            from lib.grid import TIER_RANGES_K, TIER_RANGES_MN

            mt, nt, kt, _bt = base.split("|")
            extents = {
                "M": TIER_RANGES_MN[mt],
                "N": TIER_RANGES_MN[nt],
                "K": TIER_RANGES_K[kt],
            }
            for ax, (lo, hi) in extents.items():
                axis_bounds[ax] = sc.restrict_axis_bounds(cell, ax, lo, hi)
        pool = synthesize_cell_shapes(
            base,
            target * 3,
            rng,
            bpe=bpe,
            axis_bounds=axis_bounds,
            n_strata=n_strata,
            arch=arch,
            scale_a=scale_a,
            scale_b=scale_b,
        )
        kept: List[Tuple[int, int, int, int]] = []
        for s in pool:
            if len(kept) >= target:
                break
            if s in seen:
                continue
            h = shape_hash(*s)
            if h in history_hashes:
                n_hist_drop += 1
                continue
            seen.add(s)
            kept.append(s)
        # Coverage guarantee: a (split) leaf MUST get >= min_per_cell shapes
        # so stage05 trains it -- otherwise it becomes an UNTRAINED leaf in
        # the deployed split tree (an orphaned GEMM region with no matching
        # cell). In later rounds a narrow leaf's region is largely exhausted
        # by prior-round history, so the new-only pass above falls short.
        # Top up by REUSING history shapes (re-benched this round -- valid
        # training data) from a larger pool. If the region genuinely cannot
        # yield min_per_cell distinct shapes, warn: that leaf should not have
        # been split.
        if min_per_cell > 0 and len(kept) < min_per_cell:
            big = synthesize_cell_shapes(
                base,
                max(min_per_cell * 6, target * 3),
                rng,
                bpe=bpe,
                axis_bounds=axis_bounds,
                n_strata=n_strata,
                arch=arch,
                scale_a=scale_a,
                scale_b=scale_b,
            )
            for s in big:
                if len(kept) >= min_per_cell:
                    break
                if s in seen:
                    continue
                seen.add(s)
                kept.append(s)  # history reuse permitted for coverage
            if len(kept) < min_per_cell:
                ui.warn(
                    "coverage",
                    f"leaf {cell}: only {len(kept)} distinct shapes "
                    f"(< min {min_per_cell}); region too small to cover "
                    f"-- this leaf may stay undertrained.",
                )
        final.extend(kept)
        actual_per_cell[cell] = len(kept)
    return final, actual_per_cell, n_hist_drop


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage01 -- 96-cell grid OOB / active-learning shape " "generator."
    )
    ap.add_argument("--mode", choices=("initial", "active"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config-yaml",
        type=Path,
        required=True,
        help="OOB configuration yaml describing dtype / "
        "layout / runtime (e.g. configs/"
        'gfx950_bbs_tn.yaml). "cell" elsewhere in '
        "this codebase ALWAYS means a 96-grid cell -- "
        "the config-level identifier is `config_id`.",
    )
    ap.add_argument("--seed", type=int, default=42)

    # initial-mode
    ap.add_argument(
        "--target-per-cell",
        type=int,
        default=1600,
        help="(initial mode) shapes per cell. 96 * 1600 ~= "
        "154k. The orchestrator overrides this from "
        "the config's `seed.target_per_cell`.",
    )
    # active-learning-mode
    ap.add_argument(
        "--prior-round-dir",
        "--prior-run-dir",
        type=Path,
        action="append",
        default=None,
        dest="prior_round_dir",
        help="(active mode) repeatable; each entry MUST "
        "point at a `<run>/round_<i>/` ROUND directory "
        "(NOT a top-level run dir). We auto-walk "
        "`<round>/stage01/shapes.yaml` for history "
        "dedup, `<round>/stage04/` for enriched-row "
        "dedup, and `<round>/stage05/cells.json` for "
        "the per-cell shape plan (the LATEST prior "
        "round's cells.json wins). "
        "(`--prior-run-dir` is kept as a deprecated "
        "alias.)",
    )
    ap.add_argument(
        "--prior-cells",
        type=Path,
        default=None,
        help="(active mode) explicit prior cells.json path. "
        "Overrides auto-discovery from "
        "--prior-round-dir.",
    )
    ap.add_argument(
        "--shapes-per-cell",
        type=int,
        default=150,
        help="(active mode) shapes per cell with a prior model. "
        "Same N across all populated cells.",
    )
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=0,
        help="(active mode) guarantee each leaf gets at least "
        "this many shapes so stage05 trains it -- prevents "
        "untrained/orphan leaves in the deployed split tree. "
        "Tops up from history (re-benched) when a narrow "
        "leaf's new-shape pool is exhausted. 0 = legacy off.",
    )
    ap.add_argument(
        "--n-strata",
        type=int,
        default=1,
        help="per-axis log-equal strata for within-cell "
        "corner coverage. 1 = pure log-uniform on the "
        "whole cell box (legacy). N > 1 carves each "
        "of M / N / K into N log-equal sub-ranges and "
        "samples ~target/N**3 shapes per sub-box, so "
        "no octant goes uncovered.",
    )
    # Explicit history sources -- additive on top of --prior-round-dir.
    ap.add_argument(
        "--prior-shapes-yaml",
        type=Path,
        action="append",
        default=None,
        help="repeatable; extra prior shape yamls.",
    )
    ap.add_argument(
        "--prior-enriched-csv-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable; extra prior enriched dirs.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f)

    if not args.quiet:
        ui.banner(f"stage01 OOB shape generator  (mode={args.mode})")
        ui.info("cfg", f"out_dir        : {args.out_dir}")
        ui.info("cfg", f"config_yaml    : {args.config_yaml}")
        ui.info("cfg", f"seed           : {args.seed}")
        ui.banner("Stage 1/3  Per-cell budget", ui.C.BLUE)

    # Build per-cell budget.
    per_cell: Dict[str, int]
    plan: Optional[Dict[str, Dict[str, Any]]] = None
    if args.mode == "initial":
        per_cell = _build_initial_budget(args.target_per_cell)
        if not args.quiet:
            ui.ok(
                "budget",
                f"target_per_cell={args.target_per_cell}  "
                f"total_target={sum(per_cell.values()):,}  "
                f"cells={len(per_cell)}",
            )
    else:
        # Auto-discover prior shapes + cells.json from --prior-round-dir.
        # pipeline_v3 layout: every round owns its own subfolder under
        # the run dir, and every stage owns a subfolder of that. We
        # walk:
        #   <round>/stage01/shapes.yaml     (history dedup source)
        #   <round>/stage04/                (enriched chunks)
        #   <round>/stage05/cells.json      (per-cell sel_eff -> plan)
        # An explicit --prior-cells (and --prior-shapes-yaml /
        # --prior-enriched-csv-dir) overrides / augments auto-discovery.
        prior_round_dirs = list(args.prior_round_dir or [])
        auto_shapes: List[Path] = []
        auto_enriched: List[Path] = []
        auto_cells_json: Optional[Path] = None
        for rd in prior_round_dirs:
            sy = rd / "stage01" / "shapes.yaml"
            if sy.exists():
                auto_shapes.append(sy)
            en = rd / "stage04"
            if en.exists() and any(
                p.name.startswith("chunk_") for p in en.iterdir() if p.is_file()
            ):
                auto_enriched.append(en)
            cj = rd / "stage05" / "cells.json"
            if cj.exists():
                # The LATEST prior round's cells.json wins (we iterate in
                # order; later entries override earlier).
                auto_cells_json = cj
        cells_json_path = args.prior_cells or auto_cells_json
        if cells_json_path is None or not cells_json_path.exists():
            ui.err(
                "err",
                "--mode active requires either --prior-cells "
                "OR --prior-round-dir <round>/ (each round dir must "
                "have stage05/cells.json). got prior_cells="
                f"{args.prior_cells} prior_round_dirs="
                f"{prior_round_dirs}",
            )
            return 1
        # Cumulative split tree (every prior round's stage04b/splits.json).
        split_tree = sc.load_cumulative_split_tree(prior_round_dirs)
        plan = _build_active_plan(cells_json_path, args.shapes_per_cell, split_tree)
        per_cell = {cell: int(info["n_target"]) for cell, info in plan.items()}
        # Merge auto-discovered into explicit history sources.
        args.prior_shapes_yaml = list(args.prior_shapes_yaml or []) + auto_shapes
        args.prior_enriched_csv_dir = (
            list(args.prior_enriched_csv_dir or []) + auto_enriched
        )
        if not args.quiet:
            ui.ok(
                "auto",
                f"discovered from --prior-round-dir: "
                f"{len(auto_shapes)} shapes.yaml, "
                f"{len(auto_enriched)} enriched dirs, "
                f"cells.json={cells_json_path}",
            )
            if split_tree:
                ui.ok(
                    "splits",
                    f"cumulative split tree: "
                    f"{len(split_tree)} cells previously split "
                    f"-> {sum(len(sc.leaves_under(c, split_tree))-1 for c in split_tree)}"
                    f" extra sub-cells in plan",
                )
        if not args.quiet:
            n_with = sum(1 for p in plan.values() if p["has_prior_model"])
            n_without = sum(1 for p in plan.values() if not p["has_prior_model"])
            ui.ok(
                "plan",
                f"populated_cells={n_with}  no_prior_model="
                f"{n_without}  shapes_per_cell={args.shapes_per_cell}  "
                f"total_target={sum(per_cell.values()):,}",
            )
            ui.info(
                "plan",
                "ALL populated cells receive the same shape "
                "budget. Pass/retrain decisions come AFTER bench, from "
                "the MEASURED sel_eff (stage04b).",
            )
            # Quick prior sel_eff distribution -- informational only.
            wp = [
                (c, p["sel_eff_prev_round"])
                for c, p in plan.items()
                if p["has_prior_model"] and p["sel_eff_prev_round"] is not None
            ]
            wp.sort(key=lambda x: x[1])
            if wp:
                ui.info("plan", "prior sel_eff worst 5 (informational):")
                for c, se in wp[:5]:
                    ui.info("plan", f"  {c:35s}  prev={se*100:5.2f}%")

    # History dedup.
    if not args.quiet:
        ui.banner("Stage 2/3  Synthesize + filter + dedup", ui.C.BLUE)
    history_hashes = collect_history_hashes(
        args.prior_shapes_yaml or [],
        args.prior_enriched_csv_dir or [],
        args.quiet,
    )
    if not args.quiet and history_hashes:
        ui.ok("dedup", f"history hashes loaded: {len(history_hashes):,}")
    t0 = time.time()
    # Bytes-per-element drives the HBM/OOM feasibility guard. It MUST
    # follow the storage dtype of A: f8_r=1, bf16_r=2, f32_r/xf32=4.
    # Hardwiring bpe=2 (bf16) under-counts xf32 footprint by 2x and would
    # let the synthesizer emit envelope shapes that OOM hipblaslt-bench.
    hbl_cfg = cfg.get("hipblaslt") or {}
    a_type = hbl_cfg.get("a_type", "bf16_r")
    bpe = _BPE.get(a_type, 2)
    # The HBM budget is per-arch (see `hbm_safe_bytes` in lib/shapes.py); the
    # scale MODES add the MX block-scale tensors to the footprint (only
    # for block modes 3/4/5/1001 -- Scalar/Vector/none change nothing).
    arch = str(cfg.get("arch", "gfx950"))
    scale_a = int(hbl_cfg.get("scaleA", 0) or 0)
    scale_b = int(hbl_cfg.get("scaleB", 0) or 0)
    if not args.quiet:
        ui.info(
            "bpe", f"a_type={a_type} -> bytes/elem={bpe} " f"(feasibility/OOM guard)"
        )
        ui.info(
            "bpe",
            f"arch={arch} -> hbm_budget="
            f"{hbm_safe_bytes(arch) / 1024**3:.0f} GiB  "
            f"scaleA/B={scale_a}/{scale_b}",
        )
    sampled, actual_per_cell, n_hist_drop = sample_shapes(
        per_cell,
        rng,
        history_hashes,
        n_strata=args.n_strata,
        bpe=bpe,
        min_per_cell=int(getattr(args, "min_cell_gemms", 0) or 0),
        arch=arch,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    if not args.quiet:
        ui.ok(
            "sample",
            f"sampled={len(sampled):,}  "
            f"history_dedup_dropped={n_hist_drop:,}  "
            f"({time.time()-t0:.1f}s)",
        )
        n_populated = sum(1 for v in actual_per_cell.values() if v > 0)
        ui.ok("sample", f"populated cells: {n_populated}/{len(per_cell)}")
        gaps = [
            (c, t, actual_per_cell.get(c, 0))
            for c, t in per_cell.items()
            if t > 0 and actual_per_cell.get(c, 0) < t
        ]
        if gaps:
            zeros = sum(1 for _, _, a in gaps if a == 0)
            ui.warn(
                "sample",
                f"{len(gaps)} cells under target "
                f"({zeros} produced zero shapes -- rotating-512 filter)",
            )

    # Write outputs.
    if not args.quiet:
        ui.banner("Stage 3/3  Write outputs", ui.C.BLUE)
    out_yaml = args.out_dir / "shapes.yaml"
    shape_objs = [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in sampled]
    # SHUFFLE before writing. stage03 slices this file into CONTIGUOUS blocks,
    # one process each, and cannot finish before its slowest block. Emitting in
    # cell order therefore concentrates every property of a cell into one or two
    # blocks -- a block drawn entirely from a fault-prone region pays a full
    # kernel-library reload per fault and gates the whole stage while the
    # other GPUs sit idle.
    #
    # Shuffling makes every block a random sample of the space, so risk -- and
    # cost, and any per-cell bench-time skew -- spreads evenly over the GPUs
    # instead of landing on one. It loses NOTHING: block membership is
    # arbitrary, stage04 keys rows by shape, and the seed keeps it reproducible.
    _shuf = random.Random(int(args.seed) ^ 0x5EED)
    _shuf.shuffle(shape_objs)
    if not args.quiet:
        ui.info(
            "write",
            f"shuffled {len(shape_objs):,} shapes (seed "
            f"{args.seed}) so stage03's contiguous blocks are random "
            f"samples, not whole cells",
        )
    n_written = write_bench_yaml(shape_objs, cfg, out_yaml)
    if not args.quiet:
        ui.ok("write", f"shapes.yaml: {n_written:,} lines -> {out_yaml}")

    write_json(
        args.out_dir / "categorization.json",
        {
            "mode": args.mode,
            "seed": args.seed,
            "config_yaml": str(args.config_yaml),
            "n_final": len(sampled),
            "n_history_dropped": n_hist_drop,
            "target_per_cell": per_cell,
            "actual_per_cell": actual_per_cell,
            "target_per_cell_default": (
                args.target_per_cell if args.mode == "initial" else args.shapes_per_cell
            ),
        },
    )
    if not args.quiet:
        ui.ok("write", f"categorization.json")

    if args.mode == "active" and plan is not None:
        write_json(
            args.out_dir / "per_cell_plan.json",
            {
                "shapes_per_cell": args.shapes_per_cell,
                "prior_cells_json": str(cells_json_path),
                "prior_round_dirs": [str(p) for p in (args.prior_round_dir or [])],
                "prior_shapes_yaml": [str(p) for p in (args.prior_shapes_yaml or [])],
                "prior_enriched_csv_dir": [
                    str(p) for p in (args.prior_enriched_csv_dir or [])
                ],
                "n_history_hashes": len(history_hashes),
                "cells": {
                    cell: {**info, "n_actual": actual_per_cell.get(cell, 0)}
                    for cell, info in plan.items()
                },
            },
        )
        if not args.quiet:
            ui.ok("write", "per_cell_plan.json")

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok("done", f"total shapes  : {len(sampled):,}")
        ui.ok("done", f"output dir    : {args.out_dir}")
        ui.info(
            "next",
            "stage02_probe.py to calibrate iters per shape, "
            "then stage03_load_balance_offline_tuning.py for bench.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
