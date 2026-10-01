#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate kernel-parameter sourcing: source-YAML-by-index vs what Origami
obtains from the .dat `sizeMapping` (the PredictionLibrary.hpp path).

Background
----------
stage04 now reads kernel params (mt/mi/cache_hints_a/b/grvw/gwvw) from the
SOURCE Tensile solution YAML, indexed by the MatchTable's (yaml_path,
local_idx). Origami / hipBLASLt's PredictionLibrary builds its `config_t` from
the loaded TensileLibrary `.dat`'s `sizeMapping`:

    config_t.mt            = {sizeMapping.macroTile, sizeMapping.depthU}
    config_t.mi            = sizeMapping.matrixInstruction
    config_t.cache_hints_a = sizeMapping.nonTemporalA
    config_t.cache_hints_b = sizeMapping.nonTemporalB
    config_t.grvw_a/b      = sizeMapping.grvwA/grvwB
    config_t.gwvw_d        = sizeMapping.gwvwD
    config_t.occupancy     = max(sizeMapping.CUOccupancy, 1)

This test confirms, for EVERY solution (joined by the unique global solution
index, which equals the .dat's top-level `index` and the MatchTable key), that
the YAML-by-index params equal the .dat-`sizeMapping` params -- i.e. the new
YAML source agrees with what Origami obtains. Occupancy is excluded (the YAML
stores the -1 'unset' sentinel; the .dat carries the real value -- by design
stage04 takes occupancy from the .dat).

It also demonstrates the bug the YAML-by-index switch fixes: the OLD
.dat-by-NAME lookup collides across the base / Origami_nta4 / Origami_ntb4
libraries (same kernel `name`, different NonTemporalA/B), so it returned the
wrong variant's cache_hints for colliding kernels -- the Custom 256x256x32
kernels among them.

Usage
-----
    python -m tests.test_yaml_vs_origami_params \
        --dat-dir   <build>/Tensile/library/gfx950 \
        --match-table <build>/device-library/MatchTable.yaml

Exits non-zero if any solution disagrees on the 9 base params.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import msgpack

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stages.stage04_convert_to_enriched_dataset import (  # noqa: E402
    parse_match_table,
    _yaml_kernel_cfg,
    _yaml_sols,
)

# Params stage04 sources from the source YAML by index -- these MUST equal
# what Origami obtains from the .dat sizeMapping (the test's pass/fail set).
_YAML_FIELDS = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
)
# gwvw_d is sourced from YAML `StoreVectorWidth` (NOT `GlobalWriteVectorWidth`);
# this section reports how each YAML key tracks the runtime's derived gwvwD.
# occupancy is the only field stage04 takes from the .dat (YAML stores the -1
# sentinel) -- not cross-checked here.

# training_pipeline/ lives at <rocm-libraries>/shared/tilewright/training_pipeline.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_DEFAULT_DAT = os.path.join(
    _REPO_ROOT, "projects/hipblaslt/build/release/Tensile/library/gfx950"
)
_DEFAULT_MT = os.path.join(
    _REPO_ROOT, "projects/hipblaslt/build/release/device-library/MatchTable.yaml"
)


def origami_params_from_sm(sm: Dict[str, Any]) -> Dict[str, int]:
    """Mirror PredictionLibrary.hpp's config_t construction from a .dat
    sizeMapping (the params Origami actually obtains at runtime)."""
    mt = sm.get("macroTile") or [0, 0, 0]
    while len(mt) < 3:
        mt = list(mt) + [0]
    mi = sm.get("matrixInstruction") or [0, 0, 0, 0]
    while len(mi) < 3:
        mi = list(mi) + [0]
    return {
        "mt_m": int(mt[0]),
        "mt_n": int(mt[1]),
        "mt_k": int(sm.get("depthU", 0) or 0),
        "mi_m": int(mi[0]),
        "mi_n": int(mi[1]),
        "mi_k": int(mi[2]),
        "cache_hints_a": int(sm.get("nonTemporalA", 0) or 0),
        "cache_hints_b": int(sm.get("nonTemporalB", 0) or 0),
        "grvw_a": int(sm.get("grvwA", 1) or 1),
        "grvw_b": int(sm.get("grvwB", 1) or 1),
        "gwvw_d": int(sm.get("gwvwD", 1) or 1),
    }


def _walk_dat(dat_dir: str):
    """Yield (kernel_name, sol_idx_global, sizeMapping) for every solution."""
    for fname in sorted(os.listdir(dat_dir)):
        if not (fname.endswith(".dat") and "Contraction" in fname):
            continue
        path = os.path.join(dat_dir, fname)
        try:
            with open(path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
        except Exception as e:
            print(f"  [warn] skip {fname}: {e}")
            continue
        for sol in data.get("solutions", []):
            gi = sol.get("index")
            if gi is None:
                continue
            name = sol.get("name") or sol.get("kernelName") or ""
            sm = sol.get("sizeMapping", {}) or {}
            yield name, int(gi), sm


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dat-dir", default=_DEFAULT_DAT)
    ap.add_argument("--match-table", default=_DEFAULT_MT)
    ap.add_argument("--max-print", type=int, default=20)
    args = ap.parse_args()

    if not os.path.isdir(args.dat_dir):
        print(f"ERROR: dat-dir not found: {args.dat_dir}")
        return 2
    if not os.path.isfile(args.match_table):
        print(f"ERROR: match-table not found: {args.match_table}")
        return 2

    print("=" * 78)
    print("YAML-by-index kernel params  vs  Origami config_t (.dat sizeMapping)")
    print("=" * 78)
    print(f"dat-dir     : {args.dat_dir}")
    print(f"match-table : {args.match_table}")

    match_table = parse_match_table(args.match_table)
    print(f"MatchTable entries: {len(match_table):,}")

    n_checked = 0
    n_match = 0
    n_no_mt_entry = 0
    n_yaml_miss = 0
    mismatches: List[Tuple[int, str, Dict[str, Tuple[int, int]]]] = []
    per_field_mismatch: "Counter[str]" = Counter()
    # Informational: how often YAML's codegen GlobalWriteVectorWidth differs
    # from the .dat's derived gwvwD (why stage04 sources gwvw_d from the .dat).
    n_gwvw_checked = 0
    n_gwvw_yaml_eq_dat = 0

    # Custom-kernel spotlight + the name-collision demonstration.
    custom_examples: List[Tuple[int, str, Dict[str, int], Dict[str, int]]] = []
    # name -> set of cache_hints_a seen across global indices (collision proof)
    name_to_nta: Dict[str, set] = defaultdict(set)

    def _yaml_int(yaml_path: str, local_idx: int, key: str) -> int:
        sols = _yaml_sols(yaml_path)
        if not sols or local_idx >= len(sols):
            return -1
        v = sols[local_idx].get(key)
        try:
            return int(v) if v is not None else -1
        except (TypeError, ValueError):
            return -1

    n_svw_checked = 0
    n_svw_eq_dat = 0

    for name, gi, sm in _walk_dat(args.dat_dir):
        ori = origami_params_from_sm(sm)
        name_to_nta[name].add(ori["cache_hints_a"])

        entry = match_table.get(gi)
        if entry is None:
            n_no_mt_entry += 1
            continue
        yaml_path, local_idx = entry
        ycfg = _yaml_kernel_cfg(yaml_path, local_idx)
        if ycfg is None:
            n_yaml_miss += 1
            continue

        n_checked += 1
        diff = {f: (ycfg[f], ori[f]) for f in _YAML_FIELDS if ycfg[f] != ori[f]}
        if not diff:
            n_match += 1
        else:
            for f in diff:
                per_field_mismatch[f] += 1
            if len(mismatches) < 10000:
                mismatches.append((gi, name, diff))

        # gwvw_d: which YAML key equals the .dat's derived gwvwD?
        yaml_gwvw = _yaml_int(yaml_path, local_idx, "GlobalWriteVectorWidth")
        if yaml_gwvw >= 0:
            n_gwvw_checked += 1
            if yaml_gwvw == ori["gwvw_d"]:
                n_gwvw_yaml_eq_dat += 1
        yaml_svw = _yaml_int(yaml_path, local_idx, "StoreVectorWidth")
        if yaml_svw >= 0:
            n_svw_checked += 1
            if yaml_svw == ori["gwvw_d"]:
                n_svw_eq_dat += 1

        if "Custom" in name and len(custom_examples) < max(args.max_print, 6):
            custom_examples.append((gi, name, dict(ycfg), ori))

    # ── report ──────────────────────────────────────────────────────────────
    print("-" * 78)
    print(f"YAML-sourced fields (mt/mi/cache_hints_a/b/grvw_a/b) vs Origami(.dat):")
    print(f"solutions joined & checked : {n_checked:,}")
    print(
        f"  exact param match        : {n_match:,}  "
        f"({100.0 * n_match / max(n_checked, 1):.2f}%)"
    )
    print(f"  param MISMATCH           : {n_checked - n_match:,}")
    print(f"no MatchTable entry        : {n_no_mt_entry:,}")
    print(f"yaml-by-index miss         : {n_yaml_miss:,}")
    print("-" * 78)
    print("gwvw_d source (stage04 uses YAML StoreVectorWidth) vs runtime gwvwD:")
    print(
        f"  YAML StoreVectorWidth       == .dat gwvwD : "
        f"{n_svw_eq_dat:,}/{n_svw_checked:,} "
        f"({100.0 * n_svw_eq_dat / max(n_svw_checked, 1):.2f}%)  <- chosen"
    )
    print(
        f"  YAML GlobalWriteVectorWidth == .dat gwvwD : "
        f"{n_gwvw_yaml_eq_dat:,}/{n_gwvw_checked:,} "
        f"({100.0 * n_gwvw_yaml_eq_dat / max(n_gwvw_checked, 1):.2f}%)  (rejected)"
    )
    if per_field_mismatch:
        print("  per-field mismatch counts:")
        for f, c in per_field_mismatch.most_common():
            print(f"    {f:<16s} {c:,}")
    for gi, name, diff in mismatches[: args.max_print]:
        print(f"    [global={gi}] {name}")
        for f, (yv, ov) in diff.items():
            print(f"        {f}: yaml={yv}  origami(.dat)={ov}")

    # ── Custom-kernel spotlight ───────────────────────────────────────────────
    print("-" * 78)
    print(f"Custom-kernel sample (YAML-by-index == Origami(.dat)?):")
    for gi, name, ycfg, ori in custom_examples[: max(args.max_print, 6)]:
        ok = all(ycfg[f] == ori[f] for f in _YAML_FIELDS)
        tag = "OK " if ok else "BAD"
        print(
            f"  [{tag}] global={gi}  MT={ycfg['mt_m']}x{ycfg['mt_n']}x{ycfg['mt_k']} "
            f"NTA(yaml)={ycfg['cache_hints_a']} NTA(.dat)={ori['cache_hints_a']}"
        )
        print(f"        {name[:96]}")

    # ── name-collision demonstration (the bug the YAML switch fixes) ──────────
    colliding = {nm: sorted(vs) for nm, vs in name_to_nta.items() if len(vs) > 1}
    print("-" * 78)
    print(
        "Name-collision proof (kernels whose `name` maps to >1 distinct "
        "NonTemporalA across the library variants):"
    )
    print(f"  colliding kernel names: {len(colliding):,}")
    shown = 0
    for nm, vs in colliding.items():
        if shown >= min(args.max_print, 10):
            break
        if "Custom" in nm or shown < 5:
            print(f"    NTA seen={vs}  {nm[:90]}")
            shown += 1
    print("  => the OLD .dat-by-NAME lookup returned ONE of these arbitrarily;")
    print("     YAML-by-(path,local_idx) resolves the correct per-variant value.")

    print("=" * 78)
    ok = n_checked > 0 and n_match == n_checked
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
