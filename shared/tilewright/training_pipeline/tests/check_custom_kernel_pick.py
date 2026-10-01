#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check whether the Custom kernels are correctly enriched and actually picked.

Part 1 (data): identify Custom-kernel global indices from the .dat for the
tf32_nt (f32/f32, xf32 compute, N/T) layout, then scan the enriched CSV for
GEMMs whose empirical winner is a Custom kernel -- reporting their MT and the
(now-fixed) cache_hints_a so we can confirm enrichment labels them correctly.

Part 2 (pick): for those GEMMs, rebuild the candidate set from the CSV tested
rows, load the trained model bundle, and run the DEPLOYED ranking via the
tilewright C++ binding (route -> is_feasible -> score). Report how often the model
picks the Custom kernel that actually won, and the realized sel_eff on that
slice.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import msgpack

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# training_pipeline/ lives at <rocm-libraries>/shared/tilewright/training_pipeline.
_REPO_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "..", ".."))
_DEF_DAT = os.path.join(
    _REPO_ROOT, "projects/hipblaslt/build/release/Tensile/library/gfx950"
)


def custom_globals(dat_dir: str) -> Dict[int, Dict[str, Any]]:
    """global idx -> {name, mt, nta} for every Custom kernel in the
    S_MX (xf32) Ailk_Bjlk (N/T) contraction library."""
    out: Dict[int, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(dat_dir)):
        if not (fname.endswith(".dat") and "Contraction" in fname):
            continue
        # tf32_nt: f32 in/out, xf32 compute (SS..MX), N/T layout (Ailk_Bjlk).
        if "_SS_SS_HA_Bias_SAV_MX_" not in fname or "Ailk_Bjlk" not in fname:
            continue
        with open(os.path.join(dat_dir, fname), "rb") as f:
            data = msgpack.unpack(f, raw=False)
        for sol in data.get("solutions", []):
            name = sol.get("name") or ""
            if not name.startswith("Custom"):
                continue
            gi = sol.get("index")
            if gi is None:
                continue
            sm = sol.get("sizeMapping", {}) or {}
            mt = sm.get("macroTile") or [0, 0, 0]
            out[int(gi)] = {
                "name": name,
                "mt": (int(mt[0]), int(mt[1]), int(sm.get("depthU", 0) or 0)),
                "nta": int(sm.get("nonTemporalA", 0) or 0),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched-dir", default="/tmp/reenrich_r0")
    ap.add_argument("--dat-dir", default=_DEF_DAT)
    ap.add_argument(
        "--max-gemms-pick",
        type=int,
        default=200,
        help="cap GEMMs sent through the model pick check",
    )
    ap.add_argument(
        "--bin",
        default=None,
        help="deployed tilewright_weights.bin; enables the Part-2 " "C++ pick check",
    )
    args = ap.parse_args()

    cg = custom_globals(args.dat_dir)
    print(f"Custom kernels in tf32_nt (S_MX, N/T) library: {len(cg)}")
    by_mt: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for gi, d in cg.items():
        by_mt[d["mt"]].append(gi)
    for mt, gis in sorted(by_mt.items()):
        ntas = sorted({cg[g]["nta"] for g in gis})
        print(f"  MT={mt[0]}x{mt[1]}x{mt[2]}  count={len(gis)}  NTA values={ntas}")

    custom_set = set(cg.keys())

    # ── scan enriched CSV: group rows per GEMM, find empirical winner ─────────
    chunks = sorted(glob.glob(os.path.join(args.enriched_dir, "*.csv")))
    # group key: (m,n,k,batch,transA,transB)
    gemms: Dict[Tuple, Dict[str, Any]] = {}
    n_rows = 0
    for cf in chunks:
        with open(cf) as fh:
            for row in csv.DictReader(fh):
                n_rows += 1
                key = (
                    row["m"],
                    row["n"],
                    row["k"],
                    row["batch_count"],
                    row["transA"],
                    row["transB"],
                )
                g = gemms.setdefault(key, {"winner": None, "win_us": None, "cands": []})
                sig = row.get("sol_idx_global", "")
                us = row.get("us", "")
                is_skip = str(row.get("is_skip", "")).strip() == "True"
                if not is_skip and sig not in ("", None):
                    try:
                        usf = float(us)
                    except (TypeError, ValueError):
                        usf = -1.0
                    if usf > 0:
                        g["cands"].append((int(sig), usf, row))
                        if g["win_us"] is None or usf < g["win_us"]:
                            g["win_us"] = usf
                            g["winner"] = int(sig)

    n_gemms = len(gemms)
    custom_winner_gemms = [
        (k, g) for k, g in gemms.items() if g["winner"] in custom_set
    ]
    print("-" * 70)
    print(f"enriched rows: {n_rows:,}   GEMMs: {n_gemms:,}")
    print(
        f"GEMMs whose EMPIRICAL winner is a Custom kernel: "
        f"{len(custom_winner_gemms):,} "
        f"({100.0*len(custom_winner_gemms)/max(n_gemms,1):.1f}%)"
    )
    # winner MT / nta breakdown
    wmt: Dict[Tuple[int, int, int], int] = defaultdict(int)
    wnta: Dict[int, int] = defaultdict(int)
    for _, g in custom_winner_gemms:
        wmt[cg[g["winner"]]["mt"]] += 1
        wnta[cg[g["winner"]]["nta"]] += 1
    print(
        "  winner Custom MT breakdown:",
        {f"{m[0]}x{m[1]}x{m[2]}": c for m, c in sorted(wmt.items())},
    )
    print("  winner Custom NTA breakdown:", dict(sorted(wnta.items())))

    # Also: how often is a Custom kernel even PRESENT as a candidate?
    present = sum(
        1 for _, g in gemms.items() if any(s in custom_set for s, _, _ in g["cands"])
    )
    print(
        f"GEMMs with a Custom kernel among candidates: {present:,} "
        f"({100.0*present/max(n_gemms,1):.1f}%)"
    )

    # ── Part 2: does the DEPLOYED model pick the Custom kernel? ───────────────
    if not args.bin:
        return 0
    import tilewright  # noqa: E402

    if not tilewright.load_weights(args.bin):
        print(f"ERROR: tilewright.load_weights failed: {args.bin}")
        return 2
    hw = tilewright.Hardware()
    hw.N_CU = 256
    hw.lds_capacity = 163840
    hw.L2_capacity = 4194304
    hw.parallel_mi_cu = 4
    hw.mem_bw_per_wg_coefficients = (-1.3e-05, 0.00707, 0.027355)

    _DT = {
        "f32_r": tilewright.DataType.Float,
        "bf16_r": tilewright.DataType.BFloat16,
        "f16_r": tilewright.DataType.Half,
        "f8_r": tilewright.DataType.Float8,
    }

    def _dt(s: str):
        return _DT.get((s or "").strip().lower(), tilewright.DataType.Float)

    def _mk_problem(row) -> Any:
        p = tilewright.Problem()
        d = tilewright.Dim3()
        d.m = int(row["m"])
        d.n = int(row["n"])
        d.k = int(row["k"])
        p.size = d
        p.batch = max(int(row["batch_count"] or 1), 1)
        p.a_transpose = (
            tilewright.Transpose.T if row["transA"] == "T" else tilewright.Transpose.N
        )
        p.b_transpose = (
            tilewright.Transpose.T if row["transB"] == "T" else tilewright.Transpose.N
        )
        p.a_dtype = _dt(row["a_type"])
        p.b_dtype = _dt(row["b_type"])
        p.c_dtype = _dt(row["d_type"])
        p.d_dtype = _dt(row["d_type"])
        ct = (row["compute_type"] or "").lower()
        p.mi_dtype = (
            tilewright.DataType.XFloat32 if "xf32" in ct else _dt(row["a_type"])
        )
        return p

    def _mk_config(row, idx: int) -> Any:
        c = tilewright.Config()
        mt = tilewright.Dim3()
        mt.m = int(row["mt_m"] or 0)
        mt.n = int(row["mt_n"] or 0)
        mt.k = int(row["mt_k"] or 0)
        mi = tilewright.Dim3()
        mi.m = int(row["mi_m"] or 0)
        mi.n = int(row["mi_n"] or 0)
        mi.k = int(row["mi_k"] or 0)
        c.mt = mt
        c.mi = mi
        c.occupancy = max(int(row["occupancy"] or 1), 1)
        c.cache_hints_a = int(row["cache_hints_a"] or 0)
        c.cache_hints_b = int(row["cache_hints_b"] or 0)
        c.grvw_a = max(int(row["grvw_a"] or 1), 1)
        c.grvw_b = max(int(row["grvw_b"] or 1), 1)
        c.gwvw_d = max(int(row["gwvw_d"] or 1), 1)
        c.index = idx
        return c

    import math

    sample = custom_winner_gemms[: args.max_gemms_pick]
    n = 0
    pick_is_custom = 0
    pick_exact_winner = 0
    custom_survived = 0  # Custom winner candidate kept (scored) by feasibility
    log_ratio_sum = 0.0  # for geomean of winner_us / picked_us
    for key, g in sample:
        cands = g["cands"]
        if not cands:
            continue
        configs = [_mk_config(r, i) for i, (_, _, r) in enumerate(cands)]
        problem = _mk_problem(cands[0][2])
        results = tilewright.rank_configs(problem, hw, configs)
        survivors = [r for r in results if r.scored]
        if not survivors:
            continue
        n += 1
        # did the Custom winner candidate survive feasibility?
        win_sig = g["winner"]
        win_idxs = [i for i, (s, _, _) in enumerate(cands) if s == win_sig]
        surv_idxs = {r.config_index for r in survivors}
        if any(i in surv_idxs for i in win_idxs):
            custom_survived += 1
        top = survivors[0]  # highest score = the pick
        pick_sig, pick_us, _ = cands[top.config_index]
        if pick_sig in custom_set:
            pick_is_custom += 1
        if pick_sig == win_sig:
            pick_exact_winner += 1
        log_ratio_sum += math.log(g["win_us"] / pick_us) if pick_us > 0 else 0.0

    print("-" * 70)
    print(f"DEPLOYED model pick check on Custom-winner GEMMs (n={n}):")
    print(
        f"  Custom winner candidate survived feasibility : "
        f"{custom_survived}/{n} ({100.0*custom_survived/max(n,1):.1f}%)"
    )
    print(
        f"  model picked a Custom kernel                 : "
        f"{pick_is_custom}/{n} ({100.0*pick_is_custom/max(n,1):.1f}%)"
    )
    print(
        f"  model picked the EXACT Custom winner         : "
        f"{pick_exact_winner}/{n} ({100.0*pick_exact_winner/max(n,1):.1f}%)"
    )
    print(
        f"  realized sel_eff on this slice (geomean)     : "
        f"{100.0*math.exp(log_ratio_sum/max(n,1)):.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
