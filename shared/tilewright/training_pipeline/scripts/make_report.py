#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit a single self-contained, portable HTML report for a completed run.

Called by the orchestrator at END-OF-PIPELINE (after validate). Opens in any
browser, no external refs, shareable.

Sections: headline stats (cross-round held-out aggregate), stage 7
(selection-time + parity), stage 8 (selection-efficiency on external
traces), generalization trajectory (stage04b, per round), training-phase
trajectory (stage05, per round), final-round per-cell table, final cell
design (tree), timing/timeline, and the config used.

Usage:  make_report.py <run_dir> [out.html]
Default out: <run_dir>/report.html
"""
import sys, os, re, json, glob, html, math


def _first_pos(*vals):
    """First finite, strictly-positive value among vals (skips None/NaN/<=0)."""
    for v in vals:
        if isinstance(v, (int, float)) and v == v and v > 0:
            return v
    return None


# Canonical 20-category GEMM taxonomy (mirrors geko/gemm_metrics.py CATEGORIES
# + _categorize_scalar, kept dependency-free so the report stays self-contained).
GEKO_CATEGORIES = (
    "Small GEMMs",
    "Medium GEMMs",
    "Large GEMMs",
    "Large M, smaller N and K",
    "Large M, very small N and K",
    "Large N, smaller M and K",
    "Large N, very small M and K",
    "Large K, smaller M and N",
    "Large K, very small M and N",
    "Large M and N",
    "Large N and K",
    "Large M and K",
    "Very Large GEMMs",
    "M=1",
    "N=1",
    "K=1",
    "Small batch",
    "Medium batch",
    "Large batch",
    "Very large batch",
)


def _categorize_gemm(m, n, k, batch):
    """One of GEKO_CATEGORIES for a shape (port of geko _categorize_scalar)."""
    if batch > 8192:
        return "Very large batch"
    if batch > 1024:
        return "Large batch"
    if batch > 128:
        return "Medium batch"
    if batch >= 2:
        return "Small batch"
    if m == 1:
        return "M=1"
    if n == 1:
        return "N=1"
    if k == 1:
        return "K=1"
    large_m, large_n, large_k = m > 8192, n > 8192, k > 8192
    if large_m and large_n and large_k:
        return "Very Large GEMMs"
    if large_n and large_k:
        return "Large N and K"
    if large_m and large_k:
        return "Large M and K"
    if large_m and large_n:
        return "Large M and N"
    if large_m:
        return (
            "Large M, very small N and K"
            if (n <= 128 and k <= 128)
            else "Large M, smaller N and K"
        )
    if large_n:
        return (
            "Large N, very small M and K"
            if (m <= 128 and k <= 128)
            else "Large N, smaller M and K"
        )
    if large_k:
        return (
            "Large K, very small M and N"
            if (m <= 128 and n <= 128)
            else "Large K, smaller M and N"
        )
    if m <= 1024 and n <= 1024 and k <= 1024:
        return "Small GEMMs"
    if m >= 4096 and n >= 4096 and k >= 4096:
        return "Large GEMMs"
    return "Medium GEMMs"


# ── extractors ──────────────────────────────────────────────────────────────


def _round_num(path):
    """Numeric round index from a round_<n> path (natural sort: 9 < 10 < 11)."""
    m = re.search(r"round_(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def _round_dirs(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "round_*")), key=_round_num)


def _round_trajectory(run_dir):
    """stage04b per round: trained model on that round's NEW (unknown)
    shapes. Includes n_gemms (new shapes benched that round)."""
    rows = []
    for d in _round_dirs(run_dir):
        log = os.path.join(d, "stage04b.log")
        if not os.path.isfile(log):
            continue
        txt = open(log, errors="ignore").read()
        m = re.search(r"global_sel_eff_new \(model\)\s*:\s*([\d.]+)%", txt)
        o = re.search(r"global_origami_sel_eff_new\s*:\s*([\d.]+)%", txt)
        g = re.search(r"\[load\] GEMMs:\s*([\d,]+)", txt)
        if m:
            rows.append(
                {
                    "round": os.path.basename(d),
                    "model": float(m.group(1)),
                    "origami": float(o.group(1)) if o else None,
                    "n_gemms": int(g.group(1).replace(",", "")) if g else None,
                }
            )
    return rows


def _training_trajectory(run_dir):
    """stage05 per round: training-set (in-sample) selection efficiency."""
    rows = []
    for d in _round_dirs(run_dir):
        log = os.path.join(d, "stage05.log")
        if not os.path.isfile(log):
            continue
        txt = open(log, errors="ignore").read()

        def f(pat):
            m = re.search(pat, txt)
            return float(m.group(1)) if m else None

        def i(pat):
            m = re.search(pat, txt)
            return int(m.group(1).replace(",", "")) if m else None

        model = f(r"global_sel_eff \(model\)\s*:\s*([\d.]+)%")
        if model is None:
            continue
        rows.append(
            {
                "round": os.path.basename(d),
                "model": model,
                "deployed": f(r"global_deployed_sel_eff\s*:\s*([\d.]+)%"),
                "origami": f(r"global_origami_sel_eff\s*:\s*([\d.]+)%"),
                "n_cells": i(r"n_cells_trained\s*:\s*([\d,]+)"),
                "n_gemms": i(r"n_gemms_trained\s*:\s*([\d,]+)"),
            }
        )
    return rows


def _final_cells(run_dir):
    """Full DEPLOYED per-cell set -> per-cell rows. The deployed model is the
    cumulative union of every round's trained cells (models.pt keeps the latest
    weights per cell), but each round's cells.json only holds the cells trained
    THAT round. So union ALL rounds' cells.json keeping the LATEST entry per
    cell label -- that set matches the deployed .bin's cell count (a single
    round's cells.json badly under-counts the deployed cells)."""
    cjs = glob.glob(os.path.join(run_dir, "round_*", "stage05", "cells.json"))
    if not cjs:
        return None, []

    def _rnum(p):
        m = re.search(r"round_(\d+)", p)
        return int(m.group(1)) if m else -1

    cjs = sorted(cjs, key=_rnum)
    cj = cjs[-1]
    # Union latest-per-label across all rounds (later round overwrites).
    by_label = {}
    for _cj in cjs:
        try:
            for c in json.load(open(_cj)).get("cells") or []:
                by_label[c.get("label")] = c
        except Exception:
            continue
    # Keep only DEPLOYED LEAF cells: a cell that was later SPLIT has its model
    # replaced by its children in models.pt, so the parent is NOT deployed.
    # A label is a leaf iff no other label extends it at a '#' boundary.
    _labels = [lab for lab in by_label if lab]
    leaves = {
        lab
        for lab in _labels
        if not any(o != lab and o.startswith(lab + "#") for o in _labels)
    }
    d = {"cells": [by_label[lab] for lab in by_label if lab in leaves]}
    # Per-cell HELD-OUT sel_eff (stage4b, unseen shapes) from the cross-round
    # breakdown -- merged onto each cell for the held-out distribution chart.
    held_by_cell = {}
    held_ori_by_cell = {}
    held_w_by_cell = {}  # held-out oracle (winner) geomean us
    held_pick_by_cell = {}  # held-out model pick geomean us
    held_oripick_by_cell = {}  # held-out origami pick geomean us
    bp = os.path.join(run_dir, "held_out_sel_eff_breakdown.json")
    if os.path.isfile(bp):
        try:
            for lab, v in (json.load(open(bp)).get("per_cell") or {}).items():
                se = v.get("sel_eff_new")
                if se is not None:
                    held_by_cell[lab] = se * 100
                ose = v.get("origami_sel_eff_new")
                if ose is not None:
                    held_ori_by_cell[lab] = ose * 100
                for src, dst in (
                    ("winner_us_geomean", held_w_by_cell),
                    ("pick_us_geomean", held_pick_by_cell),
                    ("origami_pick_us_geomean", held_oripick_by_cell),
                ):
                    uv = v.get(src)
                    if isinstance(uv, (int, float)) and uv > 0:
                        dst[lab] = uv
        except Exception:
            pass
    cells = []
    for c in d.get("cells", []):
        model = (c.get("best_sel_eff") or 0) * 100
        dep = c.get("deployed") or {}
        ori = c.get("origami") or {}
        deployed = (dep.get("sel_eff") or 0) * 100
        origami = (
            (ori.get("sel_eff") or 0) * 100 if ori.get("sel_eff") is not None else None
        )
        # ACTUAL per-cell deployed K = number of candidate signatures the
        # cell's model carries (adaptive, knee-truncated to [k_min,k_max]).
        # NOT deployed.smart_k, which is just the configured cap (15 for all).
        smartk = len(c.get("smart_k_signatures") or [])
        if not smartk:
            smartk = (c.get("smart_k_metrics") or {}).get("k_chosen")
        lab = c.get("label")

        # Held-out is measured by stage04b only for cells that got NEW shapes
        # in some round. A leaf CREATED by the final split has no own held-out
        # value -- inherit it from the nearest split-tree ancestor that WAS
        # measured (the parent cell before this split).
        def _inherit(m):
            cur = lab
            while True:
                if cur in m:
                    return round(m[cur], 2), (cur != lab)
                j = cur.rfind("#")
                if j < 0:
                    return None, False
                cur = cur[:j]

        held, held_inherited = _inherit(held_by_cell)
        held_ori, _ = _inherit(held_ori_by_cell)
        held_w_us, _ = _inherit(held_w_by_cell)
        held_pick_us, _ = _inherit(held_pick_by_cell)
        held_ori_pick_us, _ = _inherit(held_oripick_by_cell)
        cells.append(
            {
                "cell": lab,
                "n_gemms": c.get("n_gemms", 0),
                "model": round(model, 2),
                "origami": round(origami, 2) if origami is not None else None,
                "deployed": round(deployed, 2),
                "delta": round(model - origami, 2) if origami is not None else None,
                # smartK effect: deployed (post-K) minus model (pre-K, full ranking)
                "k_delta": (round(deployed - model, 2)),
                "held": held,
                "held_inherited": held_inherited,
                "held_ori": held_ori,
                "held_delta": (
                    round(held - held_ori, 2)
                    if (held is not None and held_ori is not None)
                    else None
                ),
                "dep_delta": (
                    round(deployed - origami, 2) if origami is not None else None
                ),
                "smartK": smartk,
                "cap": f'{c.get("embed_dim")}/{c.get("hidden_dim")}/{c.get("inter_hidden")}',
                "hd": c.get("hidden_dim"),
                # Runtime proxies for runtime/FLOP-weighted aggregation: the cell's
                # oracle (best-benched) geomean us, and the deployed/origami picks'
                # geomean us. Cell runtime weight ~= n_gemms * winner_us_geomean.
                #
                # Use the winner geomean from the SAME eval that produced the pick
                # geomeans (the deployed/origami full-cell eval, n_eval = all the
                # cell's GEMMs), NOT the top-level winner_us_geomean -- the latter is
                # taken over the model's *training eval split* (a different, usually
                # smaller subset), so mixing them made "oracle us" come out LARGER
                # than "model pick us". The oracle is the floor: within one eval set
                # winner <= pick always, so this keeps oracle <= both pick columns.
                "w_us": _first_pos(
                    dep.get("winner_us_geomean"),
                    ori.get("winner_us_geomean"),
                    c.get("winner_us_geomean"),
                ),
                "dep_pick_us": dep.get("pick_us_geomean"),
                "ori_pick_us": ori.get("pick_us_geomean"),
                # HELD-OUT latency geomeans (stage04b, unseen shapes) -- the
                # worst-cells table uses these so its us columns are consistent with
                # its held-out sel_eff columns (held oracle <= held model/origami).
                "held_w_us": held_w_us,
                "held_pick_us": held_pick_us,
                "held_ori_pick_us": held_ori_pick_us,
            }
        )
    return os.path.basename(os.path.dirname(os.path.dirname(cj))), cells


def _ensure_breakdown(run_dir):
    """Make the report self-sufficient for the held-out aggregate, 3D cell map
    and category breakdown -- all of which read held_out_sel_eff_breakdown.json.

    That file is normally written by the orchestrator at END-OF-PIPELINE, AFTER
    stage 7/8. If the run was interrupted (e.g. stage 8 cancelled) the write
    never happened -- but the breakdown is derivable PURELY from the per-round
    TRAINING data (round_*/stage04b/decisions.json), which is independent of
    stage 7/8. So when the file is absent we rebuild it here from that data, so
    the training-derived report sections never silently go blank just because a
    downstream validator was skipped. No-op if the file already exists (we never
    overwrite the orchestrator's authoritative copy)."""
    bp = os.path.join(run_dir, "held_out_sel_eff_breakdown.json")
    if os.path.isfile(bp):
        # Trust an existing breakdown ONLY if it already carries the held-out
        # us geomeans the worst-cells table needs. Older breakdowns (written
        # before that field existed) are refreshed from decisions.json below.
        try:
            _pc = json.load(open(bp)).get("per_cell") or {}
            # Trust an existing breakdown ONLY if it is non-empty AND already
            # carries the held-out us geomeans. `not _pc or ...` trusted an
            # EMPTY per_cell, skipping the rebuild-from-decisions.json salvage
            # path in exactly the case it exists for.
            if _pc and any("pick_us_geomean" in v for v in _pc.values()):
                return
        except Exception:
            return
    dec = glob.glob(os.path.join(run_dir, "round_*", "stage04b", "decisions.json"))

    def _rnum(p):
        m = re.search(r"round_(\d+)", p)
        return int(m.group(1)) if m else -1

    # Integer round order so a later round's reading overwrites an earlier one
    # ("most recent per cell"); lex order would put round_14 before round_2.
    per_cell = {}
    rounds_seen = []
    for p in sorted(dec, key=_rnum):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rounds_seen.append("round_%d" % _rnum(p))
        for c in d.get("per_cell", []):
            se = c.get("sel_eff_new")
            n = c.get("n_eval") or 0
            if se is None or n <= 0 or float(se) <= 0:
                continue
            ose = c.get("origami_sel_eff_new")
            on = c.get("origami_n_eval") or 0

            def _f(x):
                return float(x) if isinstance(x, (int, float)) else None

            per_cell[c["cell"]] = {
                "round": "round_%d" % _rnum(p),
                "sel_eff_new": float(se),
                "n_eval": int(n),
                "origami_sel_eff_new": (float(ose) if ose is not None else None),
                "origami_n_eval": int(on),
                # Held-out latency geomeans (us) over the same unseen shapes, so
                # the worst-cells table can be held-out-only AND consistent.
                "winner_us_geomean": _f(c.get("winner_us_geomean")),
                "pick_us_geomean": _f(c.get("pick_us_geomean")),
                "origami_pick_us_geomean": _f(c.get("origami_pick_us_geomean")),
            }
    if not per_cell:
        return  # no stage04b data yet -- leave absent (sections show n/a)

    def _gm(pairs):
        ls = ws = 0.0
        for se, n in pairs:
            if se is None or n <= 0 or float(se) <= 0:
                continue
            ls += math.log(float(se)) * float(n)
            ws += float(n)
        return math.exp(ls / ws) if ws > 0 else float("nan")

    agg = {
        "held_out_sel_eff": _gm(
            [(e["sel_eff_new"], e["n_eval"]) for e in per_cell.values()]
        ),
        "origami_held_out_sel_eff": _gm(
            [
                (e.get("origami_sel_eff_new"), e.get("origami_n_eval", 0))
                for e in per_cell.values()
            ]
        ),
        "n_cells": len(per_cell),
        "n_cells_with_origami": sum(
            1
            for e in per_cell.values()
            if e.get("origami_sel_eff_new") is not None
            and e.get("origami_n_eval", 0) > 0
        ),
        "n_rounds_aggregated": len(rounds_seen),
        "rounds": rounds_seen,
        "per_cell": per_cell,
        "_rebuilt_by_make_report": True,
    }
    try:
        with open(bp, "w") as f:
            json.dump(agg, f, indent=2)
    except Exception:
        pass


def _held_out_aggregate(run_dir):
    """Cross-round per-cell held-out aggregate over the DEPLOYED LEAF cells only
    (cells with no children) -- so the count matches the 3D map and the shipped
    model, and the % isn't dragged down by superseded PARENT cells (which were
    split into children and are never deployed). Recomputed GEMM-weighted over
    leaves; falls back to the breakdown's top-level number if per_cell is
    absent. (For reference, the orchestrator's all-cell aggregate -- incl.
    parents -- is in held_out_sel_eff_breakdown.json's top-level field.)"""
    import math

    f = os.path.join(run_dir, "held_out_sel_eff_breakdown.json")
    if not os.path.isfile(f):
        return None
    b = json.load(open(f))
    pc = b.get("per_cell", {})
    if pc:
        labset = set(pc)
        leaves = [
            l for l in pc if not any(o != l and o.startswith(l + "#") for o in labset)
        ]

        def _gw(mkey, nkey):
            num = den = 0.0
            for c in leaves:
                se, n = pc[c].get(mkey), pc[c].get(nkey)
                if se and se > 0 and n:
                    num += n * math.log(se)
                    den += n
            return math.exp(num / den) if den else None

        mo = _gw("sel_eff_new", "n_eval")
        oo = _gw("origami_sel_eff_new", "origami_n_eval")
        n_cells = len(leaves)
        # HYBRID (oracle per-cell router): for each leaf, take the BETTER of the
        # model vs origami held-out sel_eff -- simulating a deploy that routes
        # each cell to whichever selector wins on that cell's held-out shapes.
        # This is the ceiling a per-cell model/origami fallback could reach. We
        # use the model's n_eval as the GEMM weight (same shapes either way) and
        # only count cells that have BOTH numbers.
        hnum = hden = 0.0
        hwins = 0
        for c in leaves:
            mse, ose = pc[c].get("sel_eff_new"), pc[c].get("origami_sel_eff_new")
            n = pc[c].get("n_eval")
            if mse and mse > 0 and ose and ose > 0 and n:
                best = max(mse, ose)
                if mse >= ose:
                    hwins += 1
                hnum += n * math.log(best)
                hden += n
        hy = math.exp(hnum / hden) if hden else None
        n_both = sum(
            1
            for c in leaves
            if pc[c].get("sel_eff_new") and pc[c].get("origami_sel_eff_new")
        )
        model_win_frac = round(hwins / n_both * 100, 1) if n_both else None
    else:
        mo = b.get("held_out_sel_eff")
        oo = b.get("origami_held_out_sel_eff")
        n_cells = b.get("n_cells")
        hy = None
        model_win_frac = None
    return {
        "model": round(mo * 100, 2) if mo is not None else None,
        "origami": round(oo * 100, 2) if oo is not None else None,
        "delta": (
            round((mo - oo) * 100, 2) if (mo is not None and oo is not None) else None
        ),
        "hybrid": round(hy * 100, 2) if hy is not None else None,
        "hybrid_gain": (
            round((hy - mo) * 100, 2) if (hy is not None and mo is not None) else None
        ),
        "model_win_frac": model_win_frac,
        "n_cells": n_cells,
    }


def _stage7(run_dir):
    """Selection-TIME (rank_configs wall-clock) + C++/Python pick parity."""
    out = {}
    logs = glob.glob(os.path.join(run_dir, "validate*", "stage07.log"))
    if logs:
        txt = open(logs[0], errors="ignore").read()
        # columns: path  n  min  median  mean  max  (capture all four stats)
        mo = re.search(
            r"origami\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt
        )
        mr = re.search(r"ml_rec\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
        sp = re.search(r"speedup ml_rec vs origami:\s*([\d.]+)x", txt)
        if mo:
            out["origami_min"] = float(mo.group(1))
            out["origami_us"] = float(mo.group(2))
            out["origami_mean_us"] = float(mo.group(3))
            out["origami_max"] = float(mo.group(4))
        if mr:
            out["ml_min"] = float(mr.group(1))
            out["ml_us"] = float(mr.group(2))
            out["ml_mean_us"] = float(mr.group(3))
            out["ml_max"] = float(mr.group(4))
        if sp:
            out["speedup"] = float(sp.group(1))
    for pj in glob.glob(
        os.path.join(run_dir, "validate*", "stage07", "*", "parity.json")
    ):
        p = json.load(open(pj))
        out["parity_compared"] = p.get("n_compared")
        out["parity_match"] = p.get("n_match")
        out["match_rate"] = round((p.get("match_rate") or 0) * 100, 2)
        out["smart_k"] = p.get("smart_k")
        break
    # E2E selection time -- the C++ "Solution selection time" (full
    # findTopSolutions wall-clock, > rank_configs alone). From
    # timings_summary.json's compare block: ml_on = the ml_rec path,
    # ml_off = the plain-origami path. Reported median + mean.
    for tj in glob.glob(
        os.path.join(run_dir, "validate*", "stage07", "timings_summary.json")
    ):
        try:
            d = json.load(open(tj))
        except Exception:
            break
        # Structured rank_configs microbench (preferred over the log regex
        # above): timings_summary.json carries the full min/median/mean/max +
        # speedup, and is what any re-run of the microbench writes back.
        rc = d.get("rank_configs_call_time") or {}
        _mr, _or = (rc.get("ml_rec") or {}), (rc.get("origami") or {})
        if _mr.get("median_us") is not None:
            out["ml_min"] = round(_mr.get("min_us", 0), 2)
            out["ml_us"] = round(_mr["median_us"], 2)
            out["ml_mean_us"] = round(_mr.get("mean_us", _mr["median_us"]), 2)
            out["ml_max"] = round(_mr.get("max_us", 0), 2)
        if _or.get("median_us") is not None:
            out["origami_min"] = round(_or.get("min_us", 0), 2)
            out["origami_us"] = round(_or["median_us"], 2)
            out["origami_mean_us"] = round(_or.get("mean_us", _or["median_us"]), 2)
            out["origami_max"] = round(_or.get("max_us", 0), 2)
        if rc.get("speedup_mean_ml_vs_origami") is not None:
            out["speedup"] = round(rc["speedup_mean_ml_vs_origami"], 2)
        for y in (d.get("per_yaml") or {}).values():
            c = (y or {}).get("compare") or {}
            on, off = (c.get("ml_on") or {}), (c.get("ml_off") or {})
            if on.get("sel_us_p50") is not None:
                out["ml_e2e_us"] = round(on["sel_us_p50"], 1)
                out["ml_e2e_mean"] = round(on.get("sel_us_mean", on["sel_us_p50"]), 1)
            if off.get("sel_us_p50") is not None:
                out["origami_e2e_us"] = round(off["sel_us_p50"], 1)
                out["origami_e2e_mean"] = round(
                    off.get("sel_us_mean", off["sel_us_p50"]), 1
                )
            break
        break
    return out


def _stage8(run_dir):
    """Selection efficiency on external held-out traces, reported BOTH ways:
      - equal-weight (per cell): partition-DEPENDENT -- each run carves the
        same GEMMs into a different cell set, so Origami's number varies across
        runs;
      - GEMM-weighted (per GEMM): partition-INDEPENDENT -- Origami is
        consistent across runs (same GEMMs, same analytical picks).
    Merges every validate*/stage08*/summary.json (so a supplementary
    stage08_<name>/ eval is included) and falls back to the GEOMEAN lines in
    stage08.log for datasets with no summary (e.g. one whose eval errored)."""

    def _pc(v):
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    by_name = {}
    for f in sorted(
        glob.glob(os.path.join(run_dir, "validate*", "stage08*", "summary.json"))
    ):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for x in d.get("datasets", []):
            nm = x.get("name")
            if nm in by_name or x.get("model_geomean_sel_eff") is None:
                continue
            s = x.get("summary", {})
            by_name[nm] = {
                "name": nm,
                "model": _pc(x.get("model_geomean_sel_eff")),
                "origami": _pc(x.get("origami_geomean_sel_eff")),
                "model_gw": _pc(s.get("model_geomean_gemm_weighted")),
                "origami_gw": _pc(s.get("origami_geomean_gemm_weighted")),
            }
    from_summary = set(by_name)
    logs = glob.glob(os.path.join(run_dir, "validate*", "stage08.log")) or glob.glob(
        os.path.join(run_dir, "validate*", "stage08", "*.log")
    )
    if logs:
        txt = open(logs[0], errors="replace").read()
        cur = None
        for ln in txt.splitlines():
            ds = re.search(r"dataset ::\s+(\S+)", ln)
            if ds:
                cur = ds.group(1)
                continue
            if not cur or cur in from_summary:
                continue
            eq = re.search(
                r"GEOMEAN \(all cells, equal-weight\)\s+model=([\d.]+)"
                r"\s+origami=([\d.]+)",
                ln,
            )
            gw = re.search(
                r"GEOMEAN \(GEMM-weighted[^)]*\)\s+model=([\d.]+)"
                r"\s+origami=([\d.]+)",
                ln,
            )
            if eq:
                e = by_name.setdefault(
                    cur, {"name": cur, "model_gw": None, "origami_gw": None}
                )
                e["model"] = round(float(eq.group(1)) * 100, 2)
                e["origami"] = round(float(eq.group(2)) * 100, 2)
            elif gw:
                e = by_name.setdefault(cur, {"name": cur})
                e["model_gw"] = round(float(gw.group(1)) * 100, 2)
                e["origami_gw"] = round(float(gw.group(2)) * 100, 2)
    return sorted(
        (v for v in by_name.values() if v.get("model") is not None),
        key=lambda v: v["name"],
    )


def _bin_size_mb(run_dir):
    """Deployed model .bin size in MB: prefer the latest round's manifest
    weights_bin_size, else parse the stage06 deploy log."""
    mans = sorted(
        glob.glob(
            os.path.join(
                run_dir, "round_*", "stage06", "**", "ml_recommender.manifest.json"
            ),
            recursive=True,
        ),
        key=_round_num,
    )
    for m in reversed(mans):
        try:
            wb = json.load(open(m)).get("weights_bin_size")
            if wb:
                return round(wb / 1e6, 2)
        except Exception:
            pass
    for lg in sorted(
        glob.glob(os.path.join(run_dir, "round_*", "stage06.log")),
        key=_round_num,
        reverse=True,
    ):
        try:
            mm = re.search(
                r"\.bin[^\(]*\(([\d,]+)\s*bytes\)", open(lg, errors="replace").read()
            )
            if mm:
                return round(int(mm.group(1).replace(",", "")) / 1e6, 2)
        except Exception:
            pass
    return None


def _cell3d_b64(run_dir):
    """3D (M/N/K) cell-map PNGs (base64 data-URIs) per batch tier (Bany,
    Bnone): one cuboid per deployed leaf in log10(M/N/K) space, colored by
    split depth (unsplit faint, deeper = darker). Adapted from cell3d.py.
    Returns {bt: data_uri}; {} if matplotlib is unavailable or no cell data."""
    try:
        import io, base64, math
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return {}
    f = os.path.join(run_dir, "held_out_sel_eff_breakdown.json")
    labels = []
    if os.path.isfile(f):
        try:
            labels = list(json.load(open(f)).get("per_cell", {}).keys())
        except Exception:
            labels = []
    if not labels:
        return {}
    labset = set(labels)
    leaves = [
        l for l in labels if not any(o != l and o.startswith(l + "#") for o in labset)
    ]
    MN = {
        "Tiny": (1, 32),
        "Small": (33, 128),
        "Mid": (129, 512),
        "Large": (513, 524288),
    }
    KK = {"TinyK": (1, 32), "MidK": (33, 512), "LargeK": (513, 131072)}

    def tiers(l):
        a = l.split("|")
        return a[0], a[1], a[2], a[3].split("#")[0]

    def ranges(l):
        mt, nt, kt, _ = tiers(l)
        mlo, mhi = MN[mt]
        nlo, nhi = MN[nt]
        klo, khi = KK[kt]
        for seg in l.split("#")[1:]:
            g = re.match(r"([MNK])(<=|>)(\d+)", seg)
            if not g:
                continue
            ax, op, v = g.group(1), g.group(2), int(g.group(3))
            if ax == "M":
                mhi = min(mhi, v) if op == "<=" else mhi
                mlo = max(mlo, v + 1) if op == ">" else mlo
            elif ax == "N":
                nhi = min(nhi, v) if op == "<=" else nhi
                nlo = max(nlo, v + 1) if op == ">" else nlo
            elif ax == "K":
                khi = min(khi, v) if op == "<=" else khi
                klo = max(klo, v + 1) if op == ">" else klo
        return mlo, mhi, nlo, nhi, klo, khi

    def depth(l):
        return len(l.split("#")) - 1

    def lg(v):
        return math.log10(max(v, 1))

    def cuboid(x0, x1, y0, y1, z0, z1):
        p = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        fc = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
        return [[p[i] for i in q] for q in fc]

    # Sequential blue ramp (shallow -> deeply refined = light -> dark).
    DEPTHCOL = {
        1: "#9ecae1",
        2: "#4292c6",
        3: "#08519c",
        4: "#08306b",
        5: "#041f4a",
        6: "#02132e",
    }
    out = {}
    for BT in ("Bany", "Bnone"):
        sl = sorted([l for l in leaves if tiers(l)[3] == BT], key=depth)
        if not sl:
            continue
        fig = plt.figure(figsize=(7.6, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        for l in sl:
            mlo, mhi, nlo, nhi, klo, khi = ranges(l)
            d = depth(l)
            faces = cuboid(lg(mlo), lg(mhi), lg(nlo), lg(nhi), lg(klo), lg(khi))
            if d == 0:
                pc = Poly3DCollection(
                    faces,
                    facecolors=(0, 0, 0, 0),
                    edgecolor=(0.5, 0.5, 0.5, 0.45),
                    linewidths=0.2,
                )
            else:
                pc = Poly3DCollection(
                    faces,
                    facecolor=DEPTHCOL.get(d, "#02132e"),
                    edgecolor="white",
                    linewidths=0.4,
                    alpha=0.45 + 0.08 * min(d, 4),
                )
            ax.add_collection3d(pc)
        # Ticks carry RAW M/N/K sizes (1, 32, 512, ...) but are placed on a
        # log layout so every cell stays visible -- a pure-linear axis crushes
        # all but the Large tier (513..524288) into an invisible sliver.
        ax.set_xlim(0, lg(524288))
        ax.set_ylim(0, lg(524288))
        ax.set_zlim(0, lg(131072))
        ax.set_xlabel("M", labelpad=12)
        ax.set_ylabel("N", labelpad=12)
        ax.set_zlabel("K", labelpad=14)
        ax.set_xticks([lg(1), lg(32), lg(512), lg(524288)])
        ax.set_yticks([lg(1), lg(32), lg(512), lg(524288)])
        ax.set_zticks([lg(1), lg(32), lg(512), lg(131072)])
        ax.set_xticklabels(["1", "32", "512", "512k"])
        ax.set_yticklabels(["1", "32", "512", "512k"])
        ax.set_zticklabels(["1", "32", "512", "128k"])
        ax.tick_params(axis="z", pad=5)
        ax.view_init(elev=18, azim=-52)
        maxd = max((depth(l) for l in sl), default=0)
        if maxd >= 1:
            ax.legend(
                handles=[
                    Patch(
                        facecolor=DEPTHCOL.get(d, "#02132e"),
                        edgecolor="gray",
                        label=f"depth {d}",
                    )
                    for d in range(1, maxd + 1)
                ],
                frameon=False,
                fontsize=8,
                loc="upper left",
                ncol=2,
            )
        ax.set_title(
            f"3D cell map (M/N/K) - B = {BT}  ({len(sl)} leaf cells)\n"
            "unsplit faint; colored = refined (deeper = darker)",
            fontsize=10,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        out[BT] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


def _timing(run_dir):
    """Parse the orchestrator's [timing] block -> per-round stage rows +
    round totals + wall-clock total."""
    log = os.path.join(run_dir, "orchestrator.log")
    if not os.path.isfile(log):
        return [], None

    def _dur(s):
        h = re.search(r"(\d+)h", s or "")
        m = re.search(r"(\d+)m", s or "")
        sc = re.search(r"(\d+)s", s or "")
        return (
            (int(h.group(1)) * 3600 if h else 0)
            + (int(m.group(1)) * 60 if m else 0)
            + (int(sc.group(1)) if sc else 0)
        )

    # The log may hold several [timing] blocks (resumes / validate reruns).
    # Dedupe rounds by label (last occurrence wins, original order) and take
    # the longest WALL-CLOCK TOTAL (the full run, not a short rerun).
    by_round, order, totals, cur = {}, [], [], None
    for ln in open(log, errors="ignore"):
        m = re.match(r"\[timing\]\s+(\S.*)", ln)
        if m:
            t = m.group(1).strip()
            wm = re.match(r"WALL-CLOCK TOTAL:\s*(\S+)", t)
            if wm:
                totals.append(wm.group(1))
                cur = None
            else:
                cur = {"round": t, "stages": [], "total": None}
                if t not in by_round:
                    order.append(t)
                by_round[t] = cur
            continue
        if cur is not None:
            # A stage that FAILED still spent wall-clock; the orchestrator
            # prints it as `[rc= -9]` instead of `[ok ]`. Matching only `[ok`
            # would silently drop those rows and leave a failed stage's
            # (possibly long) wall-clock as a blank cell.
            sm = re.match(r"\s+(Stage [\d.]+ -- .+?)\s{2,}(\S+)\s+\[(?:ok|rc)", ln)
            if sm:
                cur["stages"].append({"name": sm.group(1).strip(), "time": sm.group(2)})
                continue
            tm = re.match(r"\s+\(round total\)\s+(\S+)\s+\[(?:ok|rc)", ln)
            if tm:
                cur["total"] = tm.group(1)
    rounds = [by_round[r] for r in order]
    total = max(totals, key=_dur) if totals else None
    return rounds, total


# ── canonical cell ordering ───────────────────────────────────────────────────
# Cells must render in a STABLE order identical across runs (so two runs
# list the same cells in the same place). Order by the base 96-grid
# coordinate -- the (M, N, K) size-tier tuple then Batch -- and order split
# children by (axis, numeric threshold, op). The key depends ONLY on a cell's
# own label, never on sel_eff / n_gemms / training or insertion order, so it is
# fully reproducible across runs.
_MN_TIER_RANK = {"Tiny": 0, "Small": 1, "Mid": 2, "Large": 3}
_K_TIER_RANK = {"TinyK": 0, "MidK": 1, "LargeK": 2}
_BATCH_RANK = {"Bany": 0, "Bnone": 1}
_AXIS_RANK = {"M": 0, "N": 1, "K": 2, "B": 3}


def _base_cell_key(base):
    """Canonical sort key for a base label 'M|N|K|B': the (M, N, K) size-tier
    tuple then Batch. Unknown tiers sort last but stay deterministic (fall
    back to the raw token) so a malformed label never breaks the ordering."""
    parts = (base or "").split("|")
    if len(parts) < 4:
        return (1, (base or "",))
    m, n, k, b = parts[0], parts[1], parts[2], parts[3]
    return (
        0,
        (
            _MN_TIER_RANK.get(m, 99),
            m,
            _MN_TIER_RANK.get(n, 99),
            n,
            _K_TIER_RANK.get(k, 99),
            k,
            _BATCH_RANK.get(b, 99),
            b,
        ),
    )


def _seg_key(seg):
    """Canonical sort key for one split segment like 'M<=2172' / 'N>4608':
    order by axis (M, N, K), then NUMERIC threshold, then op ('<=' before
    '>'). Unparseable segments fall back to the raw string (sorted last)."""
    m = re.match(r"([A-Za-z]+)(<=|>)(\d+)$", seg or "")
    if not m:
        return (1, 99, 0, 0, seg or "")
    axis, op, thr = m.group(1), m.group(2), int(m.group(3))
    return (0, _AXIS_RANK.get(axis, 99), thr, 0 if op == "<=" else 1, "")


def _cell_tree_ascii(cells):
    """Build the split tree from leaf labels (base|M|N|K|B + #split
    segments) and render it as a `tree`-command-style ASCII diagram."""
    root = {}
    for c in cells:
        base, *segs = (c["cell"] or "").split("#")
        node = root.setdefault(base, {})
        for s in segs:
            node = node.setdefault(s, {})
        node["__leaf__"] = c

    def stat(leaf):
        if not leaf:
            return ""
        d = leaf.get("delta")
        if d is None:
            return f"   [n={leaf['n_gemms']:,}]"
        return (
            f"   [n={leaf['n_gemms']:,}  model={leaf['model']:.1f}%  "
            f"\u0394={'+' if d >= 0 else ''}{d:.1f}  K={leaf.get('smartK')}]"
        )

    lines = []

    def render(node, prefix):
        kids = sorted(
            ((k, v) for k, v in node.items() if k != "__leaf__"),
            key=lambda kv: _seg_key(kv[0]),
        )
        for i, (k, v) in enumerate(kids):
            last = i == len(kids) - 1
            lines.append(
                prefix
                + ("\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 ")
                + k
                + stat(v.get("__leaf__"))
            )
            render(v, prefix + ("    " if last else "\u2502   "))

    for base in sorted(root, key=_base_cell_key):
        node = root[base]
        is_leaf = not [k for k in node if k != "__leaf__"]
        lines.append(base + (stat(node.get("__leaf__")) if is_leaf else ""))
        render(node, "")
        lines.append("")
    return "\n".join(lines).rstrip()


# ── inline SVG charts ─────────────────────────────────────────────────────────
# Pure server-rendered SVG: NO <script>, NO external/CDN refs, so the report
# .html stays a single self-contained file. Every chart builder returns "" when
# its input is missing, so older runs that lack a datum still render cleanly.
# Two reusable primitives (_svg_bars / _svg_lines) back all three charts rather
# than hand-coding each.

_CHART_TEXT = "#1f2328"
_CHART_MUT = "#57606a"
_CHART_GRID = "#eaeef2"
# Series palette, kept consistent across charts: model / deployed / origami.
_C_MODEL = "#0969da"
_C_DEPLOYED = "#1a7f37"
_C_ORIGAMI = "#9a6700"


def _svg_t(s):
    return html.escape(str(s), quote=True)


def _short(s, n=14):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "."


def _nice_top(v):
    """Smallest multiple of 5 >= v (for a count axis top)."""
    iv = int(v)
    if v > iv:
        iv += 1
    iv = max(1, iv)
    return ((iv + 4) // 5) * 5


def _figure(caption, svg):
    """Wrap an <svg> in a captioned <figure>; "" when the chart is empty."""
    if not svg:
        return ""
    return (
        f'<figure class="chart"><figcaption>{_svg_t(caption)}'
        f"</figcaption>{svg}</figure>"
    )


def _svg_bars(labels, series, *, width=760, height=240, y_max=None):
    """Vertical bar chart from one or more grouped series. `series` is a list
    of (name, values, color); `values` is parallel to `labels`, `color` is a
    single color OR a per-bar list. One series => plain bars (e.g. a
    histogram with per-bucket colors); many => grouped bars. Bars use a 0
    baseline so heights are proportional. Returns an <svg> string (or "")."""
    series = [(n, vs, c) for (n, vs, c) in series if any(v is not None for v in vs)]
    flat = [v for _, vs, _ in series for v in vs if v is not None]
    if not labels or not series or not flat:
        return ""
    top = float(y_max) if y_max is not None else float(_nice_top(max(flat)))
    if top <= 0:
        top = 1.0
    ml, mr, mt, mb = 46, 12, 26, 38
    x0, y1 = ml, height - mb
    pw, ph = (width - mr) - ml, y1 - mt
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for frac in (0.0, 0.5, 1.0):
        gy = y1 - ph * frac
        out.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{width-mr}" '
            f'y2="{gy:.1f}" stroke="{_CHART_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0-6}" y="{gy+3:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_CHART_MUT}">{top*frac:g}</text>'
        )
    n_groups = len(labels)
    gw = pw / n_groups
    nb = len(series)
    bw = (gw * 0.8) / nb
    for gi, lab in enumerate(labels):
        gx = x0 + gw * gi
        for si, (name, vs, color) in enumerate(series):
            v = vs[gi] if gi < len(vs) else None
            if v is None:
                continue
            bh = ph * (max(0.0, min(float(v), top)) / top)
            bx = gx + gw * 0.1 + bw * si
            col = color[gi] if isinstance(color, (list, tuple)) else color
            out.append(
                f'<rect x="{bx:.1f}" y="{y1-bh:.1f}" '
                f'width="{bw*0.92:.1f}" height="{bh:.1f}" '
                f'fill="{col}"><title>{_svg_t(lab)} '
                f"{_svg_t(name)}: {v:g}</title></rect>"
            )
        out.append(
            f'<text x="{gx+gw/2:.1f}" y="{y1+13:.1f}" '
            f'text-anchor="middle" font-size="10" '
            f'fill="{_CHART_MUT}">{_svg_t(_short(lab))}</text>'
        )
    if nb > 1:
        lx = x0
        for name, _vs, color in series:
            col = color if isinstance(color, str) else _CHART_MUT
            out.append(
                f'<rect x="{lx}" y="{mt-16}" width="9" height="9" ' f'fill="{col}"/>'
            )
            out.append(
                f'<text x="{lx+13}" y="{mt-8}" font-size="10" '
                f'fill="{_CHART_TEXT}">{_svg_t(name)}</text>'
            )
            lx += 24 + len(name) * 6
    out.append("</svg>")
    return "".join(out)


def _svg_lines(labels, series, *, width=760, height=240, y_suffix=""):
    """Multi-series line chart. `series` is a list of (name, values, color);
    `values` parallel to `labels`, may contain None (those points are
    skipped, the line connects the present ones). Auto-ranges the y-axis with
    a labeled scale (non-zero baseline is fine for a trend line)."""
    series = [(n, vs, c) for (n, vs, c) in series if any(v is not None for v in vs)]
    flat = [v for _, vs, _ in series for v in vs if v is not None]
    if not labels or not series or not flat:
        return ""
    lo, hi = min(flat), max(flat)
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.15
    y_lo, y_hi = lo - pad, hi + pad
    ml, mr, mt, mb = 46, 12, 26, 38
    x0, y1 = ml, height - mb
    pw, ph = (width - mr) - ml, y1 - mt
    n = len(labels)

    def xp(i):
        return x0 + (pw * i / (n - 1) if n > 1 else pw / 2)

    def yp(v):
        return y1 - ph * (v - y_lo) / (y_hi - y_lo)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for frac in (0.0, 0.5, 1.0):
        gy = y1 - ph * frac
        val = y_lo + (y_hi - y_lo) * frac
        out.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{width-mr}" '
            f'y2="{gy:.1f}" stroke="{_CHART_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0-6}" y="{gy+3:.1f}" text-anchor="end" '
            f'font-size="10" fill="{_CHART_MUT}">'
            f"{val:.0f}{y_suffix}</text>"
        )
    for i, lab in enumerate(labels):
        out.append(
            f'<text x="{xp(i):.1f}" y="{y1+13:.1f}" '
            f'text-anchor="middle" font-size="10" '
            f'fill="{_CHART_MUT}">{_svg_t(_short(lab))}</text>'
        )
    for name, vs, color in series:
        pres = [(xp(i), yp(v), v) for i, v in enumerate(vs) if v is not None]
        if len(pres) >= 2:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pres)
            out.append(
                f'<polyline fill="none" stroke="{color}" '
                f'stroke-width="2" points="{pts}"/>'
            )
        for x, y, v in pres:
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" '
                f'fill="{color}"><title>{_svg_t(name)}: {v:g}'
                f"{y_suffix}</title></circle>"
            )
    lx = x0
    for name, _vs, color in series:
        out.append(
            f'<rect x="{lx}" y="{mt-16}" width="9" height="9" ' f'fill="{color}"/>'
        )
        out.append(
            f'<text x="{lx+13}" y="{mt-8}" font-size="10" '
            f'fill="{_CHART_TEXT}">{_svg_t(name)}</text>'
        )
        lx += 24 + len(name) * 6
    out.append("</svg>")
    return "".join(out)


def _chart_trend(train_traj):
    """(a) Per-round model/deployed/origami sel_eff (stage05 training-set)."""
    if not train_traj:
        return ""
    labels = [r["round"].replace("round_", "r") for r in train_traj]
    svg = _svg_lines(
        labels,
        [
            ("model", [r.get("model") for r in train_traj], _C_MODEL),
            ("deployed", [r.get("deployed") for r in train_traj], _C_DEPLOYED),
            ("origami", [r.get("origami") for r in train_traj], _C_ORIGAMI),
        ],
        y_suffix="%",
    )
    return _figure(
        "Per-round selection efficiency (stage05 training-set): "
        "model vs deployed vs origami.",
        svg,
    )


def _chart_cell_dist(cells):
    """(b) Final-round per-cell DEPLOYED sel_eff distribution (histogram)."""
    vals = [c["deployed"] for c in (cells or []) if c.get("deployed") is not None]
    if not vals:
        return ""
    edges = [0, 70, 80, 85, 88, 90, 95, 100.0001]
    labels = ["<70", "70-80", "80-85", "85-88", "88-90", "90-95", ">=95"]
    colors = [
        "#cf222e",
        "#cf222e",
        "#bf8700",
        "#bf8700",
        "#bf8700",
        "#2da44e",
        "#1a7f37",
    ]
    counts = [0] * len(labels)
    for v in vals:
        for bi in range(len(labels)):
            if edges[bi] <= v < edges[bi + 1]:
                counts[bi] += 1
                break
    svg = _svg_bars(labels, [("cells", counts, colors)])
    return _figure(
        f"IN-SAMPLE (deployed) \u2014 sel_eff of each cell's deployed "
        f"model scored on its OWN benched GEMMs (the training-set / "
        f"production smart-K result, on KNOWN shapes), {len(vals)} "
        f"cells. Bar height = # cells; red < 88%, green >= 95%. "
        f"Companion to the HELD-OUT chart, which uses the same bins "
        f"but scores NEW / unseen shapes.",
        svg,
    )


def _chart_held_dist(cells):
    """Final held-out (stage4b, unknown-shape) sel_eff distribution histogram
    -- the generalization analogue of the deployed-sel_eff chart."""
    vals = [c["held"] for c in (cells or []) if c.get("held") is not None]
    if not vals:
        return ""
    edges = [0, 70, 80, 85, 88, 90, 95, 100.0001]
    labels = ["<70", "70-80", "80-85", "85-88", "88-90", "90-95", ">=95"]
    colors = [
        "#cf222e",
        "#cf222e",
        "#bf8700",
        "#bf8700",
        "#bf8700",
        "#2da44e",
        "#1a7f37",
    ]
    counts = [0] * len(labels)
    for v in vals:
        for bi in range(len(labels)):
            if edges[bi] <= v < edges[bi + 1]:
                counts[bi] += 1
                break
    svg = _svg_bars(labels, [("cells", counts, colors)])
    return _figure(
        f"HELD-OUT (out-of-sample) \u2014 sel_eff of each cell's "
        f"model scored ONLY on its NEW / unseen shapes (stage4b "
        f"generalization test; shapes never trained on), {len(vals)} "
        f"cells. Bar height = # cells; red < 88%, green >= 95%. "
        f"Companion to the IN-SAMPLE (deployed) chart, which uses the "
        f"same bins but scores KNOWN shapes.",
        svg,
    )


def _chart_s8(s8):
    """(c) Held-out model vs origami per stage08 dataset (grouped bars)."""
    if not s8:
        return ""
    svg = _svg_bars(
        [r["name"] for r in s8],
        [
            ("model", [r.get("model") for r in s8], _C_MODEL),
            ("origami", [r.get("origami") for r in s8], _C_ORIGAMI),
        ],
        y_max=100,
    )
    return _figure(
        "Held-out selection efficiency by stage08 dataset: "
        "model vs origami (equal-weight %).",
        svg,
    )


def _stage8_percell(run_dir):
    """Per-cell model-vs-origami sel_eff per stage08 dataset, parsed from the
    `[eval] <cell> <n> <model> <origami>` lines in every validate*/stage08*.log
    -- a universal source (the per_cell_sel_eff.json paths in summary.json go
    stale after a run rename, and this also covers log-fallback runs + any
    merged stage08_<name> dir). Returns {dataset: [(cell, model%, origami%), ...]}
    sorted by (model-origami) ascending (worst first)."""
    acc = {}
    for lg in sorted(glob.glob(os.path.join(run_dir, "validate*", "stage08*.log"))):
        try:
            txt = open(lg, errors="replace").read()
        except Exception:
            continue
        cur = None
        for ln in txt.splitlines():
            ds = re.search(r"dataset ::\s+(\S+)", ln)
            if ds:
                cur = ds.group(1)
                acc.setdefault(cur, {})
                continue
            m = re.match(r"\[eval\]\s+(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", ln)
            if m and cur is not None and m.group(1) not in acc[cur]:
                acc[cur][m.group(1)] = (
                    round(float(m.group(3)) * 100, 2),
                    round(float(m.group(4)) * 100, 2),
                )
    out = {}
    for ds, cells in acc.items():
        rows = sorted(
            ((c, mo, og) for c, (mo, og) in cells.items()), key=lambda r: r[1] - r[2]
        )
        if rows:
            out[ds] = rows
    return out


def _chart_s8_percell(percell):
    """Per-dataset, per-cell sel_eff vs Origami, laid out by the SPLIT TREE:
    cells are grouped under their base cell (M|N|K|B tier) -- shown once as a
    bold header -- and each leaf then shows only its SHORT split suffix + a
    diverging delta bar (model - origami; green = model wins, red = Origami
    wins) + the two % values. Grouping keeps labels short despite long full
    names. Base groups are in canonical order; leaves within a group are
    sorted worst-first. Fixed-width SVG in a scrollable box."""
    if not percell:
        return ""
    figs = []
    W, GUT, BARW = 820, 360, 230
    BARX = GUT + 8
    CX = BARX + BARW // 2
    VALX = BARX + BARW + 10
    for ds in percell:
        rows = percell[ds]
        groups = {}
        for cell, mo, og in rows:
            groups.setdefault(cell.split("#")[0], []).append((cell, mo, og))
        bases = sorted(groups, key=_base_cell_key)
        lines = []  # ("base", name, count) | ("leaf", suffix, mo, og, cell)
        for base in bases:
            members = sorted(groups[base], key=lambda r: r[1] - r[2])
            lines.append(("base", base, len(members)))
            for cell, mo, og in members:
                suf = cell[len(base) :] if cell != base else "(unsplit leaf)"
                lines.append(("leaf", suf, mo, og, cell))
        n = len(lines)
        h = max(9, min(15, int(3900 / max(n, 1))))
        H = n * h + 6
        maxd = max((abs(mo - og) for _, mo, og in rows), default=1.0) or 1.0
        scale = (BARW / 2.0 - 4) / maxd
        wins = sum(1 for _, mo, og in rows if mo >= og)
        parts = [
            f'<line x1="{CX}" y1="0" x2="{CX}" y2="{H}" '
            f'stroke="#ccc" stroke-width="1"/>'
        ]
        for i, ln in enumerate(lines):
            y = i * h + 3
            yc = y + h / 2.0
            fs = max(7, min(10, h - 2))
            if ln[0] == "base":
                _, base, cnt = ln
                parts.append(
                    f'<text x="2" y="{yc + fs * 0.34:.1f}" font-size="{fs}" '
                    f'font-weight="600" font-family="ui-monospace,Menlo,monospace">'
                    f'{html.escape(base)} <tspan fill="#999">({cnt})</tspan></text>'
                )
                continue
            _, suf, mo, og, cell = ln
            d = mo - og
            col = "#2da44e" if d >= 0 else "#cf222e"
            if d >= 0:
                parts.append(
                    f'<rect x="{CX:.1f}" y="{y + 0.5:.1f}" '
                    f'width="{d * scale:.1f}" height="{h - 1:.1f}" '
                    f'fill="{col}"/>'
                )
            else:
                parts.append(
                    f'<rect x="{CX + d * scale:.1f}" y="{y + 0.5:.1f}" '
                    f'width="{-d * scale:.1f}" height="{h - 1:.1f}" '
                    f'fill="{col}"/>'
                )
            lbl = suf if len(suf) <= 40 else suf[:39] + "\u2026"
            parts.append(
                f'<text x="16" y="{yc + fs * 0.34:.1f}" font-size="{fs}" '
                f'fill="#444" font-family="ui-monospace,Menlo,monospace">'
                f"{html.escape(lbl)}<title>{html.escape(cell)}\nmodel {mo:.1f}% / "
                f"origami {og:.1f}% (delta {d:+.1f} pp)</title></text>"
            )
            if h >= 10:
                parts.append(
                    f'<text x="{VALX}" y="{yc + fs * 0.34:.1f}" font-size="{fs}" '
                    f'fill="#57606a">{mo:.1f} / {og:.1f}</text>'
                )
        svg = (
            f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:{W}px;height:{H}px;border:none;background:#fff" '
            f'font-family="sans-serif">' + "".join(parts) + "</svg>"
        )
        cap = (
            f"<b>{html.escape(ds)}</b> &mdash; per-cell sel_eff vs Origami, "
            f"grouped by base cell (split tree); each leaf shows its split "
            f"suffix + &Delta;(model&minus;origami). "
            f'<span style="color:#2da44e">green</span> = model wins '
            f'({wins}/{len(rows)} cells), <span style="color:#cf222e">red</span>'
            f" = Origami wins. Right = model% / origami%; hover for full name."
        )
        figs.append(
            f'<figure class="chart" style="max-width:{W + 10}px"><figcaption>'
            f'{cap}</figcaption><div style="max-height:520px;overflow:auto;'
            f'border:1px solid var(--bd);border-radius:8px">{svg}</div></figure>'
        )
    return "".join(figs)


def _chart_s8_dumbbell(percell):
    """Per-dataset DUMBBELL: per cell a connector from origami to model with a
    blue dot (model) + gray dot (origami) on a zoomed sel_eff axis -- shows
    both absolute positions and the gap. Sorted worst-first; scrollable."""
    if not percell:
        return ""
    figs = []
    W, GUT = 760, 300
    AX0, AX1 = GUT + 10, W - 16
    for ds in percell:
        rows = percell[ds]
        n = len(rows)
        h = max(8, min(15, int(3400 / max(n, 1))))
        H = n * h + 12
        lo = max(0, int(min((min(mo, og) for _, mo, og in rows), default=0)) - 2)

        def X(v, lo=lo):
            return AX0 + (AX1 - AX0) * (v - lo) / max(100 - lo, 1)

        parts = []
        for gx in [g for g in (80, 90, 95, 100) if g >= lo]:
            parts.append(
                f'<line x1="{X(gx):.1f}" y1="10" x2="{X(gx):.1f}" '
                f'y2="{H}" stroke="#eee" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{X(gx):.1f}" y="8" font-size="8" '
                f'fill="#999" text-anchor="middle">{gx}</text>'
            )
        for i, (cell, mo, og) in enumerate(rows):
            yc = i * h + h / 2.0 + 11
            fs = max(7, min(10, h - 2))
            xm, xo = X(mo), X(og)
            col = "#2da44e" if mo >= og else "#cf222e"
            parts.append(
                f'<line x1="{xo:.1f}" y1="{yc:.1f}" x2="{xm:.1f}" '
                f'y2="{yc:.1f}" stroke="{col}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<circle cx="{xo:.1f}" cy="{yc:.1f}" r="2.4" ' f'fill="#888"/>'
            )
            parts.append(
                f'<circle cx="{xm:.1f}" cy="{yc:.1f}" r="2.4" ' f'fill="#0969da"/>'
            )
            lbl = cell if len(cell) <= 46 else cell[:45] + "\u2026"
            parts.append(
                f'<text x="2" y="{yc + fs * 0.34:.1f}" font-size="{fs}"'
                f' font-family="ui-monospace,Menlo,monospace">'
                f"{html.escape(lbl)}<title>{html.escape(cell)}\nmodel "
                f"{mo:.1f}% / origami {og:.1f}%</title></text>"
            )
        svg = (
            f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:{W}px;height:{H}px;border:none;background:#fff" '
            f'font-family="sans-serif">' + "".join(parts) + "</svg>"
        )
        cap = (
            f"<b>{html.escape(ds)}</b> &mdash; dumbbell: "
            f'<span style="color:#0969da">&#9679; model</span> vs '
            f'<span style="color:#888">&#9679; origami</span> on the sel_eff '
            f"axis ({lo}&ndash;100%); green line = model ahead, red = behind. "
            f"{n} cells, sorted worst-first; hover for the full name."
        )
        figs.append(
            f'<figure class="chart" style="max-width:{W + 10}px"><figcaption>'
            f'{cap}</figcaption><div style="max-height:470px;overflow:auto;'
            f'border:1px solid var(--bd);border-radius:8px">{svg}</div></figure>'
        )
    return "".join(figs)


def _chart_s8_grouped(percell, top_n=30):
    """Per-dataset GROUPED two-bar: per cell a model bar (top) + origami bar
    (bottom). Clearest with few rows, so limited to the worst `top_n` cells."""
    if not percell:
        return ""
    figs = []
    W, GUT = 760, 320
    AX0, AX1 = GUT + 10, W - 60
    for ds in percell:
        allrows = percell[ds]
        rows = allrows[:top_n]
        n = len(rows)
        rh = 26
        H = n * rh + 8
        lo = max(0, int(min((min(mo, og) for _, mo, og in rows), default=0)) - 2)

        def X(v, lo=lo):
            return AX0 + (AX1 - AX0) * (v - lo) / max(100 - lo, 1)

        parts = []
        for gx in [g for g in (80, 90, 100) if g >= lo]:
            parts.append(
                f'<line x1="{X(gx):.1f}" y1="0" x2="{X(gx):.1f}" '
                f'y2="{H}" stroke="#eee" stroke-width="1"/>'
            )
        for i, (cell, mo, og) in enumerate(rows):
            y = i * rh + 4
            lbl = cell if len(cell) <= 52 else cell[:51] + "\u2026"
            parts.append(
                f'<text x="2" y="{y + 13:.1f}" font-size="9" '
                f'font-family="ui-monospace,Menlo,monospace">'
                f"{html.escape(lbl)}<title>{html.escape(cell)}</title>"
                f"</text>"
            )
            parts.append(
                f'<rect x="{X(lo):.1f}" y="{y:.1f}" '
                f'width="{X(mo) - X(lo):.1f}" height="9" fill="#0969da"/>'
            )
            parts.append(
                f'<text x="{X(mo) + 3:.1f}" y="{y + 8:.1f}" '
                f'font-size="8" fill="#0969da">{mo:.1f}</text>'
            )
            parts.append(
                f'<rect x="{X(lo):.1f}" y="{y + 11:.1f}" '
                f'width="{X(og) - X(lo):.1f}" height="9" fill="#888"/>'
            )
            parts.append(
                f'<text x="{X(og) + 3:.1f}" y="{y + 19:.1f}" '
                f'font-size="8" fill="#888">{og:.1f}</text>'
            )
        svg = (
            f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:{W}px;height:{H}px;border:none;background:#fff" '
            f'font-family="sans-serif">' + "".join(parts) + "</svg>"
        )
        cap = (
            f"<b>{html.escape(ds)}</b> &mdash; grouped bars: "
            f'<span style="color:#0969da">model</span> (top) vs '
            f'<span style="color:#888">origami</span> (bottom), worst '
            f"{min(top_n, len(allrows))} of {len(allrows)} cells. "
            f"Axis {lo}&ndash;100%."
        )
        figs.append(
            f'<figure class="chart" style="max-width:{W + 10}px"><figcaption>'
            f'{cap}</figcaption><div style="max-height:470px;overflow:auto;'
            f'border:1px solid var(--bd);border-radius:8px">{svg}</div></figure>'
        )
    return "".join(figs)


# ── assemble ──────────────────────────────────────────────────────────────

_SHAPE_RE = re.compile(
    r"\bM:\s*(\d+),\s*N:\s*(\d+),\s*K:\s*(\d+)\b.*?\bbatch_count:\s*(\d+)"
)


def _held_out_by_category(run_dir):
    """Categorize the held-out GEMMs (each round's NEW shapes, the
    generalization test set) into the 20 canonical geko categories AND into
    their deployed leaf cell, attributing to each shape the held-out sel_eff
    measured for the leaf it routes to.

    Returns (cat_rows, base_rows, n_total) or (None, None, 0) if the routing
    library or breakdown isn't available. Per-shape categorization is exact
    (from shapes.yaml dims); the sel_eff a shape gets is its LEAF's geomean
    held-out sel_eff (we have no per-GEMM held-out score cached), so the
    per-category sel_eff is a GEMM-weighted mean of the leaf values."""
    # routing lib (cwd is usually the pipeline root; be robust either way)
    _ml = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ml not in sys.path:
        sys.path.insert(0, _ml)
    try:
        from lib.grid import cell_key
        from lib.subcells import assign_subcell, load_cumulative_split_tree
    except Exception:
        return None, None, 0

    bp = os.path.join(run_dir, "held_out_sel_eff_breakdown.json")
    if not os.path.isfile(bp):
        return None, None, 0
    try:
        per_cell = json.load(open(bp)).get("per_cell") or {}
    except Exception:
        return None, None, 0
    held = {
        lab: v["sel_eff_new"]
        for lab, v in per_cell.items()
        if v.get("sel_eff_new") is not None
    }
    held_o = {
        lab: v["origami_sel_eff_new"]
        for lab, v in per_cell.items()
        if v.get("origami_sel_eff_new") is not None
    }

    def _inherit(m, lab):
        cur = lab
        while True:
            if cur in m:
                return m[cur]
            j = cur.rfind("#")
            if j < 0:
                return None
            cur = cur[:j]

    from pathlib import Path

    round_dirs = sorted(
        glob.glob(os.path.join(run_dir, "round_*")),
        key=lambda p: (
            int(re.search(r"round_(\d+)", p).group(1))
            if re.search(r"round_(\d+)", p)
            else -1
        ),
    )
    try:
        tree = load_cumulative_split_tree([Path(d) for d in round_dirs])
    except Exception:
        tree = {}

    # Per-category and per-base-cell accumulators.
    cat = {
        c: {"n": 0, "slog": 0.0, "sn": 0, "olog": 0.0, "on": 0} for c in GEKO_CATEGORIES
    }
    base_acc = {}  # base 96-grid label -> {"n":.., "leaves":set()}
    n_total = 0
    for d in round_dirs:
        # held-out eval only ran for rounds with a stage04b decision.
        if not os.path.isfile(os.path.join(d, "stage04b", "decisions.json")):
            continue
        sy = os.path.join(d, "stage01", "shapes.yaml")
        if not os.path.isfile(sy):
            continue
        try:
            txt = open(sy, errors="ignore").read()
        except Exception:
            continue
        for mm in _SHAPE_RE.finditer(txt):
            m, n, k, b = (
                int(mm.group(1)),
                int(mm.group(2)),
                int(mm.group(3)),
                int(mm.group(4)),
            )
            n_total += 1
            c = _categorize_gemm(m, n, k, b)
            rec = cat[c]
            rec["n"] += 1
            try:
                base = cell_key(m, n, k, b)
                leaf = assign_subcell(base, m, n, k, b, tree)
            except Exception:
                base, leaf = None, None
            if base is not None:
                ba = base_acc.setdefault(base, {"n": 0, "leaves": set()})
                ba["n"] += 1
                if leaf:
                    ba["leaves"].add(leaf)
            if leaf:
                se = _inherit(held, leaf)
                oe = _inherit(held_o, leaf)
                if se and se > 0:
                    rec["slog"] += math.log(se)
                    rec["sn"] += 1
                if oe and oe > 0:
                    rec["olog"] += math.log(oe)
                    rec["on"] += 1

    if not n_total:
        return None, None, 0

    def _gm(rec, lk, nk):
        return (math.exp(rec[lk] / rec[nk]) * 100.0) if rec[nk] else None

    cat_rows = []
    for c in GEKO_CATEGORIES:
        rec = cat[c]
        if not rec["n"]:
            continue
        sm = _gm(rec, "slog", "sn")
        om = _gm(rec, "olog", "on")
        cat_rows.append(
            {
                "category": c,
                "n": rec["n"],
                "held": round(sm, 2) if sm is not None else None,
                "origami": round(om, 2) if om is not None else None,
                "delta": (
                    round(sm - om, 2) if (sm is not None and om is not None) else None
                ),
            }
        )
    base_rows = sorted(
        (
            {"cell": b, "n": v["n"], "leaves": len(v["leaves"])}
            for b, v in base_acc.items()
        ),
        key=lambda r: r["n"],
        reverse=True,
    )
    return cat_rows, base_rows, n_total


def build(run_dir, out_path):
    run_dir = run_dir.rstrip("/")
    # Rebuild the held-out breakdown from per-round training data if the
    # orchestrator never wrote it (e.g. stage 8 was cancelled). This keeps the
    # held-out aggregate, 3D cell map and category breakdown -- all independent
    # of stage 7/8 -- from going blank on an interrupted run.
    _ensure_breakdown(run_dir)
    traj = _round_trajectory(run_dir)
    train_traj = _training_trajectory(run_dir)
    last_round, cells = _final_cells(run_dir)
    held = _held_out_aggregate(run_dir)
    s7 = _stage7(run_dir)
    s8 = _stage8(run_dir)
    timing, total_wall = _timing(run_dir)
    mp = [c["model"] for c in cells if c.get("model") is not None]
    train_mean = round(sum(mp) / len(mp), 2) if mp else None

    # Per-round GEMM accounting: new shapes benched this round (stage04b
    # load; round_0 has no stage04b so it trains all its own shapes), the
    # running cumulative corpus, and n_gemms_trained (already on each row)
    # = ONLY the GEMMs in the cells retrained this round (a subset).
    gen_new = {r["round"]: r["n_gemms"] for r in traj}
    _cum = 0
    for r in train_traj:
        new = gen_new.get(r["round"])
        if new is None:
            new = r["n_gemms"]
        _cum += new or 0
        r["new_gemms"] = new
        r["cum_gemms"] = _cum

    # ── Runtime-weighted selection efficiency (the metric that tracks
    #    wall-clock) ────────────────────────────────────────────────────────
    # The headline sel_eff is a count/GEMM-weighted geomean, so a regression on
    # a few expensive GEMMs is invisible. Here each cell's sel_eff is weighted
    # by its RUNTIME (~ n_gemms * oracle_us): runtime_eff = total oracle time /
    # total achieved time = Sum(n*winner) / Sum(n*pick). This mirrors a real
    # trace's wall-clock far better than the equal-weight geomean.
    #
    # ALL us here are HELD-OUT (stage04b, unseen shapes), NOT in-sample: that
    # keeps this section consistent with its held-out sel_eff columns and avoids
    # the in-sample-vs-held-out confusion. Held-out winner/pick/origami geomeans
    # are over the SAME unseen shapes, so winner <= pick always and a higher
    # held-out sel_eff <=> a faster (lower) held-out pick us.
    def _rt_eff(pick_key):
        num = den = 0.0
        for c in cells:
            n = c.get("n_gemms") or 0
            w = c.get("held_w_us")
            p = c.get(pick_key)
            if n and w and p and w > 0 and p > 0:
                num += n * w
                den += n * p
        return (num / den * 100.0) if den else None

    _rt_m = _rt_eff("held_pick_us")
    _rt_o = _rt_eff("held_ori_pick_us")
    rt_eff = {
        "model": round(_rt_m, 2) if _rt_m is not None else None,
        "origami": round(_rt_o, 2) if _rt_o is not None else None,
        "delta": (
            round(_rt_m - _rt_o, 2)
            if (_rt_m is not None and _rt_o is not None)
            else None
        ),
    }

    # ── Worst cells: runtime-weighted held-out "extra time vs Origami" + the
    #    cells whose HELD-OUT sel_eff is BELOW Origami (negative delta).
    #    extra_us = n_gemms * (held_model_pick - held_origami_pick): positive =
    #    the model spends MORE time than Origami per GEMM on this cell's unseen
    #    shapes, scaled by how many GEMMs route there. This is what drives a
    #    real-trace (out-of-distribution) regression.
    _tot_w = (
        sum(
            (c.get("n_gemms") or 0) * (c.get("held_w_us") or 0)
            for c in cells
            if c.get("held_w_us")
        )
        or 0.0
    )
    worst = []
    for c in cells:
        n = c.get("n_gemms") or 0
        dp, op, w = (
            c.get("held_pick_us"),
            c.get("held_ori_pick_us"),
            c.get("held_w_us"),
        )
        extra = (n * (dp - op)) if (n and dp and op) else None
        rw = (n * w / _tot_w * 100.0) if (w and _tot_w) else None
        worst.append({**c, "extra_us": extra, "rw_share": rw})
    _neg = [c for c in worst if c.get("held_delta") is not None and c["held_delta"] < 0]

    # Rank by HELD-OUT delta (the generalization / OOD-predictive signal); cells
    # with no held-out measurement at all sort last.
    def _wkey(c):
        hd = c.get("held_delta")
        return (0, hd) if hd is not None else (1, 999)

    _by_extra = sorted(worst, key=_wkey)[:25]
    _tot_extra = sum(
        c["extra_us"] for c in worst if c.get("extra_us") and c["extra_us"] > 0
    )

    # Compact per-row data for the CLIENT-side sortable table (mirrors the
    # per-cell summary table: click any header to sort). Every us value here is
    # a GEOMEAN over the cell's GEMMs (not one GEMM), so the columns are
    # labelled "... geomean us" rather than "... pick us".
    _wkeys = (
        "cell",
        "n_gemms",
        "rw_share",
        "held_w_us",
        "held_pick_us",
        "held_ori_pick_us",
        "extra_us",
        "held",
        "held_ori",
        "held_delta",
        "cap",
    )
    worst_data = [{k: c.get(k) for k in _wkeys} for c in _by_extra]

    cfg_path = os.path.join(run_dir, "config_used.yaml")
    cfg_txt = open(cfg_path, errors="ignore").read() if os.path.isfile(cfg_path) else ""

    summ = {
        "run": os.path.basename(run_dir),
        "rounds": len(glob.glob(os.path.join(run_dir, "round_*"))),
        "last_round": last_round,
        "held": held,  # cross-round aggregate (trust this)
        "train_mean": train_mean,  # in-sample mean (reference)
        "rt_eff": rt_eff,  # runtime-weighted (tracks wall-clock)
        "n_neg_cells": len(_neg),
        "n_cells": len(cells),
        "s7": s7,
        "s8": s8,
        "total_wall": total_wall,
        "bin_mb": _bin_size_mb(run_dir),
        "total_shapes": (
            max((r.get("cum_gemms") or 0) for r in train_traj) if train_traj else None
        ),
        # per-dataset, per-cell [cell, model%, origami%] for the sortable
        # Stage-8 diverging-bar charts (rendered client-side).
        "s8pc": {
            ds: [[c, mo, og] for (c, mo, og) in rows]
            for ds, rows in _stage8_percell(run_dir).items()
        },
    }

    # server-rendered sections
    # Prefer per-GEMM; older runs that predate per-GEMM logging fall back to
    # equal-weight for that dataset, flagged with a dagger so no cell is blank.
    _eqfb = [False]

    def _row_cells(r):
        m, o = r.get("model_gw"), r.get("origami_gw")
        mk = ""
        if not isinstance(m, (int, float)) or not isinstance(o, (int, float)):
            m, o, mk = r.get("model"), r.get("origami"), "<sup>&dagger;</sup>"
        if not isinstance(m, (int, float)) or not isinstance(o, (int, float)):
            return "&ndash;", "&ndash;", "&ndash;"
        if mk:
            _eqfb[0] = True
        d = m - o
        return (f"{m:.2f}{mk}", f"{o:.2f}{mk}", f'{"+" if d >= 0 else ""}{d:.2f}{mk}')

    s8_rows = "".join(
        (
            lambda mc, oc, dc: f'<tr><td class="c">{html.escape(r["name"])}</td>'
            f"<td>{mc}</td><td>{oc}</td><td>{dc}</td></tr>"
        )(*_row_cells(r))
        for r in s8
    )
    s8_fb_note = (
        " Datasets from older runs that predate per-GEMM logging "
        "show equal-weight (&dagger;)."
        if _eqfb[0]
        else ""
    )

    def _f2(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else ""

    traintraj_rows = "".join(
        f'<tr><td class="c">{html.escape(r["round"])}</td>'
        f'<td>{_f2(r["model"])}</td><td>{_f2(r["deployed"])}</td>'
        f'<td>{_f2(r["origami"])}</td>'
        f'<td>{(r["n_cells"] or 0):,}</td>'
        f'<td>{(r.get("new_gemms") or 0):,}</td>'
        f'<td>{(r.get("cum_gemms") or 0):,}</td>'
        f'<td>{(r["n_gemms"] or 0):,}</td></tr>'
        for r in train_traj
    )
    # Timing grid: one aligned column per stage that ran this run (blank when
    # a round skipped a stage), + the per-round total. Much more legible than
    # the old "·"-joined single cell.
    _STG_LABEL = {
        "1": "shapes",
        "2": "probe",
        "3": "bench",
        "4": "enrich",
        "4.5": "validate",
        "5": "train",
        "6": "deploy",
        "7": "rank-time",
        "8": "sel-eff",
    }
    _present, _rd_stage = [], []
    for rd in timing:
        smap = {}
        for s in rd["stages"]:
            mnum = re.search(r"Stage ([\d.]+)", s["name"])
            if mnum:
                smap[mnum.group(1)] = s["time"]
                if mnum.group(1) not in _present:
                    _present.append(mnum.group(1))
        _rd_stage.append((rd, smap))
    _present.sort(key=float)
    _th = "".join(f'<th>{_STG_LABEL.get(n, "s" + n)}</th>' for n in _present)
    _trows = ""
    for rd, smap in _rd_stage:
        _tds = "".join(f'<td>{smap.get(n, "")}</td>' for n in _present)
        _trows += (
            f'<tr><td class="c">{html.escape(rd["round"])}</td>{_tds}'
            f'<td>{rd["total"] or ""}</td></tr>'
        )
    _trows += (
        f'<tr style="font-weight:600;border-top:2px solid #999">'
        f'<td class="c">TOTAL (wall-clock)</td>'
        f'{"<td></td>" * len(_present)}<td>{total_wall or ""}</td></tr>'
    )
    timing_table = (
        f'<table><thead><tr><th class="c">round</th>{_th}'
        f"<th>round total</th></tr></thead><tbody>{_trows}</tbody>"
        f"</table>"
    )
    tree_txt = _cell_tree_ascii(cells)

    # Runtime-weighted section (server-rendered). Explains the metric-vs-
    # wall-clock gap and lists the cells that actually cost runtime.
    def _pe(v):
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "n/a"

    _rtm, _rto, _rtd = rt_eff["model"], rt_eff["origami"], rt_eff["delta"]
    _rt_verdict = ""
    if _rtd is not None:
        _rt_verdict = (
            f" Runtime-weighted on the <b>held-out</b> "
            f"(stage-4b, unseen) GEMMs, the model is "
            f"<b>{_rtd:+.2f} pp</b> vs Origami "
            f"(model {_pe(_rtm)} vs origami {_pe(_rto)}) &mdash; "
            f"negative here means the generalization gap dominates "
            f"the large compute-bound cells that drive a real "
            f"workload's wall-clock."
        )
    worst_html = (
        f"<h2>Runtime-weighted selection efficiency &amp; worst cells "
        f"(held-out)</h2>"
        f'<p class="cap">The headline sel_eff is a count/GEMM-weighted geomean, '
        f"so a loss on a handful of expensive GEMMs is invisible. "
        f"<b>Runtime-weighted</b> efficiency weights each cell by its runtime "
        f"(&approx; n_gemms &times; oracle &micro;s) = &Sigma;(n&middot;winner) / "
        f"&Sigma;(n&middot;pick); it tracks a real trace's wall-clock.{_rt_verdict}</p>"
        f'<p class="cap">Everything in this table is <b>held-out</b> (stage-4b, '
        f"on each cell's NEW/unseen shapes) so the &micro;s columns and the "
        f"sel_eff columns are consistent (held oracle &le; held model/origami; "
        f"a higher held % &hArr; a faster held pick &micro;s). "
        f"<b>{len(_neg)}</b> of {len(cells)} deployed leaf cells have a NEGATIVE "
        f"held-out &Delta; vs Origami. Sum of positive per-cell held-out extra "
        f"time vs Origami &approx; <b>{_tot_extra:,.0f}</b> weighted &micro;s. "
        f"Ranked by <b>held-out &Delta;</b> (worst generalization first). extra "
        f"&micro;s vs Origami = n_gemms&middot;(held_model_pick &minus; "
        f"held_origami_pick) (red = negative held-out &Delta;). Every &micro;s "
        f"column is a per-cell <b>geomean</b> over the cell's unseen GEMMs (not "
        f"one GEMM); <b>click any column header to sort</b>.</p>"
        f'<div style="max-height:560px;overflow:auto;border:1px solid var(--bd);'
        f'border-radius:8px;margin-bottom:6px">'
        f'<table style="margin-bottom:0;font-size:11px;width:100%">'
        f'<thead><tr id="worsth"></tr></thead>'
        f'<tbody id="tbody_worst"></tbody></table></div>'
    )

    # ── Held-out GEMMs by canonical category + per-(base)cell counts ────────
    cat_rows, base_rows, n_cat_total = _held_out_by_category(run_dir)

    def _pct(v):
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "&ndash;"

    if cat_rows:
        _cr = "".join(
            (
                lambda d: f'<tr{(" class=\'lose\'" if (d is not None and d < 0) else "")}>'
                f'<td class="c">{html.escape(r["category"])}</td>'
                f'<td>{r["n"]:,}</td>'
                f'<td>{_pct(r["held"])}</td><td>{_pct(r["origami"])}</td>'
                f'<td>{("+" if (d or 0) >= 0 else "")}'
                f'{(f"{d:.2f}" if isinstance(d, (int, float)) else "&ndash;")}</td>'
                f"</tr>"
            )(r.get("delta"))
            for r in cat_rows
        )
        cat_html = (
            f"<h2>Held-out GEMMs by canonical category</h2>"
            f'<p class="cap">The {n_cat_total:,} held-out GEMMs (every active '
            f"round's NEW/unseen shapes &mdash; the generalization test set) "
            f"classified into the 20 canonical "
            f"<code>geko</code> categories (batched &rarr; degenerate dims "
            f"&rarr; multi-large &rarr; single-large &rarr; size buckets; "
            f"&quot;large&quot; = dim&gt;8192). <b>held-out %</b> / <b>origami %</b> "
            f"are the GEMM-weighted mean of the deployed LEAF cell each shape "
            f"routes to (we cache held-out sel_eff per cell, not per GEMM, so "
            f"within-cell variance is folded into the leaf value). "
            f"Red = model below Origami.</p>"
            f'<table style="max-width:780px"><thead><tr>'
            f'<th class="c">category</th><th># GEMMs</th><th>held-out %</th>'
            f"<th>origami %</th><th>&Delta; (held&minus;ori)</th></tr></thead>"
            f"<tbody>{_cr}</tbody></table>"
        )
    else:
        cat_html = (
            "<h2>Held-out GEMMs by canonical category</h2>"
            '<p class="mut">category breakdown unavailable (no '
            "shapes.yaml / held-out breakdown for this run).</p>"
        )

    if base_rows:
        _tot_n = sum(r["n"] for r in base_rows) or 1
        _br = "".join(
            f'<tr><td class="c">{html.escape(r["cell"])}</td>'
            f'<td>{r["n"]:,}</td><td>{r["n"] / _tot_n * 100:.1f}%</td>'
            f'<td>{r["leaves"]:,}</td></tr>'
            for r in base_rows
        )
        cells_html = (
            f"<h2>Held-out GEMMs per cell (base 96-grid, under the M/N/K/Batch "
            f"caps)</h2>"
            f'<p class="cap">How the {n_cat_total:,} held-out GEMMs distribute '
            f"across the base <code>M|N|K|Batch</code> tier grid (the caps: M/N "
            f"Large&le;524288, K LargeK&le;131072, Batch Bany sampled). "
            f"<b>deployed leaves</b> = how many split sub-cells that base cell "
            f"was carved into. Sorted by GEMM count.</p>"
            f'<div style="max-height:560px;overflow:auto;border:1px solid '
            f'var(--bd);border-radius:8px;margin-bottom:6px">'
            f'<table style="margin-bottom:0;width:100%"><thead><tr>'
            f'<th class="c">base cell (M|N|K|Batch)</th><th># held-out GEMMs</th>'
            f"<th>% of held-out</th><th>deployed leaves</th></tr></thead>"
            f"<tbody>{_br}</tbody></table></div>"
        )
    else:
        cells_html = ""

    # Inline SVG charts (server-rendered, no JS/CDN). Each is "" when its
    # source data is absent, so the placeholder simply collapses.
    chart_trend = _chart_trend(train_traj)
    chart_cells = _chart_cell_dist(cells)
    chart_held = _chart_held_dist(cells)
    cell3d = _cell3d_b64(run_dir)
    cell3d_html = "".join(
        f'<figure class="chart" style="max-width:520px">'
        f"<figcaption>B = {bt}</figcaption>"
        f'<img alt="3D cell map B={bt}" '
        f'style="width:100%;border:1px solid var(--bd);border-radius:8px" '
        f'src="{uri}"></figure>'
        for bt, uri in cell3d.items()
    ) or (
        '<p class="mut">3D cell map unavailable (matplotlib missing or no '
        "cell breakdown for this run).</p>"
    )

    H = (
        _HTML.replace("__CELL3D__", cell3d_html)
        .replace("__TRAJ__", json.dumps(traj))
        .replace("__CELLS__", json.dumps(cells))
        .replace("__SUMM__", json.dumps(summ))
        .replace("__S8ROWS__", s8_rows)
        .replace("__S8FBNOTE__", s8_fb_note)
        .replace("__TRAINTRAJ__", traintraj_rows)
        .replace("__TIMINGTABLE__", timing_table)
        .replace("__TREE__", html.escape(tree_txt))
        .replace("__CONFIG__", html.escape(cfg_txt))
        .replace("__TITLE__", html.escape(summ["run"]))
        .replace("__CHART_TREND__", chart_trend)
        .replace("__CHART_CELLS__", chart_cells)
        .replace("__CHART_HELD__", chart_held)
        .replace("__WORSTCELLS__", worst_html)
        .replace("__WORST__", json.dumps(worst_data))
        .replace("__CATTABLE__", cat_html)
        .replace("__CELLCOUNTS__", cells_html)
    )
    open(out_path, "w").write(H)
    return out_path


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report - __TITLE__</title><style>
:root{--bd:#d8dee4;--mut:#57606a;--win:#e6ffec;--lose:#ffebe9;--weak:#fff8c5;}
body{font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2328;margin:0;padding:24px;max-width:1760px;}
/* keep prose readable while letting dense tables/plots use the full width */
p,h1,h2,.stats,details{max-width:1200px;}
h1{font-size:20px;margin:0 0 2px;} h2{font-size:15px;margin:24px 0 8px;}
.cap{color:var(--mut);font-size:12.5px;margin:0 0 16px;} .mut{color:var(--mut);font-size:11.5px;}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;}
.stat{border:1px solid var(--bd);border-radius:8px;padding:9px 13px;min-width:120px;}
.stat .v{font-size:21px;font-weight:600;} .stat .l{color:var(--mut);font-size:11.5px;margin-top:2px;}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin-bottom:6px;}
th,td{padding:4px 9px;border-bottom:1px solid var(--bd);text-align:right;white-space:nowrap;}
th{position:sticky;top:0;background:#f6f8fa;font-weight:600;}
th.sort{cursor:pointer;}
td.c,th.c{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:12px;}
tr.win td{background:var(--win);} tr.lose td{background:var(--lose);} tr.weak td{background:var(--weak);}
ul.tree{list-style:none;padding-left:14px;font-size:12.5px;} ul.tree ul{padding-left:16px;}
ul.tree code{font-family:ui-monospace,Menlo,monospace;} summary{cursor:pointer;}
details>summary{margin:1px 0;} pre.cfg{background:#f6f8fa;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;font-size:12px;max-height:420px;}
figure.chart{margin:6px 0 16px;max-width:782px;}
figure.chart figcaption{color:var(--mut);font-size:12px;margin:0 0 5px;}
figure.chart svg{width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;background:#fff;}
.s8bars{max-height:520px;overflow:auto;border:1px solid var(--bd);border-radius:8px;padding:5px 10px;font:12.5px/1.4 ui-monospace,Menlo,monospace;}
.s8r{display:flex;align-items:center;height:18px;white-space:nowrap;}
.s8lab{flex:0 0 360px;overflow:hidden;text-overflow:ellipsis;color:#444;}
.s8bar{flex:0 0 240px;position:relative;height:13px;}
.s8mid{position:absolute;left:120px;top:-2px;bottom:-2px;width:1px;background:#ccc;}
.s8fill{position:absolute;top:1px;height:11px;border-radius:1px;}
.s8v{flex:0 0 200px;text-align:right;color:#57606a;padding-left:12px;white-space:nowrap;}
figure.chart select{font-size:12px;}
</style></head><body>
<h1>Run report &mdash; __TITLE__</h1>
<p class="cap" id="cap"></p>
<div class="stats" id="stats"></div>
<p class="cap" style="margin-top:8px"><b>Cell-grid tiers</b> &mdash; every cell is <code>M-tier | N-tier | K-tier | Batch</code> (classified by GEMM size; split suffixes like <code>#K&gt;512</code> further subdivide a tier). <b>M, N</b>: Tiny&nbsp;&le;32 · Small&nbsp;33&ndash;128 · Mid&nbsp;129&ndash;512 · Large&nbsp;&gt;512 (sampled to 524288). <b>K</b>: TinyK&nbsp;&le;32 · MidK&nbsp;33&ndash;512 · LargeK&nbsp;&gt;512 (sampled to 131072). <b>Batch</b>: Bnone&nbsp;= unbatched (B=1) · Bany&nbsp;= batched (B&gt;1, sampled to ~8192).</p>

<h2>Stage 7 &mdash; selection time &amp; C++/Python pick parity</h2>
<p class="cap">Table 1 &mdash; per-call selection cost (&micro;s): <b>rank_configs()</b> CPU wall-clock (min/median/mean/max) and the <b>end-to-end</b> C++ "Solution selection time" (the full findTopSolutions span, &gt; rank_configs alone; median/mean). Table 2 &mdash; pick parity, mean speedup, and candidate count.</p>
<table id="s7time" style="max-width:780px"><tbody id="tbody_s7time"></tbody></table>
<table id="s7meta" style="max-width:480px;margin-top:14px"><tbody id="tbody_s7meta"></tbody></table>

<h2>Stage 8 &mdash; selection efficiency on external held-out traces</h2>
<p class="cap"><b>Per-GEMM</b> selection efficiency = geomean over ALL GEMMs, each GEMM counted <b>once/equally</b> (a flat per-GEMM average, NOT unequal weighting). It is partition-independent, so for Origami it is purely its own data and is identical across runs on the same dataset &mdash; the right basis for cross-run comparison. &Delta; = model &minus; origami (positive = model picks better kernels).__S8FBNOTE__</p>
<table><thead><tr><th class="c">dataset</th><th>model % (per-GEMM)</th><th>origami % (per-GEMM)</th><th>&Delta; (model &minus; origami)</th></tr></thead>
<tbody>__S8ROWS__</tbody></table>
<p class="cap">Per-cell selection efficiency vs Origami (one diverging &Delta; bar per deployed leaf). Use the dropdown to sort by &Delta; or by cell label.</p>
<div id="s8charts"></div>

<h2>Generalization trajectory (stage04b: trained ML model's performance on each round's new/unknown shapes)</h2>
<table id="traj"><thead><tr><th class="c">round</th><th>model sel_eff %</th><th>origami %</th><th>model - origami</th><th>n_gemms</th></tr></thead><tbody id="tbody_traj"></tbody></table>

<h2>Training-phase selection efficiency per round (stage05: in-sample / training-set)</h2>
<p class="cap">Per-round in-sample model / deployed (smart-K) / origami selection efficiency. cells = trained this round; GEMMs: new this round / cumulative corpus / retrained-this-round.</p>
<table id="traintraj"><thead><tr><th class="c">round</th><th>model %</th><th>deployed %</th><th>origami %</th><th>cells</th><th>new GEMMs</th><th>cum GEMMs</th><th>retrained GEMMs</th></tr></thead><tbody>__TRAINTRAJ__</tbody></table>

<h2 id="cellh">Per-cell summary (final iteration)</h2>
<p class="cap">Two self-consistent groups, each &Delta; = the two columns beside it. <b>HELD-OUT</b> (generalization, stage-4b): <b>held-out %</b> = model on the cell's NEW/unseen shapes, <b>origami % (held-out)</b> = Origami on those <i>same</i> shapes, <b>&Delta; held-out</b> = held-out &minus; origami%(held-out). <b>DEPLOYED</b> (in-sample, on the cell's own benched GEMMs): <b>model %</b> = the trained model's full ranking BEFORE the smart-K cap, <b>deployed %</b> = smart-K production path, <b>K&Delta;</b> = deployed &minus; model (how much the smart-K candidate cap changes the pick &mdash; ~0 means K is harmless), <b>origami %</b> = Origami in-sample, <b>&Delta; (deployed&minus;origami)</b> = deployed &minus; origami. Row color is by <b>held-out</b> (green priority): <span style="background:var(--win);padding:0 4px">green &ge; 95%</span> <span style="background:var(--lose);padding:0 4px">red = &ge;2pp below its origami</span> <span style="background:var(--weak);padding:0 4px">yellow &lt; 88%</span>. <b>K</b> = candidate signatures ranked (adaptive). <b>ed/hd/ih</b> = embed / hidden / interaction-MLP widths. Held-out with <b>*</b> is <b>inherited from the cell's parent</b> (final-split leaf with no own measurement).</p>
<div style="max-height:640px;overflow-y:auto;overflow-x:auto;border:1px solid var(--bd);border-radius:8px;margin-bottom:6px">
<table id="cells" style="margin-bottom:0;font-size:11px;width:100%"><thead><tr id="chead"></tr></thead><tbody id="tbody_cells"></tbody></table></div>
__CHART_HELD__
__CHART_CELLS__
__WORSTCELLS__
__CATTABLE__
__CELLCOUNTS__

<h2>Final cell design (split tree)</h2>
<p class="cap">Leaf cells with their split path. Stats per leaf: n=training GEMMs, model=training sel_eff, &Delta;=model&minus;origami, K=smartK.</p>
<pre class="cfg" style="max-height:560px">__TREE__</pre>

<h2>Cell design in 3D (M / N / K) by batch tier</h2>
<p class="cap">Each box is a deployed leaf cell in log<sub>10</sub> M/N/K space; faint wireframe = unsplit base cell, colored = refined (deeper split = darker). Left: B = Bany (batched); right: B = Bnone (non-batched).</p>
<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">__CELL3D__</div>

<h2>Timing / timeline</h2>
<p class="cap">Wall-clock per stage, per round (blank = stage not run that round). Stages: 1 shapes / 2 probe / 3 bench / 4 enrich / 4.5 validate / 5 train / 6 deploy / 7 rank-time / 8 sel-eff.</p>
__TIMINGTABLE__

<h2>Configuration used</h2>
<details><summary>config_used.yaml</summary><pre class="cfg">__CONFIG__</pre></details>

<script>
const TRAJ=__TRAJ__, CELLS=__CELLS__, S=__SUMM__;
// NaN != null in JS, so a NaN held-out value used to render literally as
// "NaN%" -- which reads as a computed number instead of "not measured".
const _fin=v=>(v==null||(typeof v==="number"&&!isFinite(v)))?null:v;
const hm=_fin(S.held?S.held.model:null), ho=_fin(S.held?S.held.origami:null);
document.getElementById("cap").textContent="Run "+S.run+" - "+S.rounds+" rounds"+(S.total_wall?(" - wall-clock "+S.total_wall):"")+". Click a per-cell header to sort.";
const hy=S.held?S.held.hybrid:null;
const stats=[
  ["final held_out (MODEL)",(hm!=null?hm+"%":"n/a")],
  ["final held_out (ORIGAMI)",(ho!=null?ho+"%":"n/a")],
  ["hybrid held_out (best-per-cell)",(hy!=null?hy+"%":"n/a")+(S.held&&S.held.hybrid_gain!=null?` (+${S.held.hybrid_gain} vs model`+(S.held.model_win_frac!=null?`; model wins ${S.held.model_win_frac}% of cells)`:")"):"")],
  ["model - origami",(S.held&&S.held.delta!=null?(S.held.delta>=0?"+":"")+S.held.delta+" pp":"n/a")],
  ["held-out cells",(S.held?S.held.n_cells:S.n_cells)],
  ["training-set mean (in-sample)",(S.train_mean!=null?S.train_mean+"%":"n/a")],
  ["deployed .bin size",(S.bin_mb!=null?S.bin_mb+" MB":"n/a")],
  ["GEMM shapes (offline-tune)",(S.total_shapes!=null?S.total_shapes.toLocaleString():"n/a")],
  ["E2E pipeline time",(S.total_wall||"n/a")],
];
if(S.s7&&S.s7.ml_mean_us!=null)stats.push(["average selection time (ml_rec)",S.s7.ml_mean_us+" us"]);
document.getElementById("stats").innerHTML=stats.map(([l,v])=>`<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

// stage 7 -- rank_configs() CPU wall-clock (us): min / median / mean / max per path
const s7=S.s7||{};let s7html="";
if(s7.ml_us!=null||s7.origami_us!=null){
  s7html+=`<tr><th class="c">path</th><th>rank_configs &micro;s: min</th><th>median</th><th>mean</th><th>max</th><th>E2E sel &micro;s: median</th><th>mean</th></tr>`;
  const e2e=(m,mn)=>`<td>${m!=null?m:"&ndash;"}</td><td>${mn!=null?mn:"&ndash;"}</td>`;
  if(s7.ml_us!=null)s7html+=`<tr><td class="c">ml_rec</td><td>${s7.ml_min}</td><td><b>${s7.ml_us}</b></td><td>${s7.ml_mean_us}</td><td>${s7.ml_max}</td>`+e2e(s7.ml_e2e_us,s7.ml_e2e_mean)+`</tr>`;
  if(s7.origami_us!=null)s7html+=`<tr><td class="c">origami</td><td>${s7.origami_min}</td><td><b>${s7.origami_us}</b></td><td>${s7.origami_mean_us}</td><td>${s7.origami_max}</td>`+e2e(s7.origami_e2e_us,s7.origami_e2e_mean)+`</tr>`;
  if(s7.ml_us!=null&&s7.origami_us!=null){
    // Δ (origami − ml_rec): >0 (green) means ml_rec is faster (saves time)
    const dc=(o,m,bold)=>{ if(o==null||m==null)return`<td>&ndash;</td>`;
      const d=o-m, t=(d>=0?"+":"")+d.toFixed(1);
      return `<td style="color:${d>=0?"#0a0":"#c00"}">${bold?"<b>"+t+"</b>":t}</td>`; };
    s7html+=`<tr style="border-top:2px solid #999"><td class="c"><b>Δ origami&minus;ml_rec</b></td>`
      +dc(s7.origami_min,s7.ml_min)+dc(s7.origami_us,s7.ml_us,true)
      +dc(s7.origami_mean_us,s7.ml_mean_us)+dc(s7.origami_max,s7.ml_max)
      +dc(s7.origami_e2e_us,s7.ml_e2e_us)+dc(s7.origami_e2e_mean,s7.ml_e2e_mean)+`</tr>`;
  }
}
document.getElementById("tbody_s7time").innerHTML=s7html||`<tr><td class="c">rank-time not found</td><td></td></tr>`;
const s7x=[];
if(s7.speedup!=null)s7x.push(["speedup vs origami (mean)",s7.speedup+"x"]);
if(s7.parity_compared!=null)s7x.push(["C++/Python pick parity",s7.parity_match+" / "+s7.parity_compared+" ("+s7.match_rate+"%)"]);
if(s7.smart_k!=null)s7x.push(["smart_k (candidates ranked)",s7.smart_k]);
document.getElementById("tbody_s7meta").innerHTML=`<tr><th class="c">metric</th><th>value</th></tr>`+(s7x.length?s7x.map(([l,v])=>`<tr><td class="c">${l}</td><td>${v}</td></tr>`).join(""):`<tr><td class="c">parity not found</td><td></td></tr>`);

// generalization trajectory (stage04b)
document.getElementById("tbody_traj").innerHTML=TRAJ.map(r=>`<tr><td class="c">${r.round}</td><td>${r.model.toFixed(2)}</td><td>${r.origami!=null?r.origami.toFixed(2):""}</td><td>${r.origami!=null?(r.model-r.origami>=0?"+":"")+(r.model-r.origami).toFixed(2):""}</td><td>${r.n_gemms!=null?r.n_gemms.toLocaleString():""}</td></tr>`).join("")
+ (S.held&&hm!=null?`<tr style="font-weight:600;border-top:2px solid #999"><td class="c">aggregate (best-per-cell, all rounds)</td><td>${hm.toFixed(2)}</td><td>${ho!=null?ho.toFixed(2):""}</td><td>${S.held.delta!=null?(S.held.delta>=0?"+":"")+S.held.delta.toFixed(2):""}</td><td>${S.held.n_cells} cells</td></tr>`:"");

// per-cell summary (final iteration; sortable)
document.getElementById("cellh").textContent="Per-cell summary (final iteration) - "+S.n_cells+" cells | held-out mean "+(hm!=null?hm+"%":"n/a")+" | train-set mean "+(S.train_mean!=null?S.train_mean+"%":"n/a");
// HELD-OUT group (model/origami/Δ on the cell's NEW shapes) and DEPLOYED group
// (model%=pre-smartK full ranking, deployed%=post-smartK, KΔ=deployed-model =
// smart-K's effect, origami in-sample, Δ deployed-origami). Each Δ sits beside
// the two columns it subtracts.
const COLS=[["cell","cell"],["n_gemms","n_gemms"],["held","held-out %"],["held_ori","origami % (held-out)"],["held_delta","\u0394 held-out"],["model","model %"],["deployed","deployed %"],["k_delta","K\u0394"],["origami","origami %"],["dep_delta","\u0394 (dep\u2212ori)"],["smartK","K"],["cap","ed/hd/ih"]];
const STRCOL=k=>(k==="cell"||k==="cap");
let sk="held",asc=true;
function rc(c){if(c.held==null)return"";if(c.held>=95)return"win";if(c.held_ori!=null&&c.held<c.held_ori-2)return"lose";if(c.held<88)return"weak";return"";}
function fmt(c,k){if(k==="cell")return c.cell;if(k==="cap")return c[k]||"";if(k==="n_gemms")return (c[k]||0).toLocaleString();if(k==="smartK")return c[k]!=null?c[k]:"";if(k==="held")return c[k]!=null?(c[k].toFixed(2)+(c.held_inherited?"*":"")):"";if(k==="delta"||k==="held_delta"||k==="dep_delta"||k==="k_delta")return c[k]!=null?((c[k]>=0?"+":"")+c[k].toFixed(2)):"";return c[k]!=null?c[k].toFixed(2):"";}
function rend(){CELLS.sort((a,b)=>{const d=STRCOL(sk)?String(a[sk]||"").localeCompare(String(b[sk]||"")):((a[sk]??-1e9)-(b[sk]??-1e9));return asc?d:-d;});
document.getElementById("chead").innerHTML='<th>#</th>'+COLS.map(([k,t])=>`<th class="sort ${STRCOL(k)?'c':''}" onclick="ss('${k}')">${t}${k===sk?(asc?" \u25b2":" \u25bc"):""}</th>`).join("");
document.getElementById("tbody_cells").innerHTML=CELLS.map((c,i)=>`<tr class="${rc(c)}"><td class="c">${i+1}</td>`+COLS.map(([k])=>`<td class="${STRCOL(k)?'c':''}">${fmt(c,k)}</td>`).join("")+`</tr>`).join("");}
function ss(k){if(k===sk)asc=!asc;else{sk=k;asc=STRCOL(k);}rend();}
rend();

// runtime-weighted worst cells -- client-side sortable (same UX as above).
// All us columns are per-cell GEOMEANS over the cell's GEMMs (oracle = winner
// geomean, model/origami = picked-kernel latency geomean), NOT a single GEMM.
const WORST=__WORST__;
const WCOLS=[["cell","cell"],["n_gemms","n_gemms"],["rw_share","runtime wt%"],["held_w_us","held oracle geomean \u00b5s"],["held_pick_us","held model geomean \u00b5s"],["held_ori_pick_us","held origami geomean \u00b5s"],["extra_us","extra \u00b5s vs ori"],["held","held %"],["held_ori","origami % (held)"],["held_delta","\u0394 held"],["cap","ed/hd/ih"]];
const WSTR=k=>(k==="cell"||k==="cap");
let wsk=null,wasc=true;  // null => preserve server "worst-first" order until a header is clicked
function wrc(c){const hd=c.held_delta;return(hd!=null&&hd<0)?"lose":"";}
function wfmt(c,k){const v=c[k];if(k==="cell"||k==="cap")return v!=null?v:"";if(k==="n_gemms")return (v||0).toLocaleString();if(v==null)return "\u2013";if(k==="rw_share"||k==="held"||k==="held_ori")return v.toFixed(1)+"%";if(k==="held_delta")return (v>=0?"+":"")+v.toFixed(1);return v.toFixed(1);}
function wrend(){if(wsk!==null)WORST.sort((a,b)=>{const x=a[wsk],y=b[wsk],d=WSTR(wsk)?String(x||"").localeCompare(String(y||"")):((x==null?-1e9:x)-(y==null?-1e9:y));return wasc?d:-d;});
const wh=document.getElementById("worsth");if(!wh)return;
wh.innerHTML='<th>#</th>'+WCOLS.map(([k,t])=>`<th class="sort ${WSTR(k)?'c':''}" onclick="wss('${k}')">${t}${k===wsk?(wasc?" \u25b2":" \u25bc"):""}</th>`).join("");
document.getElementById("tbody_worst").innerHTML=WORST.map((c,i)=>`<tr class="${wrc(c)}"><td class="c">${i+1}</td>`+WCOLS.map(([k])=>`<td class="${WSTR(k)?'c':''}">${wfmt(c,k)}</td>`).join("")+`</tr>`).join("");}
function wss(k){if(k===wsk)wasc=!wasc;else{wsk=k;wasc=WSTR(k);}wrend();}
wrend();

// stage 8 -- per-cell diverging Δ bars, sortable by Δ or by cell label.
const S8PC=S.s8pc||{};
function s8esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function drawS8(id,ds,mode){
  let rows=(S8PC[ds]||[]).slice();
  if(mode=="db")rows.sort((a,b)=>(b[1]-b[2])-(a[1]-a[2]));
  else if(mode=="la")rows.sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0);
  else rows.sort((a,b)=>(a[1]-a[2])-(b[1]-b[2]));
  const maxd=Math.max(1,...rows.map(r=>Math.abs(r[1]-r[2]))),HALF=116;
  // Size the label column to the LONGEST full cell name so names stay
  // column-aligned and nothing is cut off (monospace ~7.7px/char).
  const labw=Math.min(900,Math.max(220,Math.round(7.7*Math.max(0,...rows.map(r=>String(r[0]).length)))));
  let h="";
  for(const r of rows){
    const lbl=s8esc(r[0]),d=r[1]-r[2],w=Math.abs(d)/maxd*HALF;
    const fill=d>=0?`<div class="s8fill" style="left:120px;width:${w.toFixed(1)}px;background:#2da44e"></div>`
                   :`<div class="s8fill" style="left:${(120-w).toFixed(1)}px;width:${w.toFixed(1)}px;background:#cf222e"></div>`;
    h+=`<div class="s8r"><div class="s8lab" style="flex-basis:${labw}px" title="${lbl}">${lbl}</div><div class="s8bar"><div class="s8mid"></div>${fill}</div><div class="s8v">${r[1].toFixed(1)} / ${r[2].toFixed(1)} <b style="color:${d>=0?'#2da44e':'#cf222e'}">${d>=0?"+":""}${d.toFixed(1)}</b></div></div>`;
  }
  document.getElementById(id).innerHTML=h||'<div class="mut">no cells</div>';
}
(function(){
  const dss=Object.keys(S8PC).sort();let html="";
  for(const ds of dss){
    const rows=S8PC[ds];if(!rows||!rows.length)continue;
    const id="s8_"+ds.replace(/[^a-zA-Z0-9]/g,"_"),wins=rows.filter(r=>r[1]>=r[2]).length;
    html+=`<figure class="chart" style="max-width:1700px"><figcaption><b>${s8esc(ds)}</b> &mdash; ${rows.length} cells; <span style="color:#2da44e">green</span>=model wins (${wins}/${rows.length}), <span style="color:#cf222e">red</span>=Origami wins. sort <select onchange="drawS8('${id}','${ds}',this.value)"><option value="dw">&Delta; worst&rarr;best</option><option value="db">&Delta; best&rarr;worst</option><option value="la">label A&rarr;Z</option></select></figcaption><div class="s8bars" id="${id}"></div></figure>`;
  }
  const host=document.getElementById("s8charts");
  if(host){host.innerHTML=html||'<p class="mut">no per-cell stage08 data for this run.</p>';
    for(const ds of dss){const rows=S8PC[ds];if(rows&&rows.length)drawS8("s8_"+ds.replace(/[^a-zA-Z0-9]/g,"_"),ds,"dw");}}
})();
</script></body></html>"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: make_report.py <run_dir> [out.html]", file=sys.stderr)
        sys.exit(2)
    rd = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rd, "report.html")
    print("wrote", build(rd, outp))
