# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adaptive cell-splitting for pipeline_v3 active learning.

INVARIANT: every LEAF of the cumulative split tree is one trained
model -- 1 cell, 1 model. The 96 base grid cells (lib/grid.py) form
the root of the tree; recursive binary splits add children.

Sub-cell labels are derived from the original 96-grid label by
appending one `#<axis><op><threshold>` segment per split. Recursive
splits stack and a label carries its full ancestry:

    Large|Large|LargeK|Bany                          base (root leaf)
    Large|Large|LargeK|Bany#M<=4096                  split once on M
    Large|Large|LargeK|Bany#M>4096                   ...  (sibling)
    Large|Large|LargeK|Bany#M>4096#K<=2048           split-of-a-split
    Large|Large|LargeK|Bany#M>4096#K>2048

A `splits.json` lives at `<round_R>/stage04b/splits.json` and records
the splits decided IN that round. The cumulative split tree is the
union of every prior round's splits.json (`load_cumulative_split_tree`).
Stage01 (shape generation), stage05 (training), and stage04b (decision)
all walk this tree to find every shape / GEMM's LEAF label.

SPLIT-DECISION ALGORITHM (`choose_split_for_cell`):
  1. The caller (stage04b) supplies per-GEMM log_ratios = log(model_pick_us
     / winner_us). 0 = perfect; larger = worse.
  2. For each candidate axis in {M, N, K}:
       a) threshold = log-median of that axis's values across the cell's
          GEMMs (50/50 sample-count split; outlier-resistant -- better
          than the legacy geomean-of-(min,max) cut which a single outlier
          could push to the edge of the tier).
       b) Partition GEMMs into lo / hi at the threshold and compute the
          CART variance-reduction gain
              gain(axis) = parent_var - (w_lo * lo_var + w_hi * hi_var)
          over the log_ratios. This measures how much the within-group
          prediction-error spread shrinks when we split on this axis.
       c) Reject axes whose log-range is below the depth-aware floor
          (`min_log_range_for_depth(depth)`) or which would produce a
          child with fewer than `min_split_count` GEMMs.
  3. Pick the axis with the largest gain. Ties go to the axis that
     does NOT already appear in the cell's split path (= encourages
     isotropic refinement; avoids silent M-bias).
  4. Hard depth cap: a cell at depth >= `max_split_depth` is unsplit-
     table regardless of gain.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Axes that the active-learning loop may split on. M, N AND K.
#
# K was historically excluded on the assumption that LargeK was
# effectively capped at ~16384 ("always finite -- more training data
# fixes it"). That assumption no longer holds: lib/grid.py now samples
# LargeK across [513, 131072] (~8 octaves) to cover real workloads,
# and extrapolating across that K range is a major out-of-distribution
# failure mode on real-world shapes. A single un-carvable 8-octave K
# block per cell is exactly what we don't want -- so K is now
# splittable by magnitude, same as M/N. The deployed C++ router
# (ml_recommender.cpp assign_leaf_label) already handles axis=='K'
# generically, so this stays parity-safe (verified by stage07).
#
# B is binary (Bnone vs Bany) and never splits.
SPLITTABLE_AXES: Tuple[str, ...] = ("M", "N", "K")


@dataclass(frozen=True)
class SplitRule:
    """One node in the cumulative split tree.

    `cell` is the PARENT cell label being split (a leaf of the prior
    tree). The split places shapes whose `axis` value satisfies
    `axis <= threshold` into `lo_label`, the rest into `hi_label`.
    """

    cell: str
    axis: str  # "M" / "N" / "K"
    threshold: int  # inclusive on the lo side
    lo_label: str
    hi_label: str
    reason: str = ""  # informational
    prior_sel_eff: Optional[float] = None
    this_sel_eff: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "cell": self.cell,
            "axis": self.axis,
            "threshold": int(self.threshold),
            "lo_label": self.lo_label,
            "hi_label": self.hi_label,
            "reason": self.reason,
            "prior_sel_eff": self.prior_sel_eff,
            "this_sel_eff": self.this_sel_eff,
        }

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "SplitRule":
        return cls(
            cell=str(d["cell"]),
            axis=str(d["axis"]),
            threshold=int(d["threshold"]),  # type: ignore[arg-type]
            lo_label=str(d["lo_label"]),
            hi_label=str(d["hi_label"]),
            reason=str(d.get("reason") or ""),
            prior_sel_eff=(
                float(d["prior_sel_eff"])  # type: ignore[arg-type]
                if d.get("prior_sel_eff") is not None
                else None
            ),
            this_sel_eff=(
                float(d["this_sel_eff"])  # type: ignore[arg-type]
                if d.get("this_sel_eff") is not None
                else None
            ),
        )


# ── label helpers ────────────────────────────────────────────────────────────


def make_subcell_label(parent: str, axis: str, op: str, threshold: int) -> str:
    """Append one split segment. `op` is '<=' or '>'. The threshold is
    integer so labels stay readable in JSON / filenames."""
    return f"{parent}#{axis}{op}{int(threshold)}"


def parent_of(label: str) -> Optional[str]:
    """Return the immediate parent (one segment shorter) or None for a
    base 96-cell label."""
    i = label.rfind("#")
    return label[:i] if i > 0 else None


def is_subcell(label: str) -> bool:
    return "#" in label


def split_depth(label: str) -> int:
    return label.count("#")


def base_cell(label: str) -> str:
    """Return the base 96-cell label this sub-cell descends from."""
    i = label.find("#")
    return label[:i] if i > 0 else label


# ── splits.json IO ───────────────────────────────────────────────────────────

SCHEMA_VERSION = "pipeline-v3-splits-2026-05-27"


def write_splits_json(path: Path, round_idx: int, rules: Iterable[SplitRule]) -> None:
    """Stage04b writes its splits.json via this helper."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": int(round_idx),
        "splits": {r.cell: r.to_json() for r in rules},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _load_one_splits_file(path: Path) -> Dict[str, SplitRule]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    out: Dict[str, SplitRule] = {}
    for cell, rule_d in (data.get("splits") or {}).items():
        out[cell] = SplitRule.from_json(rule_d)
    return out


def load_cumulative_split_tree(
    prior_round_dirs: Iterable[Path],
) -> Dict[str, SplitRule]:
    """Walk every prior ROUND dir and merge every `stage04b/splits.json`
    into one tree. The dict maps a cell label (parent being split) to
    the rule that splits it. Use `assign_subcell` to drill down to the
    leaf.

    NOTE: this function ONLY reads strictly-prior rounds. If the caller
    is stage05 in the CURRENT round, it must ALSO merge in this round's
    just-written stage04b/splits.json so GEMMs in this round's
    stage04 chunks get routed to the freshly-decided child labels --
    otherwise the new children appear empty and `to_train=0`. See
    stage05's `--current-round-dir` arg."""
    tree: Dict[str, SplitRule] = {}
    for rd in prior_round_dirs:
        p = Path(rd) / "stage04b" / "splits.json"
        for cell, rule in _load_one_splits_file(p).items():
            # Later rounds override earlier ones (shouldn't happen in
            # practice -- a cell can only be split once -- but be
            # explicit about precedence).
            tree[cell] = rule
    return tree


# ── leaf-cell assignment ────────────────────────────────────────────────────


def _value_for_axis(axis: str, m: int, n: int, k: int, batch: int) -> int:
    if axis == "M":
        return int(m)
    if axis == "N":
        return int(n)
    if axis == "K":
        return int(k)
    if axis == "B":
        return int(batch)
    raise ValueError(f"unknown split axis: {axis}")


def assign_subcell(
    base_label: str, m: int, n: int, k: int, batch: int, tree: Dict[str, SplitRule]
) -> str:
    """Walk the cumulative split tree from the base 96-cell label down
    to the leaf sub-cell this shape belongs to."""
    label = base_label
    seen: set = set()  # cycle guard (defensive; tree should be acyclic)
    while label in tree and label not in seen:
        seen.add(label)
        rule = tree[label]
        v = _value_for_axis(rule.axis, m, n, k, batch)
        label = rule.lo_label if v <= rule.threshold else rule.hi_label
    return label


# ── split-decision heuristic ────────────────────────────────────────────────


def min_log_range_for_depth(depth: int, base: float = 1.0) -> float:
    """The required log-range an axis must span before we'll consider
    splitting it. We taper this with depth so deep sub-cells -- which
    are by construction narrow -- aren't permanently stuck just because
    they don't span a full octave anymore.

        depth = 0  -> 1.0  octaves   (original 96-grid leaf: easy)
        depth = 1  -> 0.7  octaves
        depth = 2  -> 0.5  octaves
        depth >=3  -> 0.35 octaves
    """
    return max(0.25, base * (0.7**depth))


def _log_median(values: List[int]) -> int:
    """Median in log-space, rounded to nearest int. Robust to outliers
    -- a single shape at the edge of the tier no longer drags the
    threshold all the way there (the legacy `sqrt(min*max)` cut did)."""
    if not values:
        return 0
    log_vals = sorted(math.log(max(int(v), 1)) for v in values)
    n = len(log_vals)
    if n % 2 == 1:
        med = log_vals[n // 2]
    else:
        med = (log_vals[n // 2 - 1] + log_vals[n // 2]) / 2.0
    return max(1, int(round(math.exp(med))))


def _path_axes_of(cell_label: str) -> set:
    """The set of axes already appearing in `cell_label`'s split path
    (the `#<axis>...` segments). Used as a tie-breaker so an
    already-split axis loses to a not-yet-split one when CART gain is
    similar."""
    out: set = set()
    for seg in cell_label.split("#")[1:]:
        for ax in SPLITTABLE_AXES:
            if seg.startswith(ax):
                out.add(ax)
                break
    return out


def _large_axes_in(cell_label: str) -> List[str]:
    """Return the axes whose tier in `cell_label`'s base cell is the
    widest (unbounded-above) tier on that axis: 'Large' for M / N,
    'LargeK' for K. Only these axes span a multi-octave range worth
    carving -- the bounded tiers (Tiny / Small / Mid for M/N, Tiny/Mid
    for K) cover a finite range improved by more data, not by carving.

    K is now eligible (it was historically excluded when LargeK was
    capped at ~16384; with the widened [513, 131072] LargeK range and
    K-extrapolation being a major out-of-distribution failure mode, the
    8-octave K block must be carvable too).

    Sub-cells inherit their base 96-grid label's tier -- so
    `Large|Mid|MidK|Bnone#M<=2048` STILL has M tier == 'Large' and
    is still a valid M-split candidate (further refining the upper
    M range).
    """
    base = base_cell(cell_label)
    parts = base.split("|")
    if len(parts) < 4:
        return []
    m_t, n_t, k_t, _b_t = parts
    out: List[str] = []
    if m_t == "Large":
        out.append("M")
    if n_t == "Large":
        out.append("N")
    if k_t == "LargeK":
        out.append("K")
    return out


def choose_split_for_cell(
    cell_label: str,
    gemms: List[Dict[str, Any]],
    log_ratios: Optional[List[float]] = None,
    *,
    min_split_count: int = 20,
    min_gain: float = 0.0,
    max_split_depth: int = 6,
    tie_eps: float = 1e-3,
) -> Optional[Tuple[str, int]]:
    """Decide how to split `cell_label`. Returns (axis, threshold) or
    None if the cell can't be usefully split.

    Inputs:
      cell_label    base or sub-cell label (`#axis...` segments OK).
      gemms         the cell's GEMM dicts (each has m, n, k, batch_count).
      log_ratios    parallel array of model-prediction log_ratios
                    = log(pick_us / winner_us). If supplied, CART
                    variance-reduction picks the axis; if omitted, we
                    fall back to "widest log-range" (legacy).

    Eligibility (MUST hold or we return None):
      * The base cell has M="Large" OR N="Large" OR K="LargeK". Cells
        in finite tiers (Tiny/Small/Mid M/N, Tiny/Mid K) are not
        splittable -- more training data fixes them, no point carving.
      * Only those widest-tier axes (a subset of {M, N, K}) are
        candidates (see _large_axes_in / SPLITTABLE_AXES).

    Selection order (most -> least important) among eligible axes:
      1. CART variance-reduction gain on per-GEMM log_ratios.
      2. Tie-break: prefer an axis NOT yet in the split path; then
         by axis order (M before N).

    Other guards:
      * threshold per axis = log-median of axis values (50/50 split).
      * depth cap (max_split_depth) and depth-aware min_log_range
        prevent runaway / degenerate splits.
    """
    if not gemms:
        return None
    depth = split_depth(cell_label)
    if depth >= max_split_depth:
        return None
    # ── Eligibility: only Large M / Large N axes are splittable ──
    eligible_axes = _large_axes_in(cell_label)
    if not eligible_axes:
        return None
    have_ratios = log_ratios is not None and len(log_ratios) == len(gemms)
    if have_ratios:
        parent_mean = sum(log_ratios) / len(log_ratios)  # type: ignore[arg-type]
        parent_var = sum(
            (r - parent_mean) ** 2 for r in log_ratios
        ) / len(  # type: ignore[arg-type]
            log_ratios
        )  # type: ignore[arg-type]
    else:
        parent_var = 0.0

    floor = min_log_range_for_depth(depth)
    path_axes = _path_axes_of(cell_label)

    candidates: List[Tuple[float, str, int, float]] = []
    # (gain, axis, threshold, log_range)
    for axis in eligible_axes:
        axis_vals: List[int] = []
        for g in gemms:
            v = _value_for_axis(
                axis, int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"])
            )
            if v > 0:
                axis_vals.append(v)
        if len(axis_vals) < 2:
            continue
        lo, hi = min(axis_vals), max(axis_vals)
        log_range = math.log2(max(hi, 1)) - math.log2(max(lo, 1))
        if log_range < floor:
            continue
        threshold = _log_median(axis_vals)
        # Clamp threshold strictly inside (lo, hi); skip if degenerate.
        if threshold <= lo:
            threshold = lo + 1
        if threshold >= hi:
            threshold = hi - 1
        if threshold <= lo or threshold >= hi:
            continue
        # Partition + minimum count check.
        lo_idx = [i for i, v in enumerate(axis_vals) if v <= threshold]
        hi_idx = [i for i, v in enumerate(axis_vals) if v > threshold]
        if len(lo_idx) < min_split_count or len(hi_idx) < min_split_count:
            continue
        if have_ratios:
            lo_r = [log_ratios[i] for i in lo_idx]  # type: ignore[index]
            hi_r = [log_ratios[i] for i in hi_idx]  # type: ignore[index]
            lo_mean = sum(lo_r) / len(lo_r)
            hi_mean = sum(hi_r) / len(hi_r)
            lo_var = sum((r - lo_mean) ** 2 for r in lo_r) / len(lo_r)
            hi_var = sum((r - hi_mean) ** 2 for r in hi_r) / len(hi_r)
            w_lo = len(lo_r) / len(gemms)
            w_hi = len(hi_r) / len(gemms)
            gain = parent_var - (w_lo * lo_var + w_hi * hi_var)
        else:
            # Legacy fallback: gain proxies to log_range so wider axes
            # win, identical to the old behaviour.
            gain = log_range
        candidates.append((gain, axis, threshold, log_range))

    if not candidates:
        return None

    # Apply min_gain only when we have a CART criterion.
    if have_ratios:
        candidates = [c for c in candidates if c[0] >= min_gain]
        if not candidates:
            return None

    # (Tier-first pruning is no longer needed: `eligible_axes` is
    # already restricted to Large M / Large N at the top of this
    # function, so every candidate is guaranteed to be in the
    # widest tier we allow.)

    # Sort by gain desc, then tie-break: prefer an axis NOT yet in the
    # split path, then by SPLITTABLE_AXES iteration order (M, N, K).
    axis_order = {a: i for i, a in enumerate(SPLITTABLE_AXES)}

    def _key(c: Tuple[float, str, int, float]):
        gain, axis, _t, _lr = c
        axis_unused = 0 if axis in path_axes else 1
        # Negate gain to sort largest-first under ascending sort;
        # axis_unused negated similarly (1 wins).
        return (-gain, -axis_unused, axis_order[axis])

    # Group ties within `tie_eps` and apply tie-break.
    candidates.sort(key=_key)
    best_gain = candidates[0][0]
    tied = [c for c in candidates if abs(c[0] - best_gain) <= tie_eps]
    tied.sort(key=lambda c: (0 if c[1] not in path_axes else 1, axis_order[c[1]]))
    _, axis, threshold, _ = tied[0]
    return axis, threshold


def build_split_rule(
    cell_label: str,
    axis: str,
    threshold: int,
    reason: str = "",
    prior_sel_eff: Optional[float] = None,
    this_sel_eff: Optional[float] = None,
) -> SplitRule:
    return SplitRule(
        cell=cell_label,
        axis=axis,
        threshold=int(threshold),
        lo_label=make_subcell_label(cell_label, axis, "<=", threshold),
        hi_label=make_subcell_label(cell_label, axis, ">", threshold),
        reason=reason,
        prior_sel_eff=prior_sel_eff,
        this_sel_eff=this_sel_eff,
    )


# ── leaf enumeration helpers ────────────────────────────────────────────────


def leaves_under(label: str, tree: Dict[str, SplitRule]) -> List[str]:
    """Return every leaf reachable from `label` in the cumulative tree.
    A leaf is a label that is NOT a key in the tree."""
    if label not in tree:
        return [label]
    rule = tree[label]
    return leaves_under(rule.lo_label, tree) + leaves_under(rule.hi_label, tree)


def axis_predicate(label: str, axis: str) -> Optional[Tuple[str, int]]:
    """If `label` carries a constraint on `axis`, return ('<=', t) or
    ('>', t). Otherwise None. Walks every `#…` segment so multi-level
    splits all contribute. The TIGHTEST constraint per direction
    wins (smallest <= threshold and largest > threshold)."""
    le_thr: Optional[int] = None
    gt_thr: Optional[int] = None
    for seg in label.split("#")[1:]:
        if not seg.startswith(axis):
            continue
        rest = seg[len(axis) :]
        if rest.startswith("<="):
            t = int(rest[2:])
            le_thr = t if le_thr is None else min(le_thr, t)
        elif rest.startswith(">"):
            t = int(rest[1:])
            gt_thr = t if gt_thr is None else max(gt_thr, t)
    if le_thr is not None and gt_thr is not None:
        # Both -- the tighter pair determines the range. Caller can
        # consume both via separate calls.
        return ("range", (gt_thr, le_thr))  # type: ignore[return-value]
    if le_thr is not None:
        return ("<=", le_thr)
    if gt_thr is not None:
        return (">", gt_thr)
    return None


def format_split_tree(
    tree: Dict[str, SplitRule], roots: Optional[Iterable[str]] = None
) -> str:
    """Render the cumulative tree as an ASCII outline. Each base cell's
    splits hang off it; sub-cells whose path includes additional splits
    indent further. Used by stage04b / stage05 banners so the operator
    can eyeball the tree at a glance."""
    if not tree:
        return "(empty -- no splits yet)"
    if roots is None:
        # A root is anything that's a split key OR an ancestor of one.
        candidates = sorted(set(base_cell(c) for c in tree))
        roots = candidates
    lines: List[str] = []

    def _rec(label: str, prefix: str, last: bool) -> None:
        if label in tree:
            rule = tree[label]
            tag = (
                f"  -> split on {rule.axis} at {rule.threshold}  "
                f"(reason={rule.reason or '-'})"
            )
        else:
            tag = ""
        connector = "└── " if last else "├── "
        if prefix == "":
            lines.append(f"{label}{tag}")
        else:
            lines.append(f"{prefix}{connector}#{label.rsplit('#', 1)[-1]}" f"{tag}")
        if label in tree:
            rule = tree[label]
            child_prefix = prefix + ("    " if last else "│   ")
            _rec(rule.lo_label, child_prefix, False)
            _rec(rule.hi_label, child_prefix, True)

    for r in roots:
        _rec(r, "", True)
    return "\n".join(lines)


def restrict_axis_bounds(
    label: str, axis: str, base_lo: int, base_hi: int
) -> Tuple[int, int]:
    """Tighten `[base_lo, base_hi]` on `axis` using every split
    constraint stamped into `label`. Returns (lo, hi) inclusive on
    both sides (lo > hi means an empty range)."""
    lo, hi = int(base_lo), int(base_hi)
    for seg in label.split("#")[1:]:
        if not seg.startswith(axis):
            continue
        rest = seg[len(axis) :]
        if rest.startswith("<="):
            t = int(rest[2:])
            hi = min(hi, t)
        elif rest.startswith(">"):
            t = int(rest[1:])
            lo = max(lo, t + 1)
    return lo, hi


def cell_axis_octaves(cell_label: str) -> Dict[str, float]:
    """For each WIDE / splittable axis (Large M/N, LargeK) of
    `cell_label`, return how many octaves (log2 magnitude span) its
    NOMINAL tier bounds cover, tightened by any `#`-splits already
    applied.

    This is the signal behind the *structural* width-split trigger: a
    leaf the deploy runtime routes ANY shape in a 9-10 octave range to
    cannot be served well by a single MLP -- no matter how high it scores
    on the (same-distribution) synthetic held-out set. Carving by
    magnitude is the robust fix, independent of that potentially
    over-optimistic sel_eff signal."""
    from lib import grid as _grid  # local import avoids import cycle

    base = base_cell(cell_label)
    parts = base.split("|")
    if len(parts) < 4:
        return {}
    m_t, n_t, k_t, _b_t = parts
    extents = {
        "M": _grid.TIER_RANGES_MN.get(m_t),
        "N": _grid.TIER_RANGES_MN.get(n_t),
        "K": _grid.TIER_RANGES_K.get(k_t),
    }
    out: Dict[str, float] = {}
    for ax in _large_axes_in(cell_label):
        rng = extents.get(ax)
        if not rng:
            continue
        lo, hi = restrict_axis_bounds(cell_label, ax, int(rng[0]), int(rng[1]))
        if hi > lo > 0:
            out[ax] = math.log2(hi / lo)
    return out
