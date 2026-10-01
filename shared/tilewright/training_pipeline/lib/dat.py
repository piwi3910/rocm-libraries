# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt Tensile-library .dat file reader and per-kernel info.

This is the single source of truth in pipeline_v3 for:

  * `DAT_DTYPE_PATTERNS` + `DAT_LAYOUT_MAP`  -- .dat filename decoding
  * `parse_dtype_layout` / `parse_scale_mode` -- the two halves of that
                                                 decode (5-tuple key, and
                                                 the hipblaslt scaleA mode
                                                 that disambiguates the
                                                 MX vs non-MX F8 libraries)
  * `KERNEL_SIG_FIELDS`                       -- structural signature
                                                 (kept minimal per design:
                                                  MT + depthU + MI +
                                                  nonTemporalA/B only)
  * `load_dat_solutions`                      -- walk all .dat files
  * `kernel_dat_info`                         -- per-solution recovery keys
                                                 (sol_idx_global/_local/name)
                                                 + base config params.

Every .dat solution dict carries:
    sol["name"]                full kernel name (Tensile codegen tag)
    sol["index"]               TOP-LEVEL == sol_idx_global
                               (the hipblaslt-bench MatchTable key)
    sol["libraryLogicIndex"]   == sol_idx_local within source yaml
    sol["sizeMapping"]         macroTile, matrixInstruction, depthU,
                               nonTemporalA, nonTemporalB, grvwA,
                               grvwB, gwvwD, CUOccupancy,
                               customMainLoopScheduling,
                               ...

We expose the .dat's top-level `.index` field as the recovery key:
when `origami.rank_configs` returns a permuted list, we stash that
position on `config_t.index` (now exposed in the python binding) so we
can recover `sol_idx_global` AND `sol_idx_local` AND LDS-pad
fields (parsed out of `name`) directly from a single .dat lookup --
without re-walking yamls.

Note: `workspace_size` is NOT carried through. hipblaslt's
PredictionLibrary.hpp hardcodes both `workspace_size` and
`workspace_size_per_elem_c` to UINT64_MAX at runtime, so these are
never variable kernel features -- they would just be constant zeros (or
constant max) for every row in the dataset. We omit them entirely.
"""
from __future__ import annotations

import os
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack


def is_tensile_contraction_logic(fname: str) -> bool:
    """True for a Tensile contraction logic file -- plain `.dat` (msgpack) OR
    compressed `.dat.zlib` (the productized hipBLASLt build deflates them)."""
    return "Contraction" in fname and (
        fname.endswith(".dat") or fname.endswith(".dat.zlib")
    )


def read_tensile_logic(path) -> Optional[Dict[str, Any]]:
    """Read one Tensile library logic file and return the unpacked msgpack
    dict (or None on error). Transparently handles both plain `.dat` and
    zlib-compressed `.dat.zlib` (the format the productized build emits)."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        if p.name.endswith(".zlib"):
            raw = zlib.decompress(raw)
        return msgpack.unpackb(raw, raw=False)
    except Exception:
        return None


# ── single source of truth: filename -> dtype/layout decoding ────────────────

# Each row is (filename_pattern, a_type, d_type, compute_type, scale_mode).
#
# `scale_mode` is the hipblaslt-bench `scaleA`/`scaleB` INT the library
# expects (clients/common/include/hipblaslt_scaling_format.hpp):
#     0 = none                 (no per-tensor scale)
#     1 = Scalar               (per-tensor scalar; UseScaleAB: Scalar)
#     2 = Vector
#     3 = Block_32_UE8M0       (MX: block-32 E8M0 scales -- MXFP8)
#     4 = Block_16_UE8M0, 5 = Block_32_UE4M3, ...
#
# WHY the mode has to be part of the row and not just documentation:
# on gfx1250 there are TWO F8 TN libraries that decode to the SAME
# (a_type, d_type, compute_type, transA, transB) 5-tuple --
#
#   ..._F8F8_BF8_HA_Bias_SAB_SAV_UA_...            non-MX, scalar scales
#   ..._F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_ MX, E8M0 block-32 scales
#
# Their kernels are NOT interchangeable (different MI shapes, different
# k-per-instruction, different scale plumbing), so bucketing them under
# one key would silently mix two kernel pools and hand Origami / the
# model candidates the runtime can never select. `parse_scale_mode`
# plus `load_dat_solutions(..., scale_mode=N)` keeps the pools disjoint.
#
# ORDER MATTERS: the first pattern that substring-matches wins, so any
# MX pattern must precede a shorter pattern that could also match its
# filename. (Today the two F8 patterns are mutually exclusive -- the MX
# stem carries `_HA_MXAE8B32_` where the non-MX one carries `_HA_Bias_`
# -- but the MX row stays first so adding a looser F8 row later cannot
# start swallowing MX libraries. tests assert both directions.)
DAT_DTYPE_PATTERNS: Tuple[Tuple[str, str, str, str, int], ...] = (
    # (filename_pattern, a_type, d_type, compute_type, scale_mode)
    ("_BB_BB_", "bf16_r", "bf16_r", "c_f32_r", 0),
    ("_HH_HH_", "f16_r", "f16_r", "c_f32_r", 0),
    ("_SS_SS_HA_Bias_SAV_MX_", "f32_r", "f32_r", "c_xf32_r", 0),
    ("_SS_SS_HA_Bias_SAV_UA_", "f32_r", "f32_r", "c_f32_r", 0),
    ("_SS_SB_", "f32_r", "bf16_r", "c_f32_r", 0),
    # MXF8BS: fp8 (OCP e4m3) inputs with E8M0 block-32 scales on BOTH A
    # and B -- `MXAE8B32` / `MXBE8B32` in the stem is exactly
    # DataTypeMXSA/B = E8 (rocisa DataType 22) with MXBlockA/B = 32.
    # Selected at bench time by `scaleA: 3 / scaleB: 3`
    # (Block_32_UE8M0). Listed BEFORE the non-MX F8 row on purpose.
    ("_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_", "f8_r", "bf16_r", "c_f32_r", 3),
    # F8BS: f8 inputs (A, B) -> bf16 output (D) -> f32 compute. The
    # `_HA_Bias_SAB_SAV_UA_` flavor is the production codegen used by
    # hipBLASLt for f8 GEMMs at deploy time (the SAB suffix means
    # per-tensor scales on both A and B, which is what `scaleA: 1 /
    # scaleB: 1` activates at bench time -- see lib.bench_yaml).
    # The alternate `SABV` variant is a microbenchmarking flavor we
    # leave out so the configs_by_key list stays unambiguous.
    ("_F8F8_BF8_HA_Bias_SAB_SAV_UA_", "f8_r", "bf16_r", "c_f32_r", 1),
)

# hipblaslt-bench scale-mode int -> MX block size (elements per scale).
# Origami's `problem_t` carries `a_mx_block_size` / `b_mx_block_size`;
# a non-zero value is what makes the analytical model rank the problem
# as MX (extra scale traffic + the MX MI variants) instead of plain F8.
# Modes that are not block formats map to 0 == "not MX".
SCALE_MODE_MX_BLOCK: Dict[int, int] = {
    0: 0,  # none
    1: 0,  # Scalar
    2: 0,  # Vector
    3: 32,  # Block_32_UE8M0
    4: 16,  # Block_16_UE8M0
    5: 32,  # Block_32_UE4M3
    # 1001 = Block_32_UE8M0_32_8_EXT: still an E8M0 scale over blocks of
    # 32, so the block size is 32 like mode 3. Listed explicitly because
    # lib/shapes.py's _MX_SCALE_BLOCK already accounts for its scale bytes;
    # if only one of the two tables knows the mode, stage01 budgets the
    # scale tensors while stage04/05 rank the problem as non-MX.
    1001: 32,
}


def mx_block_size_for_scale_mode(scale_mode: Optional[int]) -> int:
    """hipblaslt scaleA/scaleB mode -> origami `*_mx_block_size`.
    Unknown / None modes are treated as non-MX (0) so a config that
    never set `hipblaslt.scaleA` behaves exactly as it does today."""
    if scale_mode is None:
        return 0
    return SCALE_MODE_MX_BLOCK.get(int(scale_mode), 0)


DAT_LAYOUT_MAP: Dict[str, Tuple[str, str]] = {
    "Ailk_Bljk": ("N", "N"),
    "Ailk_Bjlk": ("N", "T"),
    "Alik_Bljk": ("T", "N"),
    "Alik_Bjlk": ("T", "T"),
}


def _match_dtype_row(fname: str) -> Optional[Tuple[str, str, str, str, int]]:
    """First `DAT_DTYPE_PATTERNS` row whose pattern is a substring of
    `fname`, or None. Single place the "first match wins" rule lives so
    `parse_dtype_layout` and `parse_scale_mode` can never disagree about
    which library a filename belongs to."""
    for row in DAT_DTYPE_PATTERNS:
        if row[0] in fname:
            return row
    return None


def parse_dtype_layout(fname: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Decode a .dat filename into (a_type, d_type, compute_type,
    transA, transB) or None on no-match.

    NOTE: deliberately still a 5-tuple. Many call sites unpack it
    positionally and use it as the `configs_by_key` key (stage04:455,
    stage05:2149/:2805, stage06:184, stage07:625, stage08:412), so the
    scale mode is exposed through the SEPARATE `parse_scale_mode` rather
    than widened into this return value."""
    row = _match_dtype_row(fname)
    if row is None:
        return None
    _, a_type, d_type, compute_type, _scale_mode = row
    m = re.search(r"Contraction_l_(A\w+?_B\w+?)_C", fname)
    if not m:
        return None
    layout = DAT_LAYOUT_MAP.get(m.group(1))
    if not layout:
        return None
    transA, transB = layout
    return (a_type, d_type, compute_type, transA, transB)


def parse_scale_mode(fname: str) -> Optional[int]:
    """The hipblaslt-bench scaleA/scaleB mode int the library in `fname`
    expects (0 none / 1 Scalar / 3 Block_32_UE8M0), or None when the
    filename matches no known dtype pattern.

    This is what separates the two gfx1250 F8 TN libraries, which are
    indistinguishable by `parse_dtype_layout` alone."""
    row = _match_dtype_row(fname)
    return None if row is None else int(row[4])


# ── kernel structural signature ──────────────────────────────────────────────
#
# Minimal-set signature: ONLY the kernel-physics fields that uniquely
# disambiguate a Tensile codegen variant. Per design we drop
# grvw / gwvw / occupancy / workGroupMapping from the signature because:
#
#   * grvw_a, grvw_b, gwvw_d -- vector widths are downstream of MT+MI;
#     two kernels with the same MT+MI+NT but different vector widths
#     compute the same product (the model still sees them as features
#     in the enriched dataset, just not in the signature).
#   * CUOccupancy            -- predicted, not codegen-fixed.
#   * workGroupMapping       -- dropped entirely from pipeline_v3
#                                (it's a hipblaslt runtime knob, not a
#                                kernel-codegen knob).

KERNEL_SIG_FIELDS = (
    "mt_m",
    "mt_n",
    "depthU",
    "mi_m",
    "mi_n",
    "mi_k",
    "cache_hints_a",
    "cache_hints_b",
)


def sig_from_sm(sm: Dict[str, Any]) -> Tuple[int, ...]:
    """8-tuple: (MT0, MT1, depthU, MI0, MI1, MI2, nonTemporalA,
    nonTemporalB) -- the minimal kernel signature.

    cache hints are the RAW 0..7 NonTemporal value, matching what the shipped
    models were trained on. See TH-DISABLED(24145512bf) in tilewright model.cpp.
    """
    mt = sm.get("macroTile") or [0, 0, 0]
    mi = sm.get("matrixInstruction") or [0, 0, 0, 0]
    return (
        int(mt[0]),
        int(mt[1]),
        int(sm.get("depthU", 0) or 0),
        int(mi[0]),
        int(mi[1]),
        int(mi[2]),
        int(sm.get("nonTemporalA", 0) or 0),
        int(sm.get("nonTemporalB", 0) or 0),
    )


def sig_from_config_t(c: Any) -> Tuple[int, ...]:
    """Same 8-tuple, but from an origami `config_t`."""
    return (
        int(c.mt.m),
        int(c.mt.n),
        int(c.mt.k),  # origami's config_t exposes depthU as mt.k
        int(c.mi.m),
        int(c.mi.n),
        int(c.mi.k),
        int(c.cache_hints_a),
        int(c.cache_hints_b),
    )


def sig_from_row(row: Dict[str, Any]) -> Tuple[int, ...]:
    """Same 8-tuple, but from an enriched chunk_*.csv row."""

    def _i(k, d=0):
        v = row.get(k, d)
        try:
            return int(v) if v != "" else d
        except (TypeError, ValueError):
            return d

    return (
        _i("mt_m"),
        _i("mt_n"),
        _i("mt_k"),
        _i("mi_m"),
        _i("mi_n"),
        _i("mi_k"),
        _i("cache_hints_a"),
        _i("cache_hints_b"),
    )


# ── full .dat solution walker ────────────────────────────────────────────────


def load_dat_solutions(
    dat_dir: Path,
    *,
    scale_mode: Optional[int] = None,
) -> Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]]:
    """Walk every `TensileLibrary*Contraction*.dat` under `dat_dir`
    and return a dict keyed by (a_type, d_type, compute_type, transA,
    transB) mapping to the full solutions list. Each list entry is
    the RAW msgpack solution dict (with `name`, `index`,
    `libraryLogicIndex`, `sizeMapping`, ...).

    `scale_mode` (keyword-only) is the run's hipblaslt `scaleA` int.
    When None -- the default, and what every pre-MX caller gets -- no
    filtering happens and behavior is byte-identical to before. When an
    int, any .dat whose `parse_scale_mode` differs is SKIPPED, which is
    the only thing keeping the gfx1250 MX and non-MX F8 pools from
    merging under one identical 5-tuple key."""
    out: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    n_skipped = 0
    for fname in sorted(os.listdir(dat_dir)):
        if not is_tensile_contraction_logic(fname):
            continue
        key = parse_dtype_layout(fname)
        if key is None:
            continue
        # Scale-mode filter BEFORE the read: a skipped library should
        # cost a filename compare, not an inflate + msgpack unpack.
        if scale_mode is not None and parse_scale_mode(fname) != int(scale_mode):
            n_skipped += 1
            continue
        data = read_tensile_logic(Path(dat_dir) / fname)
        if data is None:
            continue
        out.setdefault(key, []).extend(data.get("solutions", []) or [])
    if scale_mode is not None and n_skipped:
        # Deliberately printed, not silent: a wrong --scale-mode is the
        # difference between "the MX library" and "an empty pool", and
        # the count is the fastest way for the operator to see which.
        print(
            f"[dat] scale_mode={int(scale_mode)} filter skipped "
            f"{n_skipped} .dat file(s)"
        )
    return out


# ── per-solution enrichment payload (the dat_info row stage04 uses) ──────────


def kernel_dat_info(sol: Dict[str, Any]) -> Dict[str, Any]:
    """For one .dat solution dict, extract the recovery keys + base config:

      sol_idx_global, sol_idx_local, kernel_name,
      mt_m, mt_n, mt_k, mi_m, mi_n, mi_k,
      occupancy, cache_hints_a, cache_hints_b,
      grvw_a, grvw_b, gwvw_d

    stage04's skip path consumes only the indices/name from here (it re-reads
    the kernel params from the source YAML). The former config_ml_t block
    (cache_hints_c/d/e, prefetch / LDS-RVW / LSU) and the LDS-pad fields were
    removed -- Origami's analytical scorer never read them and the ML feature
    group consuming them was dropped (see lib/features.py)."""
    sm = sol.get("sizeMapping", {}) or {}
    name = sol.get("name") or sol.get("kernelName") or ""
    mt = sm.get("macroTile") or [0, 0, 0]
    mi = sm.get("matrixInstruction") or [0, 0, 0, 0]
    return {
        "sol_idx_global": sol.get("index"),
        "sol_idx_local": sol.get("libraryLogicIndex"),
        "kernel_name": name,
        "mt_m": int(mt[0]),
        "mt_n": int(mt[1]),
        "mt_k": int(sm.get("depthU", 0) or 0),
        "mi_m": int(mi[0]),
        "mi_n": int(mi[1]),
        "mi_k": int(mi[2]),
        "occupancy": max(int(sm.get("CUOccupancy", 1) or 1), 1),
        "cache_hints_a": int(sm.get("nonTemporalA", 0) or 0),
        "cache_hints_b": int(sm.get("nonTemporalB", 0) or 0),
        "grvw_a": int(sm.get("grvwA", 1) or 1),
        "grvw_b": int(sm.get("grvwB", 1) or 1),
        "gwvw_d": int(sm.get("gwvwD", 1) or 1),
    }
