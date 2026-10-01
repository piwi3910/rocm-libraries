# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""origami python binding loader + helpers.

The origami binding ships out-of-tree (built from
rocm-libraries/shared/origami/python). pipeline_v3 imports it lazily
so the rest of the pipeline runs even without origami (with the
per-cell baseline + skip-row enrichment disabled).

This module also wraps the small amount of "wiring" each stage needs:

    set_python_path(p)              insert into sys.path before import
    import_origami() -> module|None lazy import; None on failure
    make_problem(...)               build problem_t from a GEMM row
                                     (mx_block_size=32 for MXFP8)
    config_from_dat_solution(...)   build config_t from .dat sizeMapping
                                     (also stamps config_t.index = cfg_position
                                      for round-trip recovery)
    rank(problem, hw, configs)      origami.rank_configs() with a try/except
    get_hardware_for_device(idx)    origami.get_hardware_for_device(idx)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# hipblaslt operand-dtype string -> the token `origami.string_to_datatype`
# understands. THE SINGLE COPY: stage04 and stage05 import this dict rather
# than redefining it (they each used to carry a private copy that drifted).
#
# `'f8_r'` is the spelling every config yaml and every enriched CSV row
# actually uses -- the old `'fp8_r'` key never matched anything, so
# `make_problem` silently fell through to the `"bf16"` default and built a
# BF16 problem for f8 GEMMs, corrupting Origami's analytical ranking for the
# whole F8 family. Both keys are present now, and both map to `'f8'`:
# origami's `string_to_datatype("fp8")` RAISES (verified against the
# gfx1250 binding) -- only `"f8"` resolves to `data_type_t.Float8`.
_DTYPE_TO_ORIGAMI = {
    "bf16_r": "bf16",
    "f16_r": "f16",
    "f32_r": "f32",
    "f8_r": "f8",
    "fp8_r": "f8",
}


def pool_has_cache_hints(candidates: Iterable[Any]) -> bool:
    """Whether a candidate pool actually ships NonTemporal/TemporalHint
    variants. Mirror of the `std::any_of` in `tilewright::rank_configs_impl`;
    gfx950's Origami_nta4 / Origami_ntb4 pools return True, gfx1250 False.

    Accepts either enriched-CSV rows (mappings) or origami `config_t` objects.
    """

    def _hint(c: Any, name: str) -> int:
        if isinstance(c, Mapping):
            return int(c.get(name, 0) or 0)
        return int(getattr(c, name, 0) or 0)

    return any(
        _hint(c, "cache_hints_a") or _hint(c, "cache_hints_b") for c in candidates
    )


def set_python_path(path: str) -> None:
    """Insert origami's `python/src` into sys.path so `import origami`
    resolves to the binding we just built. Idempotent."""
    if path and path not in sys.path:
        sys.path.insert(0, path)


def import_origami() -> Optional[Any]:
    """Lazy import. Returns the module or None if origami is missing /
    binding hasn't been built. Callers check for None and degrade
    gracefully (baseline disabled, skip-row enrichment skipped)."""
    try:
        import origami  # type: ignore

        return origami
    except ImportError:
        return None


# ── origami.rank_configs arity shim ──────────────────────────────────────────
#
# Newer origami requires a 4th argument, `model_t`, selecting which cost model
# to rank with:
#     rank_configs(problem_t, hardware_t, Sequence[config_t], model_t)
# Against such an origami, a call in the older 3-argument form raises
# TypeError. The caller's try/except swallows it, so it surfaces only as
# "skip_blank: 100% -- rank beyond origami's ranked count", i.e. NO skip-row
# enrichment at all, which looks like a data property rather than a broken
# call.
#
# Resolve once, tolerate both arities, and always ask for the GEMM model.
def _rank_configs_compat(origami_mod, problem, hw, configs):
    """origami.rank_configs across both the 3-arg and 4-arg signatures."""
    mt = getattr(origami_mod, "model_t", None)
    if mt is not None and hasattr(mt, "gemm"):
        try:
            return origami_mod.rank_configs(problem, hw, configs, mt.gemm)
        except TypeError:
            pass
    return origami_mod.rank_configs(problem, hw, configs)


def make_problem(
    origami,
    m: int,
    n: int,
    k: int,
    batch: int,
    a_type: str,
    b_type: str,
    d_type: str,
    compute_type: str,
    transA: str,
    transB: str,
    mx_block_size: int = 0,
) -> Any:
    """Build an `origami.problem_t` for one GEMM.

    `mx_block_size` is the number of elements each block scale covers
    (32 for MXFP8 == hipblaslt `scaleA: 3` / Block_32_UE8M0; 0 = not an
    MX problem). It is written to BOTH `a_mx_block_size` and
    `b_mx_block_size`, mirroring how the runtime fills the struct
    (hipblaslt rocroller/solution_selection.cpp sets them from
    `scaleTypeA/B.blockRowSize * blockColSize`; the Tensile
    PredictionLibrary.hpp path hardcodes 0 because MX there comes from
    rocroller). Without it, an MXFP8 problem is ranked as if it were
    plain F8 and the analytical model misses the scale traffic.
    Defaults to 0 so every existing caller is unchanged."""
    p = origami.problem_t()
    p.size = origami.dim3_t(int(m), int(n), int(k))
    p.batch = int(batch)
    p.a_transpose = origami.transpose_t.T if transA == "T" else origami.transpose_t.N
    p.b_transpose = origami.transpose_t.T if transB == "T" else origami.transpose_t.N
    a_dt = _DTYPE_TO_ORIGAMI.get(a_type, "bf16")
    d_dt = _DTYPE_TO_ORIGAMI.get(d_type, a_dt)
    p.a_dtype = origami.string_to_datatype(a_dt)
    p.b_dtype = origami.string_to_datatype(_DTYPE_TO_ORIGAMI.get(b_type, a_dt))
    p.c_dtype = origami.string_to_datatype(d_dt)
    p.d_dtype = origami.string_to_datatype(d_dt)
    if "xf32" in (compute_type or "").lower():
        p.mi_dtype = origami.string_to_datatype("xf32")
    else:
        p.mi_dtype = origami.string_to_datatype(a_dt)
    p.a_mx_block_size = int(mx_block_size)
    p.b_mx_block_size = int(mx_block_size)
    return p


def config_from_dat_solution(origami, sm: Dict[str, Any], cfg_position: int) -> Any:
    """Build an `origami.config_t` from a .dat sizeMapping. Stamps
    `c.index = cfg_position` so we can recover the .dat position
    after `rank_configs` permutes the list (this is what powers
    skip-row enrichment in stage04).

    Note: pipeline_v3 drops both workGroupMapping AND workspace_size
    from the feature/signature path:
      * workGroupMapping is a hipblaslt runtime knob, not a kernel-
        codegen knob; we set `c.workgroup_mapping = 0` only because
        origami's predicate evaluation reads the field.
      * workspace_size and workspace_size_per_elem_c are HARDCODED to
        UINT64_MAX in hipblaslt's PredictionLibrary.hpp at runtime, so
        they are never variable per-kernel features. We mirror that
        hardcode here for origami's predicates."""
    c = origami.config_t()
    c.index = cfg_position
    mt = sm.get("macroTile") or [0, 0, 0]
    c.mt = origami.dim3_t(int(mt[0]), int(mt[1]), int(sm.get("depthU", 0) or 0))
    mi = sm.get("matrixInstruction") or [0, 0, 0, 0]
    if not mi or (mi[0] == 0 and mi[1] == 0 and mi[2] == 0):
        c.mi = origami.dim3_t(1, 1, 64)
    else:
        c.mi = origami.dim3_t(int(mi[0]), int(mi[1]), int(mi[2]))
    cms = sm.get("customMainLoopScheduling", 0) or 0
    try:
        c.hand_optimized_main_loop = int(cms) > 0
    except (TypeError, ValueError):
        c.hand_optimized_main_loop = False
    c.occupancy = max(int(sm.get("CUOccupancy", 1) or 1), 1)
    c.cache_hints_a = int(sm.get("nonTemporalA", 0) or 0)
    c.cache_hints_b = int(sm.get("nonTemporalB", 0) or 0)
    # Hardcoded to UINT64_MAX in hipblaslt's PredictionLibrary.hpp at
    # runtime; mirror that here. Both fields are NOT used as features.
    c.workspace_size = (2**64) - 1
    c.workspace_size_per_elem_c = (2**64) - 1
    c.grvw_a = int(sm.get("grvwA", 1) or 1)
    c.grvw_b = int(sm.get("grvwB", 1) or 1)
    c.gwvw_d = int(sm.get("gwvwD", 1) or 1)
    # workGroupMapping is intentionally NOT used by the model
    # (pipeline_v3 design); we still set it on config_t because some
    # origami predicates check it during rank_configs.
    c.workgroup_mapping = 0
    return c


def get_hardware_for_device(origami, device: int) -> Any:
    return origami.get_hardware_for_device(int(device))


def rank(origami, problem: Any, hw: Any, configs: List[Any]) -> List[Any]:
    """Wrapper around `origami.rank_configs` -- returns [] on any
    backend failure rather than propagating the exception."""
    try:
        return list(_rank_configs_compat(origami, problem, hw, configs))
    except Exception:
        return []


# Bits-per-element table for the dtypes the short-circuit logic
# inspects. Mirrors `origami::datatype_to_bits` (gemm.cpp). We keep
# a local table so this module does NOT need a built origami `.so`
# to apply the feasibility filter -- pure Python.
_DTYPE_BITS = {
    "f32_r": 32,
    "f32": 32,
    "float": 32,
    "bf16_r": 16,
    "bf16": 16,
    "bfloat16": 16,
    "f16_r": 16,
    "f16": 16,
    "half": 16,
    "fp8_r": 8,
    "fp8": 8,
    "float8": 8,
    "f8_r": 8,
    "bf8_r": 8,
    "bf8": 8,
    "bfloat8": 8,
    "xf32_r": 32,
    "xf32": 32,
    "xfloat32": 32,
    "c_f32_r": 32,  # compute_type prefix used by enriched rows
}


def _dtype_to_bits(dt: str) -> int:
    key = (dt or "").strip().lower()
    if key in _DTYPE_BITS:
        return _DTYPE_BITS[key]
    # Defensive: strip the `c_` prefix used in enriched CSV
    # compute_type cells.
    if key.startswith("c_") and key[2:] in _DTYPE_BITS:
        return _DTYPE_BITS[key[2:]]
    # Fall back to bf16 (16 bits) -- the data we train on is
    # overwhelmingly bf16 today.
    return 16


def is_kernel_feasible(
    m: int,
    n: int,
    k: int,
    batch: int,
    a_dtype: str,
    b_dtype: str,
    transA: str,
    transB: str,
    mt_m: int,
    mt_n: int,
    mt_k: int,
    mi_m: int,
    mi_n: int,
    mi_k: int,
    cache_hints_a: int,
    cache_hints_b: int,
    *,
    pool_has_hints: bool = True,
) -> bool:
    """Byte-for-byte Python mirror of `origami::is_kernel_feasible`
    (shared/origami/src/origami/gemm.cpp). Kept in pure Python so
    training / parity / stage 8 can apply EXACTLY the same
    short-circuit the deployed C++ runtime applies -- without
    requiring a built origami `.so` and without paying the cost of
    `origami.rank_configs` (which would also compute analytical
    latencies we never use).

    Update this whenever the C++ short-circuit rules change.
    """
    M, N, K, B = int(m), int(n), int(k), int(batch)
    MT_M, MT_N, MT_K = int(mt_m), int(mt_n), int(mt_k)
    MI_M, MI_N, MI_K = int(mi_m), int(mi_n), int(mi_k)
    cha, chb = int(cache_hints_a), int(cache_hints_b)
    a_trans = transA == "T"
    b_trans = transB == "T"
    a_bits = _dtype_to_bits(a_dtype)
    b_bits = _dtype_to_bits(b_dtype)

    # 1) Small problem with batch -- must fit in one tile.
    if M <= 256 and N <= 256 and K < 1024 and B != 1 and (MT_M < M or MT_N < N):
        return False
    # 2) Dot2 is only correct for M < 3.
    if MI_M == 1 and MI_N == 1 and MI_K == 64 and M > 2:
        return False
    # 3) NTA/NTB rules. The per-problem branch picks WHICH NTA/NTB
    #    combination is allowed; everything else is rejected. This
    #    is what makes the survivor set effectively unique per
    #    (MT,MI) for THIS problem -- the dedup is a side effect of
    #    the short-circuit rules, not an explicit step.
    #    The two hint-requiring branches only apply when the pool
    #    actually ships NT/TH variants (`pool_has_hints`); gfx1250
    #    has none, and requiring one there rejects every candidate.
    # TH-DISABLED(24145512bf): the TILEWRIGHT_TH_RANK early-out lived here.
    K_mod_128b = (K * a_bits) % 1024
    MT_K_mod_128b = (MT_K * a_bits) % 1024
    if K_mod_128b == 0 and MT_K_mod_128b == 0:
        if M <= MT_M * 2 and (not b_trans) and ((N * b_bits) // max(M * a_bits, 1) > 5):
            if pool_has_hints and chb != 4:
                return False
        elif N <= MT_N * 2 and a_trans and ((M * a_bits) // max(N * b_bits, 1) > 5):
            if pool_has_hints and cha != 4:
                return False
        else:
            if cha or chb:
                return False
    elif cha or chb:
        return False
    return True


# The structural kernel signature lives in lib/dat.py
# (single source of truth). Re-export here for convenience.
from .dat import sig_from_config_t as config_t_sig  # noqa: E402,F401
from .dat import sig_from_row as bench_row_sig  # noqa: E402,F401
