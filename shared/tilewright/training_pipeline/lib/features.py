# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pipeline_v3 generic feature catalog.

"Generic" because every input (problem_t, config_t, hardware_t) is
backend-agnostic GEMM/kernel metadata, deployable on top of hipBLASLt,
Composable Kernel, Triton, or any other kernel-selection runtime.

ALLOWED INPUTS
  problem_t    : M, N, K, batch, transA, transB, dtypes (a/b/c/d/mi).
  config_t     : MT (m/n/k), MI (m/n/k), occupancy, cache_hints_a/b,
                 grvw_a/b, gwvw_d.   (NO tensile_params_t. NO
                 workGroupMapping. NO workspace_size: hardcoded to
                 UINT64_MAX in hipblaslt's PredictionLibrary.hpp -- no
                 per-kernel signal.)
  hardware_t   : N_CU, LDS, L2, mem_bw_per_wg_coefficients, etc.
                 We query these LIVE from origami's Python binding
                 (`origami.get_hardware_for_device(<device>)`) so the
                 training side and the C++ deployment side can NEVER
                 drift apart. No hardware_constants.json, no per-arch
                 duplication in Python. Training runs on the same
                 host as hipblaslt-bench so a HIP device is always
                 available; catalog-only helpers
                 (`feature_names()`, `feature_count()`,
                 `feature_names_hash()`) don't call `get_hw()` so
                 they still work device-less.

DESIGN PRINCIPLES
  1. NO features that are exact linear combinations of other features.
     Two-tower NN models can recover sums/differences trivially; we
     don't need explicit log2_aspect_mn, log2_tile_area, etc.
  2. NO `_proxy` features. The previous catalog approximated tensile
     codegen physics with hand-tuned constants (pgr_alpha = 0.5 etc.);
     those were rarely better than just feeding the model the inputs
     they were built from.
  3. NO features that are constant over the workload. Leading-dim and
     swizzle features had zero variance on every bf16/TN run.
  4. NO duplicate scalings. `occupancy_norm = occupancy / 9` is fine;
     emitting both is not.

OUTPUT FOR ONE (problem, config) PAIR
  Three vectors (query / item / interaction), in the order
  FEATURE_GROUPS lists them. See `feature_names()` / `split_features()`.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from .arch import canonical_arch


# ── Version ──────────────────────────────────────────────────────────────────

FEATURES_VERSION = "pipeline-v3-2026-06-drop-config-ml"


# ── dtype tables ─────────────────────────────────────────────────────────────

_BPE: Dict[str, float] = {
    "float": 4.0,
    "f32": 4.0,
    "xfloat32": 4.0,
    "xf32": 4.0,
    "double": 8.0,
    "f64": 8.0,
    "complexfloat": 8.0,
    "complexdouble": 16.0,
    "half": 2.0,
    "f16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float8": 1.0,
    "f8": 1.0,
    "float8_fnuz": 1.0,
    "bfloat8": 1.0,
    "bf8": 1.0,
    "bfloat8_fnuz": 1.0,
    "float8bfloat8": 1.0,
    "bfloat8float8": 1.0,
    "float8bfloat8_fnuz": 1.0,
    "bfloat8float8_fnuz": 1.0,
    "float6": 0.75,
    "bfloat6": 0.75,
    "f6": 0.75,
    "bf6": 0.75,
    "float4": 0.5,
    "f4": 0.5,
    "int8": 1.0,
    "int8x4": 4.0,
    "int32": 4.0,
    "int64": 8.0,
    "int4": 0.5,
}

_DTYPE_ALIAS_EXPAND = {
    "bf16": "bfloat16",
    "f16": "half",
    "fp16": "half",
    "bf8": "bfloat8",
    "f8": "float8",
    "fp8": "float8",
    "bf6": "bfloat6",
    "f6": "float6",
    "fp6": "float6",
    "f4": "float4",
    "fp4": "float4",
    "f32": "float",
    "fp32": "float",
    "xf32": "xfloat32",
    "f64": "double",
    "fp64": "double",
}

_DTYPE_ID: Dict[str, int] = {
    "float": 0,
    "f32": 0,
    "xfloat32": 1,
    "xf32": 1,
    "half": 2,
    "f16": 2,
    "bfloat16": 3,
    "bf16": 3,
    "float8": 4,
    "f8": 4,
    "bfloat8": 5,
    "bf8": 5,
    "float6": 6,
    "bfloat6": 7,
    "float4": 8,
    "int8": 9,
    "int32": 10,
    "float8_fnuz": 4,
    "bfloat8_fnuz": 5,
    "float8bfloat8": 11,
    "bfloat8float8": 12,
    "double": 13,
    "int4": 14,
}


def _normalize_dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "bfloat16"
    s = str(dtype)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    s = s.lower()
    return _DTYPE_ALIAS_EXPAND.get(s, s)


def bpe_for_dtype(dtype: Any) -> float:
    return _BPE.get(_normalize_dtype_name(dtype), 2.0)


def _dtype_id(dtype: Any) -> int:
    return _DTYPE_ID.get(_normalize_dtype_name(dtype), 0)


def _transpose_bit(x: Any) -> int:
    s = str(x).upper()
    return 1 if s.endswith("T") or s == "T" else 0


# ── hardware constants -- queried live from origami's Python binding ──────
#
# We do NOT keep a parallel table of hardware constants. Instead we
# ask origami directly, via `get_hardware_for_device(<device>)`,
# which populates `hardware_t` from `hipDeviceProp_t` AND origami's
# baked-in `get_arch_constants()`. That makes origami the SINGLE
# source of truth for every constant feature.py uses (N_CU, L2, LDS,
# compute_clock, mem_bw_per_wg_coefficients, parallel_mi_cu, etc.).
#
# This requires a HIP device. That's not a real constraint for this
# pipeline -- training runs on the SAME host as
# stage02/03/08/09's hipblaslt-bench, which is by definition a GPU
# host. Catalog-only operations (`feature_names()`, `feature_count()`,
# `feature_names_hash()`) don't call `get_hw()` so they still work
# without a device.
#
# `_HW_CACHE` is keyed by `(device, arch)` so a multi-arch training
# run (theoretically possible) can hold both copies; in practice we
# query device 0 and one arch.

_HW_CACHE: Dict[Tuple[int, str], Dict[str, Any]] = {}


def _import_origami() -> Any:
    """Lazy import of the origami binding. We use lib.origami's
    helper rather than `import origami` directly so the error message
    is consistent across the pipeline."""
    try:
        from . import origami as _ml_ori  # type: ignore
    except ImportError:  # not installed as a pkg
        try:
            import lib.origami as _ml_ori  # type: ignore
        except ImportError:
            _ml_ori = None  # type: ignore
    if _ml_ori is None:
        raise RuntimeError(
            "lib.origami not importable -- ensure " "ml_recommender/ is on sys.path"
        )
    og = _ml_ori.import_origami()
    if og is None:
        raise RuntimeError(
            "origami python binding not importable. Either:\n"
            "  (a) pass --origami-python-path "
            "<repo>/shared/origami/python/src to the stage CLI, or\n"
            "  (b) prepend that path to PYTHONPATH, or\n"
            "  (c) `cp <repo>/shared/origami/build/python/src/"
            "origami/origami.cpython-*.so` into your venv's "
            "site-packages/origami/.\n"
            "Stage07's --build-after-apply does (c) automatically."
        )
    return og


def _hardware_t_to_dict(h: Any) -> Dict[str, Any]:
    """Project the origami `hardware_t` object onto the dict shape
    features.py uses (keeps the rest of features.py oblivious to
    origami's exact field names)."""
    return {
        "N_CU": int(h.N_CU),
        "lds_capacity_bytes": int(h.lds_capacity),
        "L2_capacity_bytes": int(h.L2_capacity),
        "mem_bw_per_wg_coefficients": list(h.mem_bw_per_wg_coefficients),
        "parallel_mi_cu": int(h.parallel_mi_cu),
        "CU_per_L2": int(h.CU_per_L2),
        "compute_clock_ghz": float(h.compute_clock_ghz),
        "NUM_XCD": int(h.NUM_XCD),
    }


def get_hw(arch: str = "gfx950", device: int = 0) -> Dict[str, Any]:
    """Return the hardware constants needed by feature computation,
    sourced live from origami via `get_hardware_for_device(device)`.
    Cached per (device, arch). Raises a clear error if origami or the
    device is unavailable -- features.py NEVER duplicates per-arch
    sizing constants in Python."""
    key = (int(device), arch)
    cached = _HW_CACHE.get(key)
    if cached is not None:
        return cached
    og = _import_origami()
    try:
        h = og.get_hardware_for_device(int(device))
    except Exception as e:
        raise RuntimeError(
            f"origami.get_hardware_for_device({device}) failed: {e}\n"
            f"Training is expected to run on a GPU host (the same one "
            f"that runs hipblaslt-bench). If you need a device-less "
            f"hardware probe, add a `get_default_hardware_for_arch` "
            f"binding to origami and call it here -- features.py must "
            f"never duplicate per-arch sizing constants."
        ) from e
    hw = _hardware_t_to_dict(h)
    _HW_CACHE[key] = hw
    return hw


# ── MI-instruction latency table (subset of origami::INSTRUCTION_MAP) ────────

_MI_LATENCY_TABLE: Dict[str, Dict[Tuple[int, int, int, str], int]] = {
    "gfx950": {
        (32, 32, 2, "float"): 64,
        (32, 32, 1, "float"): 64,
        (16, 16, 4, "float"): 32,
        (16, 16, 1, "float"): 32,
        (32, 32, 16, "bfloat16"): 32,
        (16, 16, 32, "bfloat16"): 16,
        (32, 32, 8, "half"): 32,
        (32, 32, 16, "half"): 32,
        (16, 16, 16, "half"): 16,
        (16, 16, 32, "half"): 16,
        (32, 32, 64, "float8"): 32,
        (16, 16, 128, "float8"): 16,
        (32, 32, 64, "bfloat8"): 32,
        (16, 16, 128, "bfloat8"): 16,
        # xf32 (TF32) on gfx950 is EMULATED as 3x BF16 MFMA passes, so
        # each MI costs 3x its bf16 counterpart (matches origami
        # hardware.hpp INSTRUCTION_MAP). The TF32_NN library kernels use
        # MI 16x16x32; the 32x32x8 form is the wide variant.
        (32, 32, 8, "xfloat32"): 96,  # v_mfma_f32_32x32x8_bf16 * 3
        (16, 16, 32, "xfloat32"): 48,  # v_mfma_f32_16x16x16_bf16 * 3
        (16, 16, 16, "xfloat32"): 48,
    },
    # gfx1250 is a WMMA part, NOT MFMA: the matrix instructions are
    # v_wmma_* with a 16x16 output tile only (no 32x32 form except the
    # 32x16x128 fp4 variant), which is why every key below is 16x16x*.
    # Its fp8 types are OCP e4m3/e5m2 ("float8"/"bfloat8" in rocisa's
    # DataType enum, ids 15/16) and NOT the _fnuz variants used on
    # gfx942/gfx950 -- _normalize_dtype_name() maps both spellings onto
    # the same "float8"/"bfloat8" strings, so the keys read the same but
    # the *cycle counts* differ and must not be shared with gfx950.
    #
    # Values are a verbatim transcription of origami's gfx1250 block in
    # shared/origami/include/origami/hardware.hpp (INSTRUCTION_MAP,
    # ~lines 525-605). Keep the two in lockstep: if that block changes,
    # this dict is stale and the MI-latency feature silently drifts away
    # from what origami's own cost model predicts.
    "gfx1250": {
        (16, 16, 4, "float"): 16,
        (16, 16, 4, "double"): 16,
        (16, 16, 4, "complexfloat"): 64,
        (16, 16, 4, "complexdouble"): 64,
        (16, 16, 32, "half"): 8,
        (16, 16, 32, "bfloat16"): 8,
        (16, 16, 64, "float8"): 4,
        (16, 16, 128, "float8"): 8,
        (16, 16, 64, "bfloat8"): 4,
        (16, 16, 128, "bfloat8"): 8,
        (16, 16, 64, "float8bfloat8"): 4,
        (16, 16, 128, "float8bfloat8"): 8,
        (16, 16, 64, "bfloat8float8"): 4,
        (16, 16, 128, "bfloat8float8"): 8,
        (16, 16, 128, "float6"): 8,
        (16, 16, 128, "float4"): 4,
        (32, 16, 128, "float4"): 8,
        (16, 16, 64, "int8"): 8,
        # xf32 here is the same 3x-bf16-emulation trick as on gfx950:
        # 8 cycles for the bf16 WMMA, x3 passes = 24.
        (16, 16, 32, "xfloat32"): 24,
    },
}
_MI_LATENCY_TABLE["gfx942"] = _MI_LATENCY_TABLE["gfx950"]

# Archs we have already complained about, so the fallback warning below
# fires once per process per arch instead of once per get_mi_latency()
# call (this is on the per-kernel feature path -- millions of calls).
_WARNED_UNKNOWN_ARCH: set = set()


def get_mi_latency(
    mi_m: int, mi_n: int, mi_k: int, mi_dtype: Any, arch: str = "gfx950"
) -> float:
    """Origami::hardware_t::get_mi_latency: cycles divided by parallel_mi_cu."""
    hw = get_hw(arch)
    # An arch with no table of its own used to silently inherit gfx950's
    # MFMA cycle counts. That is not a harmless default: gfx1250's WMMA
    # instructions are 2-8x cheaper per MI, so a mis-keyed arch would
    # train on systematically wrong latency features with nothing in the
    # logs to show for it. We still fall back rather than raising -- this
    # runs deep inside feature extraction and a hard failure would kill a
    # multi-hour run -- but we say so, loudly and exactly once per arch.
    # An ASIC revision (`gfx1250v0`) executes the same ISA as its bare
    # compiler target (`gfx1250`), so the MI table is keyed by the target.
    # The config's `arch` is the REVISION -- it has to be, to select the
    # right kernel tree and deploy path -- so canonicalise before lookup
    # or every revisioned arch takes the unknown-arch fallback below.
    table = _MI_LATENCY_TABLE.get(canonical_arch(arch))
    if table is None:
        if arch not in _WARNED_UNKNOWN_ARCH:
            _WARNED_UNKNOWN_ARCH.add(arch)
            print(
                f"[features] WARNING: no _MI_LATENCY_TABLE entry for arch "
                f"{arch!r}; substituting gfx950 MI latencies. MI-latency "
                f"features for this arch are approximate -- add a "
                f"_MI_LATENCY_TABLE[{arch!r}] block transcribed from "
                f"origami hardware.hpp's INSTRUCTION_MAP.",
                file=sys.stderr,
                flush=True,
            )
        table = _MI_LATENCY_TABLE["gfx950"]
    dt = _normalize_dtype_name(mi_dtype)
    key = (int(mi_m), int(mi_n), int(mi_k), dt)
    raw = table.get(key, 32)
    pmc = hw.get("parallel_mi_cu", 4)
    return raw / max(pmc, 1)


# ── main feature computation ─────────────────────────────────────────────────


def compute_generic_features(
    # problem_t
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    a_transpose: Any = "N",
    b_transpose: Any = "N",
    a_dtype: Any = "bf16",
    b_dtype: Any = "bf16",
    c_dtype: Any = "bf16",
    d_dtype: Any = "bf16",
    mi_dtype: Any = "bf16",
    # config_t
    mt_m: int = 1,
    mt_n: int = 1,
    mt_k: int = 1,
    mi_m: int = 1,
    mi_n: int = 1,
    mi_k: int = 1,
    hand_optimized_main_loop: Any = False,  # kept for call-site compat; unused
    occupancy: int = 1,
    cache_hints_a: int = 0,
    cache_hints_b: int = 0,
    grvw_a: int = 1,
    grvw_b: int = 1,
    gwvw_d: int = 1,
    # hardware
    arch: str = "gfx950",
    # output shape
    as_dict: bool = False,
) -> Any:
    """Compute the v3 generic feature vector.

    Returns either an ordered dict {name: value} (when `as_dict=True`)
    or a list of floats in the order `feature_names()` defines.

    The `max(x, 1)` clamps on tile/MI/grvw dimensions are floors that
    protect downstream divisions and log2() calls from div-by-zero on
    pathological inputs (e.g. a CSV row with `mt_m=""` decoded to 0,
    or a default kwarg used to score a problem with no candidate yet).
    Real bench rows always have mt/mi/grvw >= 1, so the clamp is a
    no-op for valid data.
    """
    hw = get_hw(arch)
    N_CU: int = hw["N_CU"]
    LDS: int = hw["lds_capacity_bytes"]
    L2: int = hw["L2_capacity_bytes"]
    mem_bw_coef: List[float] = hw["mem_bw_per_wg_coefficients"]

    m_f, n_f, k_f, b_f = float(m), float(n), float(k), float(batch)
    mt_mf = float(max(mt_m, 1))
    mt_nf = float(max(mt_n, 1))
    mt_kf = float(max(mt_k, 1))
    mi_mf = float(max(mi_m, 1))
    mi_nf = float(max(mi_n, 1))
    mi_kf = float(max(mi_k, 1))
    occ_f = float(max(occupancy, 1))
    grvw_af = float(max(grvw_a, 1))
    grvw_bf = float(max(grvw_b, 1))
    gwvw_df = float(max(gwvw_d, 1))

    bpe_a = bpe_for_dtype(a_dtype)
    bpe_b = bpe_for_dtype(b_dtype)
    bpe_c = bpe_for_dtype(c_dtype)
    bpe_d = bpe_for_dtype(d_dtype)
    nta = int(cache_hints_a)
    ntb = int(cache_hints_b)

    # TF32 on gfx950 is emulated as 3x BF16 MFMA passes (TF32_x3), so a
    # MAC costs ~3x. Reflect that in every FLOP-derived intensity feature
    # (problem AI, tile AI, compute density, total FLOPs) so the model
    # sees xf32 as the compute-heavy workload it is -- matching origami's
    # emulated_tf32_arithmetic_intensity (3*2*MNK) and the 3x MI-latency
    # table above. bf16 / f8 / f32-native paths keep mult=1.0.
    flop_mult = 3.0 if _normalize_dtype_name(mi_dtype) == "xfloat32" else 1.0

    # ── problem-side derived ────────────────────────────────────────────────
    mn = m_f * n_f
    mk = m_f * k_f
    nk = n_f * k_f
    total_flops = flop_mult * 2.0 * m_f * n_f * k_f * b_f
    a_bytes = mk * bpe_a * b_f
    b_bytes = nk * bpe_b * b_f
    c_bytes = mn * bpe_c * b_f
    d_bytes = mn * bpe_d * b_f
    total_bytes = a_bytes + b_bytes + c_bytes + d_bytes
    ai_prob = total_flops / max(total_bytes, 1.0)

    # ── tile / grid / utilization ──────────────────────────────────────────
    nt_m = math.ceil(m_f / mt_mf)
    nt_n = math.ceil(n_f / mt_nf)
    num_tiles = nt_m * nt_n
    num_tiles_total = num_tiles * b_f
    k_iters = math.ceil(k_f / mt_kf)
    tile_coverage = (mt_mf * mt_nf) / max(mn, 1.0)

    waves = math.ceil(num_tiles / N_CU)
    wave_eff = num_tiles / (waves * N_CU) if waves > 0 else 1.0
    rho = num_tiles_total / N_CU
    batch_tiles_ratio = b_f * num_tiles / N_CU

    launched_m = nt_m * mt_mf
    launched_n = nt_n * mt_nf
    launched_k = k_iters * mt_kf
    util_out = mn / max(launched_m * launched_n, 1.0)
    util_3d = (m_f * n_f * k_f) / max(launched_m * launched_n * launched_k, 1.0)

    # ── memory / capacity ──────────────────────────────────────────────────
    lds_bytes = mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b
    lds_ratio = lds_bytes / LDS
    l2_fit_ratio = total_bytes / L2
    bw_per_cu = total_bytes / N_CU
    l2_working_set = (nt_m * mt_mf * mt_kf * bpe_a) + (nt_n * mt_nf * mt_kf * bpe_b)
    l2_fit_ws = min(l2_working_set / L2, 2.0) / 2.0

    # ── compute latency / origami AI_tile ──────────────────────────────────
    L_MI = get_mi_latency(int(mi_mf), int(mi_nf), int(mi_kf), mi_dtype, arch)
    n_mi = (
        math.ceil(mt_mf / mi_mf) * math.ceil(mt_nf / mi_nf) * math.ceil(mt_kf / mi_kf)
    )
    L_MT = n_mi * L_MI
    ai_tile = (flop_mult * 2.0 * mt_mf * mt_nf * mt_kf) / (
        mt_mf * mt_kf + mt_nf * mt_kf + mt_mf * mt_nf
    )
    active_cus = min(num_tiles_total, float(N_CU))
    bw_occ = min(
        1.0,
        mem_bw_coef[0] * active_cus * active_cus
        + mem_bw_coef[1] * active_cus
        + mem_bw_coef[2],
    )

    # ── assemble features ──────────────────────────────────────────────────
    feats: "OrderedDict[str, float]" = OrderedDict()

    # Problem (problem_t only)
    feats["log2_m"] = math.log2(max(m_f, 1.0))
    feats["log2_n"] = math.log2(max(n_f, 1.0))
    feats["log2_k"] = math.log2(max(k_f, 1.0))
    feats["log2_b"] = math.log2(max(b_f, 1.0))
    feats["ai_prob"] = min(ai_prob, 1e6)
    feats["log2_ai_prob"] = math.log2(max(ai_prob, 0.001))
    feats["aspect_mn"] = min(max(m_f / max(n_f, 1.0), 0.001), 10000.0)
    feats["aspect_mk"] = min(max(m_f / max(k_f, 1.0), 0.001), 10000.0)
    feats["aspect_nk"] = min(max(n_f / max(k_f, 1.0), 0.001), 10000.0)

    def _is_pow2(x: float) -> float:
        xi = int(x)
        return 1.0 if xi > 0 and (xi & (xi - 1)) == 0 else 0.0

    feats["is_pow2_m"] = _is_pow2(m_f)
    feats["is_pow2_n"] = _is_pow2(n_f)
    feats["is_pow2_k"] = _is_pow2(k_f)

    feats["trans_a"] = float(_transpose_bit(a_transpose))
    feats["trans_b"] = float(_transpose_bit(b_transpose))
    feats["dtype_a_id"] = float(_dtype_id(a_dtype))
    feats["dtype_b_id"] = float(_dtype_id(b_dtype))
    feats["dtype_c_id"] = float(_dtype_id(c_dtype))
    feats["dtype_d_id"] = float(_dtype_id(d_dtype))
    feats["dtype_mi_id"] = float(_dtype_id(mi_dtype))
    feats["bpe_a"] = bpe_a
    feats["bpe_b"] = bpe_b

    # Modular alignment fingerprints
    for base in (256, 128, 64, 32, 16, 8):
        feats[f"m_mod_{base}"] = float(int(m_f) % base)
        feats[f"n_mod_{base}"] = float(int(n_f) % base)
    for base in (64, 128, 256):
        feats[f"m_align_{base}"] = min(int(m_f) % base, base - (int(m_f) % base)) / base
        feats[f"n_align_{base}"] = min(int(n_f) % base, base - (int(n_f) % base)) / base

    # Enumerated-tile geometry (problem-only proxies of tile geometry at
    # fixed candidate tile sizes -- usable in the query tower without
    # knowing the actual config_t).
    for st in (32, 64, 128, 256):
        s_nt_m = math.ceil(m_f / st)
        s_nt_n = math.ceil(n_f / st)
        feats[f"log2_enum_tiles_{st}"] = math.log2(max(s_nt_m * s_nt_n, 1))
        feats[f"enum_coverage_{st}"] = mn / max(s_nt_m * st * s_nt_n * st, 1.0)
    for kd in (32, 64, 128, 256):
        feats[f"log2_enum_k_iters_{kd}"] = math.log2(max(math.ceil(k_f / kd), 1))
    for st in (128, 256):
        s_nt = math.ceil(m_f / st) * math.ceil(n_f / st)
        s_w = math.ceil(s_nt / N_CU)
        feats[f"log2_enum_waves_{st}"] = math.log2(max(s_w, 1))
        feats[f"enum_wave_eff_{st}"] = s_nt / max(s_w * N_CU, 1.0)

    # Tile (config_t only)
    feats["log2_mt_m"] = math.log2(mt_mf)
    feats["log2_mt_n"] = math.log2(mt_nf)
    feats["log2_mt_k"] = math.log2(mt_kf)
    feats["log2_mi_m"] = math.log2(mi_mf)
    feats["log2_mi_n"] = math.log2(mi_nf)
    feats["log2_mi_k"] = math.log2(mi_kf)
    feats["nta_norm"] = nta / 7.0
    feats["ntb_norm"] = ntb / 7.0
    feats["occupancy_norm"] = occ_f / 9.0
    feats["grvw_a_norm"] = grvw_af / 8.0
    feats["grvw_b_norm"] = grvw_bf / 8.0
    feats["gwvw_d_norm"] = gwvw_df / 8.0

    # Interaction (problem × config)
    feats["log2_num_tiles"] = math.log2(max(num_tiles, 1))
    feats["log2_num_tiles_total"] = math.log2(max(num_tiles_total, 1))
    feats["log2_k_iters"] = math.log2(max(k_iters, 1))
    feats["log2_waves"] = math.log2(max(waves, 1))
    feats["wave_eff"] = wave_eff
    feats["log2_rho"] = math.log2(max(rho, 0.001))
    feats["log2_batch_tiles_ratio"] = math.log2(max(batch_tiles_ratio, 0.001))
    feats["util_out"] = util_out
    feats["util_3d"] = util_3d
    feats["tile_coverage"] = min(tile_coverage, 1.0)
    feats["log2_lds_bytes"] = math.log2(max(lds_bytes, 1.0))
    feats["lds_ratio"] = lds_ratio
    feats["l2_fit_ratio"] = min(l2_fit_ratio, 4.0) / 4.0
    feats["l2_fit_ws"] = l2_fit_ws
    feats["log2_bw_per_cu"] = math.log2(max(bw_per_cu, 1.0))
    feats["log2_total_bytes"] = math.log2(max(total_bytes, 1.0))
    feats["log2_total_flops"] = math.log2(max(total_flops, 1.0))
    feats["m_coverage"] = min(1.0, mt_mf / max(m_f, 1.0))
    feats["n_coverage"] = min(1.0, mt_nf / max(n_f, 1.0))
    feats["k_tail"] = (k_f - (k_iters - 1) * mt_kf) / max(k_f, 1.0)
    feats["k_aligned_128B"] = (
        1.0 if ((k_f * bpe_a) % 128.0 == 0 and (mt_kf * bpe_a) % 128.0 == 0) else 0.0
    )
    feats["small_m_tile"] = 1.0 if m_f <= 2 * mt_mf else 0.0
    feats["small_n_tile"] = 1.0 if n_f <= 2 * mt_nf else 0.0
    feats["is_batched"] = 1.0 if b_f > 1 else 0.0
    feats["log2_a_read_bytes"] = math.log2(max(mt_mf * mt_kf * bpe_a, 1.0))
    feats["log2_b_read_bytes"] = math.log2(max(mt_nf * mt_kf * bpe_b, 1.0))
    feats["log2_compute_density"] = math.log2(
        max(
            (mt_mf * mt_nf * mt_kf * 2.0 * flop_mult)
            / max(mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b, 1.0),
            0.001,
        )
    )
    feats["log2_total_work"] = math.log2(max(k_iters * num_tiles * b_f, 1))
    feats["batch_k_iters_per_tile"] = b_f * k_iters / max(num_tiles, 1)
    feats["batch_wave_eff"] = (
        num_tiles_total / (math.ceil(num_tiles_total / N_CU) * N_CU)
        if num_tiles_total > 0
        else 1.0
    )
    feats["log2_grvw_a_bytes"] = math.log2(max(grvw_af * bpe_a, 1.0))
    feats["log2_grvw_b_bytes"] = math.log2(max(grvw_bf * bpe_b, 1.0))

    # Hardware proxies. These are features computed from HARDWARE
    # CONSTANTS (per-arch N_CU, parallel_mi_cu, MI instruction-cycle
    # table, mem-bw-vs-occupancy polynomial) provided by Origami's
    # `hardware_t` -- NOT outputs of Origami's analytical latency
    # predictor. They're inputs to the MLP; the MLP still does its
    # own ranking on top of them. Pulled from Origami so we don't
    # duplicate per-arch sizing constants in Python.
    feats["ai_tile"] = ai_tile
    feats["l_mi_cycles"] = L_MI
    feats["log2_l_mt_cycles"] = math.log2(max(L_MT, 1.0))
    feats["bw_occ"] = bw_occ
    feats["active_cus_norm"] = active_cus / N_CU

    if as_dict:
        return feats
    return list(feats.values())


# ── schema / two-tower split ─────────────────────────────────────────────────

FEATURE_GROUPS = {
    "problem": [
        # m/n/k/b log scales
        "log2_m",
        "log2_n",
        "log2_k",
        "log2_b",
        # arithmetic intensity
        "ai_prob",
        "log2_ai_prob",
        # aspect ratios
        "aspect_mn",
        "aspect_mk",
        "aspect_nk",
        # power-of-2 indicators
        "is_pow2_m",
        "is_pow2_n",
        "is_pow2_k",
        # layout / dtype
        "trans_a",
        "trans_b",
        "dtype_a_id",
        "dtype_b_id",
        "dtype_c_id",
        "dtype_d_id",
        "dtype_mi_id",
        "bpe_a",
        "bpe_b",
    ],
    "modular": [
        "m_mod_256",
        "m_mod_128",
        "m_mod_64",
        "m_mod_32",
        "m_mod_16",
        "m_mod_8",
        "n_mod_256",
        "n_mod_128",
        "n_mod_64",
        "n_mod_32",
        "n_mod_16",
        "n_mod_8",
        "m_align_64",
        "m_align_128",
        "m_align_256",
        "n_align_64",
        "n_align_128",
        "n_align_256",
    ],
    "enum_tiles": [
        "log2_enum_tiles_32",
        "log2_enum_tiles_64",
        "log2_enum_tiles_128",
        "log2_enum_tiles_256",
        "enum_coverage_32",
        "enum_coverage_64",
        "enum_coverage_128",
        "enum_coverage_256",
        "log2_enum_k_iters_32",
        "log2_enum_k_iters_64",
        "log2_enum_k_iters_128",
        "log2_enum_k_iters_256",
        "log2_enum_waves_128",
        "log2_enum_waves_256",
        "enum_wave_eff_128",
        "enum_wave_eff_256",
    ],
    "tile": [
        "log2_mt_m",
        "log2_mt_n",
        "log2_mt_k",
        "log2_mi_m",
        "log2_mi_n",
        "log2_mi_k",
        "nta_norm",
        "ntb_norm",
        "occupancy_norm",
        "grvw_a_norm",
        "grvw_b_norm",
        "gwvw_d_norm",
    ],
    "interaction": [
        "log2_num_tiles",
        "log2_num_tiles_total",
        "log2_k_iters",
        "log2_waves",
        "wave_eff",
        "log2_rho",
        "log2_batch_tiles_ratio",
        "util_out",
        "util_3d",
        "tile_coverage",
        "log2_lds_bytes",
        "lds_ratio",
        "l2_fit_ratio",
        "l2_fit_ws",
        "log2_bw_per_cu",
        "log2_total_bytes",
        "log2_total_flops",
        "m_coverage",
        "n_coverage",
        "k_tail",
        "k_aligned_128B",
        "small_m_tile",
        "small_n_tile",
        "is_batched",
        "log2_a_read_bytes",
        "log2_b_read_bytes",
        "log2_compute_density",
        "log2_total_work",
        "batch_k_iters_per_tile",
        "batch_wave_eff",
        "log2_grvw_a_bytes",
        "log2_grvw_b_bytes",
    ],
    # Features derived from HARDWARE CONSTANTS (N_CU, parallel_mi_cu,
    # MI-instruction-cycle table, mem-bw-vs-occupancy polynomial)
    # pulled from Origami's `hardware_t`. NOT outputs of Origami's
    # analytical latency predictor -- those would be SCORES, not
    # features. These are inputs the MLP consumes alongside problem
    # / tile / interaction features. Renamed from "origami_borrowed"
    # to make the distinction explicit.
    "hw_proxies": [
        "ai_tile",
        "l_mi_cycles",
        "log2_l_mt_cycles",
        "bw_occ",
        "active_cus_norm",
    ],
}


# The former `ml_kernel_params` / `ml_kernel_params_interaction` groups
# (origami config_ml_t: NTC/D/E, prefetch depth, LDS pad / LRVW, LocalSplitU,
# log2_lds_bytes_padded) were REMOVED. They were populated only on
# un-benched origami "skip" rows and were empty/0 on every real benched
# candidate the deployed model ranks -- a dead, train/serve-skew-prone set.
# Origami's analytical scorer never read them either. Dropping them here
# bumps feature_names_hash, so models must be retrained and served with the
# matching C++ runtime that omits these features.
QUERY_GROUPS = ["problem", "modular", "enum_tiles"]
ITEM_GROUPS = ["tile"]
INTERACTION_GROUPS = ["interaction", "hw_proxies"]


def query_feature_names() -> List[str]:
    return [n for g in QUERY_GROUPS for n in FEATURE_GROUPS[g]]


def item_feature_names() -> List[str]:
    return [n for g in ITEM_GROUPS for n in FEATURE_GROUPS[g]]


def interaction_feature_names() -> List[str]:
    return [n for g in INTERACTION_GROUPS for n in FEATURE_GROUPS[g]]


def feature_names() -> List[str]:
    return query_feature_names() + item_feature_names() + interaction_feature_names()


def feature_count() -> int:
    return len(feature_names())


def feature_names_hash() -> str:
    """SHA256 hex prefix over the canonical feature name list. Pair with
    FEATURES_VERSION to catch silent schema drift across runs."""
    import hashlib

    return hashlib.sha256("\n".join(feature_names()).encode("utf-8")).hexdigest()[:16]


def split_features(feats: Any) -> Tuple[List[float], List[float], List[float]]:
    """Split a feature dict (or compute_generic_features list) into
    (query_vec, item_vec, interaction_vec)."""
    if isinstance(feats, list):
        names = feature_names()
        feats = {n: v for n, v in zip(names, feats)}
    q = [feats[n] for n in query_feature_names()]
    it = [feats[n] for n in item_feature_names()]
    ix = [feats[n] for n in interaction_feature_names()]
    return q, it, ix


# ── smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    names = feature_names()
    groups = {g: len(v) for g, v in FEATURE_GROUPS.items()}
    print(f"FEATURES_VERSION = {FEATURES_VERSION}")
    print(f"feature_count    = {len(names)}")
    print(f"groups           = {groups}  (sum={sum(groups.values())})")
    print(f"feature_names_hash = {feature_names_hash()}")
    if sum(groups.values()) != len(names):
        all_in_groups = set().union(*FEATURE_GROUPS.values())
        print(
            "  extras in compute but not in groups:", sorted(set(names) - all_in_groups)
        )
        print("  in groups but not computed:", sorted(all_in_groups - set(names)))
    sample = compute_generic_features(
        m=2048,
        n=1024,
        k=512,
        batch=1,
        a_transpose="T",
        b_transpose="N",
        a_dtype="bf16",
        b_dtype="bf16",
        c_dtype="bf16",
        d_dtype="bf16",
        mi_dtype="bf16",
        mt_m=128,
        mt_n=128,
        mt_k=32,
        mi_m=16,
        mi_n=16,
        mi_k=32,
        occupancy=2,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
        as_dict=True,
    )
    print(f"sample_len = {len(sample)}")
