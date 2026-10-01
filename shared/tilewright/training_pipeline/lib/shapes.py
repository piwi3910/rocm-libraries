# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shape dataclass + rotating-512 OOM filter + per-cell synthesis.

Self-contained: no imports from outside pipeline_v3.

`is_rotating512_safe` is the rotating:512 OOM-safety predicate. It was
first calibrated against observed OOM cases on MI300X / MI350X as two
hard caps:
  - rules out shapes with per-matrix bytes > 256 MiB;
  - rules out B>1 with max(M,N)>256 OR K>512
    (the rotating-buffer allocator scales batched buffers super-
     linearly and OOMs the bench tool at rotating:512).
Both are now superseded by the total-footprint + FLOP guard in
`per_matrix_bytes_feasible` (see `is_rotating512_safe`'s docstring).

`synthesize_cell_shapes` is the cell-uniform sampler used by stage01:
log-uniform on M, N, K within tier range, with a 30% chance per axis
of snapping onto a common macro-tile boundary (16/32/64/128/256) so
the most-tested kernel sizes are well-represented.
"""
from __future__ import annotations

import math
from pathlib import Path as _Path
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .grid import BANY_VALUES, MT_VALUES, TIER_RANGES_K, TIER_RANGES_MN


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    batch: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.m, self.n, self.k, self.batch)


# Bytes-per-element table -- used by the feasibility filter to bound
# the total device-memory footprint of a benched shape.
_BPE = {
    "bf16_r": 2,
    "f16_r": 2,
    "f8_r": 1,
    "bf8_r": 1,
    "f32_r": 4,
    "i8_r": 1,
    "i32_r": 4,
    "xf32_r": 4,
}

# _BPE covers the OPERAND elements only. An MX GEMM also allocates a
# block-SCALE tensor alongside A and alongside B: one 1-byte scale
# element (E8M0 or E4M3 -- both 1 byte) per `block` operand elements.
# MXF8 with E8M0 block-32 scales on both operands is therefore +1/32 =
# +3.125% on A and on B, which _BPE alone silently under-counts.
# Keys are hipblaslt-bench's scaleA/scaleB MODE ints (ScalingFormat, in
# clients/common/include/hipblaslt_scaling_format.hpp):
#   0=None 1=Scalar 2=Vector 3=Block_32_UE8M0 4=Block_16_UE8M0
#   5=Block_32_UE4M3 ... 1001=Block_32_UE8M0_32_8_EXT
# Only the BLOCK modes are listed; 0/1/2 allocate no per-block tensor
# (Vector is O(M)+O(N) bytes, <1e-4 of the operand -- ignored), so every
# non-MX path keeps exactly the arithmetic it had before.
_MX_SCALE_BLOCK = {3: 32, 4: 16, 5: 32, 1001: 32}


def _scale_bytes_per_element(scale_mode: int) -> float:
    """Extra scale-tensor bytes per OPERAND element for `scale_mode`.

    1 byte per `block` elements for the MX block modes; 0.0 for every
    non-block mode (none / Scalar / Vector / unknown)."""
    block = _MX_SCALE_BLOCK.get(int(scale_mode or 0))
    return 0.0 if block is None else 1.0 / block


# ── feasibility = OOM guard, calibrated to the real OOB envelope ──────
#
# The feasibility filter exists ONLY to keep a shape benchable without
# OOM-ing hipblaslt-bench at the fixed rotating:512 standard -- it is
# NOT a cost knob. It was previously a 256 MiB *per-matrix* cap, which
# wrongly rejected a sizeable share of the canonical large OOB benchmark
# workload -- shapes that demonstrably bench fine. That workload (all
# known-good at rotating:512 on a 309 GB MI350X) contains single
# matrices and total A+B+C+D footprints far above 256 MiB, so the real
# OOM ceiling is the TOTAL footprint vs HBM, not a small per-matrix
# cap. We bound total footprint (A+B+C+D + a rotating:512 overhead
# allowance) to a safe fraction of the 309 GB device.
_HBM_SAFE_BYTES = 220 * 1024**3  # 220 GiB of 309 GB; leaves headroom
# for the rotating buffer + library
# working set + fragmentation.
# gfx950 value AND the no-arch default,
# so every existing caller is unchanged.

# Per-arch budgets, keyed by the same `arch` string the rest of the
# pipeline threads around (cfg["arch"], lib.features.get_hw(arch), ...).
# Deliberately NOT probed from the device here: stage01 shape synthesis
# must be reproducible from (config, seed) alone -- on a host with no
# GPU, and identically across every device of a multi-GPU box -- so a
# rocm-smi / HIP query at this point would make the generated envelope
# machine-dependent and could silently move the frozen gfx950 envelope.
# A fixed per-arch number is baked in instead:
#   gfx950 : 220 GiB of the 309 GB MI350X -- EXACT prior value, frozen.
#   gfx1250: held at the SAME proportion of the device's total VRAM as
#            gfx950 (220 / 309 ~= 0.712), rounded down to a whole GiB.
#            (Reading "309 GB" as decimal bytes instead gives gfx950 a
#            larger fraction, ~76%, and so a larger budget here. We take
#            the ~71% reading -- the conservative one -- because an
#            OOM costs a whole bench block while a trimmed envelope only
#            costs a few corner shapes.)
_HBM_SAFE_BYTES_BY_ARCH = {
    "gfx950": _HBM_SAFE_BYTES,
    "gfx1250": 307 * 1024**3,
}


def hbm_safe_bytes(arch: Optional[str] = None) -> int:
    """Total-footprint budget for `arch`; gfx950's 220 GiB when `arch`
    is None or unknown.

    Unknown archs fall back to the SMALLEST known budget on purpose:
    over-estimating an unseen device's memory OOMs the bench, while
    under-estimating only trims the high-footprint corner."""
    # Smallest known budget for None/unknown, per the contract above -- NOT
    # gfx950's constant. Those coincide today only because gfx950 happens to
    # be the smallest entry; hardcoding it would silently OVER-estimate memory
    # for any future arch with a tighter budget, which is exactly the OOM this
    # guard exists to prevent.
    _smallest = min(_HBM_SAFE_BYTES_BY_ARCH.values(), default=_HBM_SAFE_BYTES)
    if arch is None:
        return _smallest
    key = str(arch)
    if key in _HBM_SAFE_BYTES_BY_ARCH:
        return _HBM_SAFE_BYTES_BY_ARCH[key]
    # Tolerate a silicon-revision suffix: the built library directory is
    # `gfx1250v0` while configs say `gfx1250`. Longest prefix wins.
    for known in sorted(_HBM_SAFE_BYTES_BY_ARCH, key=len, reverse=True):
        if key.startswith(known):
            return _HBM_SAFE_BYTES_BY_ARCH[known]
    return _smallest


# FLOPs is the COST dial, NOT a feasibility limit. 2*M*N*K*B bounds the
# per-kernel wall-time AND -- because the feasibility filter runs at
# SYNTHESIS time -- thins the expensive shapes out of the sampled pool
# before they ever hit the bench.
#
# A looser cap (8e13) lets the all-axes-LARGE octant fill with
# large-SQUARE shapes (M, N and K all in the tens of thousands) that are
# very slow to bench across all candidate solutions; with dozens of them
# per cell under n_strata=2, per-cell bench time becomes untenable.
# Those large-square shapes are ALSO the saturated regime (a few large
# tiles win) and are NOT real-workload-shaped (real huge GEMMs are
# tall-skinny: one big axis). 1e13 keeps the tall-skinny huge shapes
# (M or N or K alone up to the tier ceiling) -- which is the real OOB
# structure we must cover -- while rejecting the large-square corner at
# generation time, keeping per-shape bench cost low. The
# excluded high-FLOP corner is covered by (a) winner saturation (model
# extrapolates the asymptotic large-tile pick) and (b) Origami-fallback
# for anything outside the trained envelope. This is also the dial that
# keeps the productized full pipeline within its <=6h budget.
_MAX_FLOPS = 1e13

# Largest per-matrix element count (a leading-dim batch stride) a hipBLASLt
# kernel can index with int32 before an offset overflows into an illegal memory
# access. Caps M*N / M*K / N*K (see per_matrix_bytes_feasible).
_INT32_MAX = 2**31 - 1

# Batched-only guards (see per_matrix_bytes_feasible). Batched shapes hang
# hipblaslt-bench at rotating:512 in two ways: (1) huge working sets near
# HBM, and (2) EXTREME aspect ratios (one dim ~10^5) wedge kernel
# selection even at modest footprint. So B>1 is bounded by BOTH a
# working-set ceiling AND a per-dimension cap. 8192 stays well inside the
# Large tier (>512) so batched Large/LargeK cells still populate, and
# covers the dims real batched (attention/MoE) workloads actually use.
_BATCHED_SAFE_BYTES = 16 * 1024**3
_BATCHED_MAX_DIM = 8192
# Batched kernels run B independent matmuls; very large total FLOPs make
# per-kernel warm-up slow and correlate with hipblaslt-bench teardown
# faults (`hipModuleUnload: illegal memory access`). Cap batched total
# FLOPs at 1e12 (10x below the 1e13 non-batched cap): this keeps
# batched kernels fast, matches the real "big-B => small compute"
# envelope (the gfx942 GridBased B=4096 point is ~6.5e11 FLOPs), and
# admits the batched shapes seen in real-world workload traces.
_BATCHED_MAX_FLOPS = 1e12


def per_matrix_bytes_feasible(
    m: int,
    n: int,
    k: int,
    b: int,
    bpe: int = 2,
    arch: Optional[str] = None,
    scale_a: int = 0,
    scale_b: int = 0,
) -> bool:
    """OOM/cost feasibility for one benched shape at rotating:512.

    Bounds (1) total device footprint A+B+C+D (+ MX block scales, +
    rotating overhead) against `hbm_safe_bytes(arch)`, the real OOM
    constraint, and (2) FLOPs against `_MAX_FLOPS`, the cost dial. Name
    kept for API stability; semantics are now total-footprint, not
    per-matrix (the old 256 MiB per-matrix cap dropped a sizeable share
    of the known-benchable OOB envelope).

    `arch` selects the HBM budget (None -> gfx950's 220 GiB, so callers
    that do not pass it are unchanged). `scale_a` / `scale_b` are the
    hipblaslt-bench scale MODE ints; only the MX block modes add bytes
    (see `_MX_SCALE_BLOCK`)."""
    if m <= 0 or n <= 0 or k <= 0 or b <= 0:
        return False
    # int32 indexing guard. hipBLASLt/Tensile kernels index some per-matrix
    # element offsets (leading-dim batch strides: A=m*k, B=n*k, C/D=m*n) with
    # int32; when a matrix's element count exceeds INT32_MAX the kernel computes
    # an out-of-bounds address and the bench dies with "an illegal memory access
    # was encountered (700)". Not caught by the HBM/FLOP guards: a tiny-K
    # extreme-aspect shape (large M and N, K of a few dozen -> stride_d = m*n
    # beyond INT32_MAX) has low FLOPs and a footprint inside the HBM budget,
    # so it passes both yet crashes the bench. Reject any shape whose largest
    # 2D matrix overflows.
    if max(m * n, m * k, n * k) > _INT32_MAX:
        return False
    a_bytes = m * k * b * bpe
    b_bytes = n * k * b * bpe
    if scale_a or scale_b:
        # MX block scales ride along with A and B (C/D are unscaled).
        # Round UP -- a systematic under-estimate here is an OOM, an
        # over-estimate only costs a corner shape. Guarded so the
        # non-MX path keeps its exact integer arithmetic.
        a_bytes += math.ceil(m * k * b * _scale_bytes_per_element(scale_a))
        b_bytes += math.ceil(n * k * b * _scale_bytes_per_element(scale_b))
    cd_bytes = 2 * m * n * b * bpe  # C and D (beta=0 still allocs D)
    working_set = a_bytes + b_bytes + cd_bytes
    rotating_overhead = 512 * 1024 * 1024  # rotating:512 worst-case extra
    if working_set + rotating_overhead > hbm_safe_bytes(arch):
        return False
    if 2.0 * m * n * k * b > _MAX_FLOPS:
        return False
    # Batched (B>1) guard: hipblaslt-bench HANGS at rotating:512 on very
    # large batched working sets WELL BELOW the HBM ceiling (220 GiB on
    # gfx950; see `hbm_safe_bytes`) -- B
    # multiplies the per-call allocation and rotating buffers, and extreme
    # aspect ratios (e.g. N*K*B ~ 200 GiB) wedge the bench (GPU idle, no
    # progress). The non-batched envelope is unchanged (FLOP cap + 220 GiB
    # ceiling); for batched we additionally cap the working set to a
    # benchable size. 16 GiB still admits e.g. Large|Large|LargeK|Bany at
    # M=N=K~8192, B=8 (~4 GiB) so the batched cells get populated + trained
    # without the multi-GiB allocations that hang the prober.
    if b > 1 and (
        working_set > _BATCHED_SAFE_BYTES
        or max(m, n, k) > _BATCHED_MAX_DIM
        or 2.0 * m * n * k * b > _BATCHED_MAX_FLOPS
    ):
        return False
    return True


# ── known-faulting shape exclusion ───────────────────────────────────────────
#
# Some (arch, dtype) kernel pools contain candidates that FAULT rather than run
# on a well-defined region of the shape space. On gfx1250v0 that region is
# dominated by large reductions (large K) plus degenerate-aspect shapes -- see
# `is_known_faulting` for the exact rules. hipblaslt-bench emits
# HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION and then either wedges or exits
# non-zero.
#
# These shapes yield NO training data -- they cannot be measured at all -- but
# stage03 still pays a full process restart per shape to kill the process and
# reload every kernel module in a fresh one. Because stage01 emits shapes in
# CELL order the faulting ones CLUSTER into a few bench blocks, each paying
# many reloads on a single GPU. stage03 cannot finish before its slowest
# block, so those blocks alone set the runtime.
#
# Excluding them costs a small fraction of the shape budget (and synthesis
# refills the cell, so the count is preserved) to take the reloads off the
# critical path. This is a WORKAROUND for a kernel bug, not a modelling
# decision -- drop it once the faulting kernels are fixed.

# ── observed-fault blocklist ─────────────────────────────────────────────────
# Shapes that actually crashed a bench in a PRIOR run. Geometry rules alone
# cannot predict every gfx1250 fault, but a shape that crashed once is likely
# to crash again, so excluding what has already crashed removes part of the
# fault cost per iteration, and the list grows as more runs report.
#
# Why this matters: a fault is not just a lost shape -- stage03 respawns the
# bench and re-pays a full kernel-module load, so when faults are frequent the
# reloads, not the benchmarking, dominate the run.
#
# The blocklist file (`{"shapes": [[M, N, K], ...]}`) is optional and is not
# shipped with this pipeline. When it is absent or unreadable the blocklist is
# treated as empty and only the geometric rules in `is_known_faulting` apply.
_BLOCKLIST_PATH = _Path(__file__).resolve().parent / "data" / "bbs_fault_blocklist.json"
_BLOCKLIST_CACHE: Optional[set] = None


def _fault_blocklist() -> set:
    global _BLOCKLIST_CACHE
    if _BLOCKLIST_CACHE is None:
        try:
            import json as _json

            with open(_BLOCKLIST_PATH) as fh:
                _BLOCKLIST_CACHE = {tuple(x) for x in _json.load(fh).get("shapes", [])}
        except Exception:
            _BLOCKLIST_CACHE = set()
    return _BLOCKLIST_CACHE


def is_known_faulting(
    m: int,
    n: int,
    k: int,
    scale_a: int = 0,
    scale_b: int = 0,
    arch: Optional[str] = None,
) -> bool:
    """True for shapes in a known kernel-fault region. Conservative: an arch
    other than gfx1250 is never filtered, so nothing else is touched."""
    # NOT gated on MX. bf16 (BBS_TN) shows the SAME K-driven fault behaviour
    # as MXFP8 once large-K shapes reach the bench unfiltered, so an MX-only
    # gate would miss it. The fault is a property of the gfx1250 Stream-K path
    # at large K, not of the block-scale mode, so scale_a/scale_b are not
    # consulted.
    if arch is not None and not str(arch).startswith("gfx1250"):
        return False  # only seen on gfx1250
    # K ALONE. An aspect-ratio predicate (`min(m,n) <= 32 and k >= 8000`, see
    # `_FAULT_MAX_MN`) does not match the observed crashes: they span a wide
    # range of min(M,N), and the min(M,N) <= 32 region is not where they
    # concentrate. Pooled over all M/N, the crash rate is driven by K -- rare
    # below the K threshold, much higher above it. The shape is not the whole
    # story either -- a shape can crash inside a block yet run clean in
    # isolation at every rotating size -- so this is a cost filter, not a
    # correctness claim: it removes the region where the stochastic fault
    # concentrates.
    #
    # A fault costs far more than a successful bench (stage03 respawns the
    # bench, paying a fresh lazy kernel-module load), so even a small fault
    # rate dominates the effective per-shape cost; filtering the high-fault K
    # region buys many more shapes per GPU-hour.
    #
    # 8000, NOT 16000. The [8000, 16000) band is not a safe region either, and
    # excluding only K >= 16000 makes stage01 RESAMPLE those shapes down into
    # it, concentrating the hazard.
    # Degenerate-aspect shapes fault independently of K: the faults the K rule
    # does not explain are dominated by min(M,N) <= 3. A GEMM with a dimension
    # of 2 or 3 degenerates the tiling: every macro tile is almost entirely
    # padding, and the Stream-K partial-tile path most of this library's
    # solutions use is where the faults land.
    # Kept deliberately tight at 3: min(M,N) does NOT predict faults in general
    # (the crash set spans small and large min(M,N) alike), so this excludes
    # only the degenerate sliver, not the Tiny tier as a whole.
    # Previously-observed crashes, regardless of geometry.
    if (int(m), int(n), int(k)) in _fault_blocklist():
        return True
    if min(m, n) <= _FAULT_MAX_DEGENERATE_MN:
        return True
    return k >= _FAULT_MIN_K


# Measured fault envelope (see is_known_faulting). _FAULT_MAX_MN is retained
# only because callers/tests import it; the predicate no longer reads it.
_FAULT_MAX_MN = 32
_FAULT_MIN_K = 8000
# Degenerate aspect ratio: a GEMM this thin faults regardless of K.
_FAULT_MAX_DEGENERATE_MN = 3


def is_rotating512_safe(
    m: int,
    n: int,
    k: int,
    b: int,
    bpe: int = 2,
    arch: Optional[str] = None,
    scale_a: int = 0,
    scale_b: int = 0,
) -> bool:
    """rotating:512 OOM-safety predicate (MI300X / MI350X).

    A shape is "safe" to bench at `rotating: 512` iff its total device
    footprint (A+B+C+D + any MX block scales, which scales with B) plus
    the rotating overhead fits `hbm_safe_bytes(arch)` (gfx950's 220 GiB
    when `arch` is None), and its FLOPs fit `_MAX_FLOPS` -- both checked
    by `per_matrix_bytes_feasible`, which is fully batch-aware.

    History: this used to add hard caps for B>1 (`max(M,N)<=256 AND
    K<=512`), a calibration response to rotating:512 OOMs that
    PREDATED the HBM-grounded footprint guard. Those caps were a crude
    proxy for "footprint too big" -- but they CONTRADICT the Large M/N
    (>=513) and LargeK (>=513) tier lower bounds, so they made 30/48 Bany
    (batched) grid cells STRUCTURALLY EMPTY (impossible to populate) ->
    untrained -> the deploy runtime mis-served batched GEMMs via the
    out-of-distribution Bnone fallback. With the HBM+FLOP footprint guard
    now doing the real OOM bounding (a batched shape that fits 220 GiB and
    1e13 FLOPs benches fine; extreme ones are rejected by footprint), the
    hard caps are removed so batched Large/LargeK cells can be populated
    and TRAINED. (stage03's per-block crash tolerance remains the safety
    net for any rare residual OOM.)
    """
    return per_matrix_bytes_feasible(
        m, n, k, b, bpe, arch=arch, scale_a=scale_a, scale_b=scale_b
    )


# ── per-cell shape synthesis ─────────────────────────────────────────────────


def _log_unif(rng: random.Random, lo: int, hi: int) -> int:
    if lo == hi:
        return lo
    lg = math.log(max(lo, 1)) + rng.random() * (math.log(hi) - math.log(max(lo, 1)))
    return max(lo, min(hi, int(round(math.exp(lg)))))


def _mt_align(v: int, lo: int, hi: int) -> int:
    # 256 first: real LargeK winners are almost always divisible by 256
    # and large M/N are often divisible by 128/256; biasing to the largest
    # macro-tile multiple that still fits the (sub-)box matches that
    # structure far better than the prior 128-max snap.
    for mult in (256, 128, 64, 32, 16):
        snapped = (v // mult) * mult
        if lo <= snapped <= hi:
            return snapped if snapped > 0 else mult
    return v


def _log_sub_range(lo: int, hi: int, bin_idx: int, n_bins: int) -> Tuple[int, int]:
    """Carve `[lo, hi]` into `n_bins` log-equal sub-intervals and
    return the (lo, hi) of bin `bin_idx`. Bins are inclusive on both
    sides; collapsed ranges (`lo == hi`) return that single point."""
    if lo >= hi or n_bins <= 1:
        return lo, hi
    log_lo = math.log(max(lo, 1))
    log_hi = math.log(max(hi, 1))
    step = (log_hi - log_lo) / n_bins
    sub_log_lo = log_lo + bin_idx * step
    sub_log_hi = log_lo + (bin_idx + 1) * step
    sub_lo = max(lo, int(math.floor(math.exp(sub_log_lo))))
    sub_hi = min(hi, int(math.ceil(math.exp(sub_log_hi))))
    if sub_lo > sub_hi:
        sub_hi = sub_lo
    return sub_lo, sub_hi


def _mx_align_k(K: int, klo: int, khi: int, block: int) -> Optional[int]:
    """Snap `K` onto a multiple of `block` inside [klo, khi], or None if
    the box admits no such K.

    A block-scaled (MX) GEMM carries one scale per `block` elements of the
    reduction dimension, so K MUST be a multiple of the block size -- there
    is no partial trailing scale. hipBLASLt does not round for you: an
    unaligned K simply matches no kernel, and hipblaslt-bench reports
    `NO solution found!` (and, on gfx1250v0, can go on to crash).
    A cell whose entire K range sits below one block (the TinyK tiers)
    therefore has NO valid MX shape at all, which is why this returns None
    rather than clamping -- the caller drops the sample instead of emitting
    a shape that cannot run."""
    if block <= 1:
        return K
    lo = ((klo + block - 1) // block) * block  # first multiple >= klo
    hi = (khi // block) * block  # last  multiple <= khi
    if lo > hi:
        return None  # box narrower than one block
    snapped = int(round(K / block)) * block
    return max(lo, min(hi, snapped))


def synthesize_cell_shapes(
    cell: str,
    n: int,
    rng: random.Random,
    bpe: int = 2,
    mt_align_fraction: float = 0.70,
    axis_bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    n_strata: int = 1,
    arch: Optional[str] = None,
    scale_a: int = 0,
    scale_b: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Generate up to `n` random shapes that fall in `cell` and pass
    `is_rotating512_safe`.

    `cell` is the BASE 96-grid label `"<Mt>|<Nt>|<Kt>|<Btier>"`. When
    the caller is synthesising for a sub-cell, it should pass
    `axis_bounds` with the tighter `(lo, hi)` already restricted per
    split rule (see `lib.subcells.restrict_axis_bounds`). Axes not in
    `axis_bounds` fall back to the tier's full range from lib/grid.

    `arch` (None -> gfx950 budget) and `scale_a` / `scale_b` (scale
    MODE ints; only MX block modes matter) are forwarded verbatim to the
    feasibility filter, so omitting them reproduces the previous pool
    exactly.

    `n_strata` (default 1 = uniform log-uniform across the whole box)
    enables CORNER-COVERAGE stratification: each of M / N / K is
    carved into `n_strata` log-equal sub-ranges, every (mi, ni, ki)
    bin gets ~`n / n_strata**3` shapes via log-uniform sampling
    within that sub-box, and bins that the rotating-512 OOM filter
    rejects (e.g. M-large bin under Bany) silently drop out of the
    pool. n_strata=4 = quartiles per axis = 64 sub-boxes.

    Returns fewer than `n` shapes when the (cell, axis_bounds) box is
    mostly empty (e.g. `Large|*|*|Bany`, or a sub-cell whose bounds
    collapsed to a single integer)."""
    # Elements per scale for a block-scaled config (32 for MXFP8/MXFP4
    # E8M0-block-32, 16 for the block-16 formats, 0/1 when unscaled).
    # Taken as the max of the two operands: if either side is block
    # scaled, K has to satisfy that side's grid.
    from .dat import mx_block_size_for_scale_mode as _mxb

    _mx_block = max(int(_mxb(scale_a) or 0), int(_mxb(scale_b) or 0))
    mt, nt, kt, bt = cell.split("|")
    m_lo, m_hi = TIER_RANGES_MN[mt]
    n_lo, n_hi = TIER_RANGES_MN[nt]
    k_lo, k_hi = TIER_RANGES_K[kt]
    if axis_bounds:
        if "M" in axis_bounds:
            m_lo = max(m_lo, axis_bounds["M"][0])
            m_hi = min(m_hi, axis_bounds["M"][1])
        if "N" in axis_bounds:
            n_lo = max(n_lo, axis_bounds["N"][0])
            n_hi = min(n_hi, axis_bounds["N"][1])
        if "K" in axis_bounds:
            k_lo = max(k_lo, axis_bounds["K"][0])
            k_hi = min(k_hi, axis_bounds["K"][1])
    if m_lo > m_hi or n_lo > n_hi or k_lo > k_hi:
        return []

    if n_strata <= 1:
        # Legacy path: pure log-uniform on the whole cell box.
        bins: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = [
            ((m_lo, m_hi), (n_lo, n_hi), (k_lo, k_hi))
        ]
        per_bin_n = n
    else:
        bins = [
            (
                _log_sub_range(m_lo, m_hi, i, n_strata),
                _log_sub_range(n_lo, n_hi, j, n_strata),
                _log_sub_range(k_lo, k_hi, k_, n_strata),
            )
            for i in range(n_strata)
            for j in range(n_strata)
            for k_ in range(n_strata)
        ]
        per_bin_n = max(1, n // len(bins))

    out: List[Tuple[int, int, int, int]] = []
    for m_sub, n_sub, k_sub in bins:
        mlo, mhi = m_sub
        nlo, nhi = n_sub
        klo, khi = k_sub
        bin_kept = 0
        attempts = 0
        while bin_kept < per_bin_n and attempts < per_bin_n * 20:
            attempts += 1
            M = _log_unif(rng, mlo, mhi)
            # int32-stride feasibility BY CONSTRUCTION: cap N's (then K's)
            # sampling ceiling so the per-matrix element counts (M*N, M*K, N*K)
            # never exceed _INT32_MAX. Without this, a huge-x-huge corner cell
            # (a sub-cell with both M and N in the tens of thousands) is mostly
            # infeasible, so reject-and-resample against is_rotating512_safe
            # BURNS the attempt budget and UNDER-FILLS the cell. Clamping fills
            # it with feasible shapes. Alignment only snaps DOWN, so it cannot
            # break the bound; is_rotating512_safe stays the HBM/FLOP backstop.
            nhi_eff = min(nhi, _INT32_MAX // max(1, M))
            if nhi_eff < nlo:
                continue  # this M too large for any N in box
            N = _log_unif(rng, nlo, nhi_eff)
            khi_eff = min(khi, _INT32_MAX // max(1, M, N))
            if khi_eff < klo:
                continue  # M,N too large for any K in box
            K = _log_unif(rng, klo, khi_eff)
            if rng.random() < mt_align_fraction:
                M = _mt_align(M, mlo, mhi)
            if rng.random() < mt_align_fraction:
                N = _mt_align(N, nlo, nhi_eff)
            # K: real winners are almost entirely 256-aligned; align K
            # at the same high rate as M/N (was a weak 0.20).
            if rng.random() < mt_align_fraction:
                K = _mt_align(K, klo, khi_eff)
            # MX block-scaling constrains K AFTER the macro-tile snap: the
            # scale operand has exactly K/block entries, so K must land on
            # the block grid. Unconditional (not probabilistic like the MT
            # snap above) because an unaligned K is not merely suboptimal
            # here -- it matches no kernel at all.
            if _mx_block > 1:
                K_mx = _mx_align_k(K, klo, khi_eff, _mx_block)
                if K_mx is None:
                    break  # this cell's K range is narrower than one
                    # scale block -- no MX shape can exist here
                K = K_mx
            if bt == "Bnone":
                if not is_known_faulting(
                    M, N, K, scale_a, scale_b, arch
                ) and is_rotating512_safe(
                    M, N, K, 1, bpe, arch=arch, scale_a=scale_a, scale_b=scale_b
                ):
                    out.append((M, N, K, 1))
                    bin_kept += 1
            else:
                # Couple B to the sampled dims: choose from the BANY_VALUES
                # that keep THIS (M,N,K) benchable. Large dims => only small
                # B is feasible; small dims => B up to 8192 -- exactly the
                # real "big-B => small-dims" envelope (gfx942 GridBased +
                # real-world workload traces). Avoids wasting attempts on
                # infeasible large-B-x-large-dim combos and gives the model
                # dense B coverage where it physically occurs.
                feasible_b = [
                    b
                    for b in BANY_VALUES
                    if (not is_known_faulting(M, N, K, scale_a, scale_b, arch))
                    and is_rotating512_safe(
                        M, N, K, b, bpe, arch=arch, scale_a=scale_a, scale_b=scale_b
                    )
                ]
                if feasible_b:
                    # Log-weighted choice (weight proportional to log2(b)):
                    # biases toward the large-B end (4096/8192) so it is
                    # represented per the design intent, instead of being
                    # under-sampled by a uniform pick over the many small-B
                    # values. Deterministic given `rng` (rng.choices draws
                    # from the same seeded stream).
                    b_weights = [math.log2(b) for b in feasible_b]
                    chosen_b = rng.choices(feasible_b, weights=b_weights, k=1)[0]
                    out.append((M, N, K, chosen_b))
                    bin_kept += 1
        if len(out) >= n:
            break
    # Bins are filled low-stratum-first (M-low octants come first), so
    # callers that OVER-generate then keep the first `target` (e.g.
    # stage01 sample_shapes requests target*3 for history-dedup
    # headroom) would otherwise discard every high-magnitude octant --
    # silently defeating the stratification. Shuffle so any take-first
    # truncation stays balanced across strata.
    rng.shuffle(out)
    return out[:n]


# ── shape hashing (history dedup) ────────────────────────────────────────────


def shape_hash(
    m: int,
    n: int,
    k: int,
    b: int,
    a_type: Optional[str] = None,
    b_type: Optional[str] = None,
    compute_type: Optional[str] = None,
    transA: Optional[str] = None,
    transB: Optional[str] = None,
) -> str:
    """Stable shape identifier used for cross-round dedup.

    Default (single-config) behavior is UNCHANGED: with only
    (M, N, K, batch) supplied, the hash is the legacy
    ``f"{m}_{n}_{k}_{b}"`` -- correct within one pipeline run, where the
    bench config (dtype/layout/iters) is constant across rounds.

    Cross-config callers (different dtype/layout, SAME MNKB -- which
    would otherwise collide) MUST pass the bench-config fields so
    distinct configs hash distinctly. The extra fields mirror stage05's
    ``gemm_key`` dtype/layout subset (a_type, b_type, compute_type,
    transA, transB) and are namespaced onto the key ONLY when at least
    one is provided, so every existing 4-arg call site stays
    byte-identical."""
    base = f"{m}_{n}_{k}_{b}"
    cfg = (a_type, b_type, compute_type, transA, transB)
    if any(v is not None for v in cfg):
        suffix = "_".join("" if v is None else str(v) for v in cfg)
        return f"{base}|{suffix}"
    return base
