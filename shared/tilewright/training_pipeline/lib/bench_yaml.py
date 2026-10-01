# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt-bench yaml writer and parser.

Each line in the bench yaml describes ONE GEMM with all bench knobs
inline (`iters`, `cold_iters`, `rotating`, `skip_slow_solution_ratio`,
etc.). pipeline_v3 keeps this format so we can re-use the same
benchmark_load_balancer.py script unmodified.

This module is intentionally line-oriented (no full yaml load) so even
a very large shape file stays cheap to process.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .shapes import Shape


# ── stride defaults: derive from M/N/K + layout ──────────────────────────────


def strides_for(
    transA: str, transB: str, m: int, n: int, k: int
) -> Tuple[int, int, int, int]:
    """Default leading dimensions for hipblaslt's M-major matrices."""
    if transA == "N":  # A is M x K, stride along M
        lda = m
    else:  # A^T is K x M, stride along K
        lda = k
    if transB == "N":  # B is K x N, stride along K
        ldb = k
    else:  # B^T is N x K, stride along N
        ldb = n
    ldc = m
    ldd = m
    return lda, ldb, ldc, ldd


# ── render one bench line ────────────────────────────────────────────────────

DEFAULT_LINE_KNOBS = {
    "alpha": 1,
    "beta": 0,
    "initialization": "trig_float",
    "iters": 50,
    "cold_iters": 50,
    "use_gpu_timer": 1,
    "flush": "true",
    "rotating": 512,
    "print_kernel_info": 1,
    "requested_solution_num": -1,
}

# ── adaptive timing ──────────────────────────────────────────────────────────
#
# hipblaslt-bench can time a kernel ADAPTIVELY instead of running a fixed
# iteration count: it warms up, collects timed samples, and stops as soon as
# the mean's relative standard error falls under `noise_threshold` (status
# `converged`), or when a robust-spread plateau is detected (`stable`), or at
# the ceiling (`noisy`). That removes the entire reason stage02 existed --
# stage02 only ever measured a shape's `us` in order to BACK-CALCULATE an
# iteration count. Let the bench do it per kernel instead, where it has the
# actual variance in hand.
#
# CRITICAL -- the YAML keys are the `Arguments` STRUCT field names
# (hipblaslt_arguments.hpp's OPER list), NOT the `--adaptive_*` command-line
# spellings. `adaptive_warmup_time:` in a yaml is silently IGNORED and you
# quietly inherit the 500 ms `measure_time` default, hundreds of times the budget
# this pipeline wants per kernel. The correct keys are exactly these:
ADAPTIVE_KEYS = (
    "warmup_time",
    "sample_time",
    "measure_time",
    "max_measure_time",
    "min_iters",
    "max_iters",
    "noise_threshold",
    "stability_threshold",
    "stability_window",
    "stability_interval",
)

# `adaptive` is mutually exclusive with iters/cold_iters -- the bench rejects
# the combination ("Not compatible with --iters / --cold_iters"), so the
# renderer DROPS them when adaptive is on rather than emitting both.
_ITER_KEYS = ("iters", "cold_iters")


def adaptive_cfg(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the `bench.adaptive` block when it is enabled, else None.

    Shape:
        bench:
          adaptive:
            enabled: true
            measure_time: 2.0      # ms floor per kernel
            ...
    Any key in ADAPTIVE_KEYS present in the block is passed through verbatim;
    unknown keys are dropped with no silent renaming, because a typo'd knob
    that reaches the bench as an unknown yaml field is ignored there too and
    the failure mode is invisible."""
    blk = (config.get("bench") or {}).get("adaptive") or {}
    if not blk or not blk.get("enabled", False):
        return None
    return {k: blk[k] for k in ADAPTIVE_KEYS if k in blk}


def _default_scale_for_dtype(dtype: str) -> int:
    """Default scaleA / scaleB mode for an operand dtype.

    hipblaslt-bench's `scaleA` (and `scaleB`) is an integer that maps
    to a `hipblaslt_scaling_format` (see hipBLASLt clients/.../client.cpp):
      0 = none    (no per-tensor scale; default for non-quantized dtypes)
      1 = Scalar  (per-tensor scalar scale; matches the F8BS Origami
                   UserArgs whose ProblemType has `UseScaleAB: Scalar`)
      2 = Vector  (per-tensor vector scale)
      3+= Block / MX formats

    For the F8BS_TN library path (UseScaleAB=Scalar) the correct mode
    is **1** (Scalar) -- NOT 2 (which is Vector and selects a different
    kernel family). Callers OVERRIDE this via `config.hipblaslt.scaleA`
    / `scaleB`; the F8BS config pins scaleA=scaleB=1 (Scalar, value 1.0).
    """
    return 1 if "f8_r" in dtype else 0


# ── rotating buffer sizing ───────────────────────────────────────────────────
#
# `rotating` is a rotating-buffer budget in MiB. The bench divides it by the
# problem's footprint to get a BLOCK COUNT and allocates that many copies, so
# it can cycle through cold copies and defeat cache reuse between iterations.
#
# A FIXED 512 MiB budget is pathological for small problems. For a 2x3x32
# MXFP8 GEMM (a perfectly ordinary Tiny|Tiny|TinyK grid cell) the bench
# reports:
#     Rotating buffer 512 MiB. Needed Size: 0 MiB. Needed block count: 2130441
# i.e. it tries to allocate 2.1 MILLION copies of a 252-byte problem, and
# that allocation alone can dominate a probe/bench pass over small shapes.
#
# Size the budget to the PROBLEM instead, so the block count is bounded:
#     rotating_bytes = footprint_bytes * target_blocks
# capped at `rotating_mb` and floored at 0. Below 1 MiB we emit 0 (rotation
# off): a problem that small lives in cache no matter how many copies exist,
# so rotating buffers buy nothing and cost allocation time.
_MIB = 1024 * 1024


def _footprint_bytes(shape: Shape, hl: Dict[str, Any]) -> int:
    """Approximate per-instance device footprint: A + B + C + D (+ MX scales)."""
    from .shapes import _BPE  # bytes/element by dtype

    m, n, k, b = shape.m, shape.n, shape.k, max(1, shape.batch)
    bpe_a = _BPE.get(hl.get("a_type", "bf16_r"), 2)
    bpe_b = _BPE.get(hl.get("b_type", "bf16_r"), 2)
    bpe_c = _BPE.get(hl.get("c_type", "bf16_r"), 2)
    bpe_d = _BPE.get(hl.get("d_type", "bf16_r"), 2)
    total = b * (m * k * bpe_a + k * n * bpe_b + m * n * bpe_c + m * n * bpe_d)
    # MX block scales: one e8m0 byte per `block` elements of A and of B.
    try:
        from .dat import mx_block_size_for_scale_mode as _mxb

        blk = max(
            int(_mxb(hl.get("scaleA", 0)) or 0), int(_mxb(hl.get("scaleB", 0)) or 0)
        )
    except Exception:
        blk = 0
    if blk > 0:
        total += b * ((m * k) // blk + (k * n) // blk)
    return int(total)


def rotating_mb_for(shape: Shape, config: Dict[str, Any]) -> int:
    """Rotating-buffer MiB for one shape, bounded so the bench's derived block
    count stays near `bench.rotating_target_blocks`."""
    bench = config.get("bench") or {}
    cap = int(bench.get("rotating_mb", 512))
    target = int(bench.get("rotating_target_blocks", 16))
    if cap <= 0 or target <= 0:
        return 0
    want = _footprint_bytes(shape, config["hipblaslt"]) * target
    # FLOOR AT 1 MiB -- never 0. Emitting 0 disables rotation entirely, and that
    # is NOT harmless for small problems: with a single buffer, candidates that
    # run back-to-back on the same data read it warm from cache, so whichever
    # kernel is measured LAST looks fastest. Byte-identical kernel variants (e.g.
    # the TemporalHint variants of one signature) then show an in-sweep spread
    # far larger than the modest one they show when measured in ISOLATION; the
    # excess is pure measurement order, and an earlier revision of this
    # function caused it by returning 0 for the smallest shapes.
    #
    # 1 MiB is cheap even for a 252-byte problem (it just means many small
    # copies) and restores the cache-defeating the fixed 512 MiB budget gave,
    # without the 2,130,441-block explosion that motivated sizing this at all.
    want_mib = -(-want // _MIB)  # ceil(bytes -> MiB)
    return int(min(cap, max(1, want_mib)))  # clamp to [1, cap] MiB


def render_bench_line(
    shape: Shape, config: Dict[str, Any], knobs: Optional[Dict[str, Any]] = None
) -> str:
    """Return a single hipblaslt-bench YAML line for `shape` using the
    dtype/layout from `config.hipblaslt` and the additional knobs
    (iters / cold_iters / rotating / skip_slow_solution_ratio) merged
    over the defaults.

    stride_{a,b,c,d} are emitted as 0 (the bench tool then computes
    the natural defaults internally; we never use non-default strides
    in pipeline_v3).

    scaleA / scaleB are read from `config.hipblaslt.scaleA` /
    `scaleB` when set, otherwise defaulted by operand dtype: 2 for
    F8 (mode = "explicit per-tensor scale tensor"), 0 for everything
    else.
    """
    hl = config["hipblaslt"]
    transA, transB = hl["transA"], hl["transB"]
    a_t = hl["a_type"]
    b_t = hl["b_type"]
    c_t = hl["c_type"]
    d_t = hl["d_type"]
    scale_t = hl["scale_type"]
    compute_t = hl["compute_type"]
    m, n, k, b = shape.m, shape.n, shape.k, shape.batch
    lda, ldb, ldc, ldd = strides_for(transA, transB, m, n, k)
    scale_a = int(hl["scaleA"]) if "scaleA" in hl else _default_scale_for_dtype(a_t)
    scale_b = int(hl["scaleB"]) if "scaleB" in hl else _default_scale_for_dtype(b_t)
    k_dict = dict(DEFAULT_LINE_KNOBS)
    if knobs:
        k_dict.update(knobs)
    # skip_slow_solution_ratio is the candidate-pruning screen, and it is
    # INDEPENDENT of how a kept candidate is timed: with `adaptive` the client
    # keeps exactly one cold call whenever this ratio is non-zero, purely to
    # run the screen (testing_matmul.hpp ~:5535). A solution is dropped when
    #     warm_up * ratio > best_warm_up_so_far
    # i.e. ratio=0.5 skips anything slower than 2x the best seen. stage02 used
    # to derive this per shape from probe_us; with stage02 gone it has to come
    # from config, or it defaults to 0.0 = NO PRUNING and every one of the
    # library's candidates gets fully timed.
    _ssr = (config.get("bench") or {}).get("skip_slow_solution_ratio")
    if _ssr is not None:
        k_dict["skip_slow_solution_ratio"] = _ssr
    # Shape-aware rotating budget (see rotating_mb_for). Overridable per call
    # via `knobs`, which is how stage08 pins its own value.
    if not (knobs and "rotating" in knobs):
        k_dict["rotating"] = rotating_mb_for(shape, config)
    ad = adaptive_cfg(config)
    if ad is not None:
        # Adaptive replaces the fixed iteration count; emitting both is a
        # hard error in the bench, so the iter knobs come out here even if a
        # caller (or a stale stage02 stamp) tried to set them.
        for k_ in _ITER_KEYS:
            k_dict.pop(k_, None)
        k_dict["adaptive"] = "true"
        k_dict.update(ad)
    kv = ", ".join(f"{k_}: {v}" for k_, v in k_dict.items())
    return (
        f"- {{function: matmul, transA: {transA}, transB: {transB}, "
        f"a_type: {a_t}, b_type: {b_t}, c_type: {c_t}, d_type: {d_t}, "
        f"scale_type: {scale_t}, compute_type: {compute_t}, "
        f"scaleA: {scale_a}, scaleB: {scale_b}, "
        f"M: {m}, N: {n}, K: {k}, "
        f"ldc: {ldc}, ldd: {ldd}, lda: {lda}, ldb: {ldb}, "
        f"stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, "
        f"batch_count: {b}, {kv}}}"
    )


def write_bench_yaml(
    shapes: Iterable[Shape],
    config: Dict[str, Any],
    out_path: Path,
    knobs: Optional[Dict[str, Any]] = None,
    per_shape_knobs: Optional[Dict[Tuple[int, int, int, int], Dict[str, Any]]] = None,
) -> int:
    """Write one bench line per shape. If `per_shape_knobs` is provided,
    its `(M, N, K, B) -> dict` override `knobs` for that specific shape
    (used by stage02 to stamp probe-calibrated `iters` /
    `cold_iters` / `skip_slow_solution_ratio` per shape)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as f:
        for s in shapes:
            merged = dict(knobs or {})
            if per_shape_knobs:
                m = per_shape_knobs.get((s.m, s.n, s.k, s.batch))
                if m:
                    merged.update(m)
            f.write(render_bench_line(s, config, merged) + "\n")
            n_written += 1
    return n_written


# ── line-oriented parser (no full yaml load) ─────────────────────────────────

_MNKB_RE = re.compile(r"M:\s*(\d+).*?N:\s*(\d+).*?K:\s*(\d+).*?batch_count:\s*(\d+)")


def parse_shapes_yaml(path: Path) -> List[Tuple[int, int, int, int]]:
    """Pull (M, N, K, batch) from a hipblaslt-bench yaml without a
    yaml.safe_load (regex per line). Returns the list in file order
    (which IS the bench-execution order)."""
    out: List[Tuple[int, int, int, int]] = []
    with Path(path).open() as f:
        for line in f:
            m = _MNKB_RE.search(line)
            if m:
                out.append(
                    (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                )
    return out


_KV_INT_RE = {
    k: re.compile(rf"(?<!\w){re.escape(k)}:\s*(\d+)")
    for k in ("iters", "cold_iters", "rotating", "M", "N", "K", "batch_count")
}
_KV_FLOAT_RE = {
    k: re.compile(rf"(?<!\w){re.escape(k)}:\s*([\d.]+)")
    for k in ("skip_slow_solution_ratio",)
}


def parse_knobs(line: str) -> Dict[str, Any]:
    """Parse the int/float per-line knobs from a single bench yaml
    line. Returns a dict containing keys present in the line."""
    out: Dict[str, Any] = {}
    for k, pat in _KV_INT_RE.items():
        m = pat.search(line)
        if m:
            out[k] = int(m.group(1))
    for k, pat in _KV_FLOAT_RE.items():
        m = pat.search(line)
        if m:
            out[k] = float(m.group(1))
    return out
