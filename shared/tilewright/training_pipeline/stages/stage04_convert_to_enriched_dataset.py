#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single end-to-end script: hipblaslt-bench block logs -> 10k-row chunked
CSV folder of enriched training rows.

Pipeline (one streaming pass per log file):
  1. Parse [N]: blocks (tested) and "Skip solution: K" lines (skipped).
     Drop the Winner: re-print row; set is_winner=True on the min-us tested
     row of each GEMM.
  2. Tested row enrichment:
        a) sol_idx_global = log's --Solution index
        b) sol_idx_local  = MatchTable[sol_idx_global][1]
        c) kernel params  = library .dat solution whose `name` exactly equals
                            the log's --Solution name. The .dat is the
                            authoritative built form -- its `sizeMapping`
                            holds the runtime values (CUOccupancy, mt, mi,
                            nonTemporalA, etc.). MatchTable + yaml are used
                            only to verify (yaml local_idx == .dat
                            libraryLogicIndex for the same kernel).
  3. Skip row enrichment via origami.rank_configs():
        a) For each problem we look at the tested rows' sol_idx_global -> yaml
           path map (via MatchTable) to figure out *which* source yaml variants
           the bench actually pulled configs from for this problem (usually a
           single Origami variant). We then build the config list passed to
           rank_configs() from just that variant's .dat solutions, which makes
           origami's rank order line up with the bench's `Is supported` set
           much better than feeding it the union of all variants.
        b) For each Skip solution: K row, configs from rank_configs()[K]
           supply mt/mi/occupancy/etc. sol_idx_global / sol_idx_local are
           left blank since the log does not print them.
  4. Stream rows into chunk_NNNN.csv files, 10000 rows per chunk by default.

Kernel params (mt/mi/cache_hints_a/b/grvw/gwvw) are read from the SOURCE
Tensile solution YAML by (yaml_path, local_idx) via the MatchTable -- the
authoritative, collision-free source. Occupancy is the one field taken from
the .dat (keyed by global solution index), mirroring hipBLASLt's runtime
SizeMapping (the YAML stores CUOccupancy=-1).

Output schema (kernel-param tail):
  ... is_winner, rank, sol_idx_global, sol_idx_local, <bench data fields> ...
  mt_m, mt_n, mt_k,
  mi_m, mi_n, mi_k,
  occupancy,
  cache_hints_a, cache_hints_b,
  grvw_a, grvw_b, gwvw_d

Bench-row fields that are dropped (the bench's print of these is not
authoritative): scaleC, scaleD, amaxD, swizzle_a, swizzle_b,
activation_type, bias_vector, bias_type, aux_type.

Kernel-param fields that are dropped:
  - The former config_ml_t columns (cache_hints_c/d/e, prefetch_global_read /
    prefetch_local_read, lds_read_vector_width, local_split_u) and the LDS-pad
    columns (lds_pad_a/b, lds_block_size_per_pad_a/b). Origami's analytical
    scorer never read them and the ML feature group consuming them was
    removed: they were populated only on un-benched skip rows and empty on
    every benched candidate -- a dead, train/serve-skew-prone set.
  - swizzle_tensor_a, swizzle_tensor_b: problem-type-level, constant per combo.
  - mi_lanes (MatrixInstruction[3]): always 1 in modern AMD MI kernels.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── terminal UI ───────────────────────────────────────────────────────────────


class C:
    """ANSI color codes; auto-disabled when stdout is not a TTY."""

    _on = sys.stdout.isatty()
    HEADER = "\033[95m" if _on else ""
    BLUE = "\033[94m" if _on else ""
    CYAN = "\033[96m" if _on else ""
    GREEN = "\033[92m" if _on else ""
    YELLOW = "\033[93m" if _on else ""
    RED = "\033[91m" if _on else ""
    GREY = "\033[90m" if _on else ""
    ENDC = "\033[0m" if _on else ""
    BOLD = "\033[1m" if _on else ""


def _banner(title: str, color: str = "") -> None:
    color = color or C.BOLD
    bar = "=" * 80
    print(f"\n{color}{bar}{C.ENDC}", flush=True)
    print(f"{color}  {title}{C.ENDC}", flush=True)
    print(f"{color}{bar}{C.ENDC}", flush=True)


def _info(tag: str, msg: str) -> None:
    print(f"{C.CYAN}[{tag}]{C.ENDC} {msg}", flush=True)


def _ok(tag: str, msg: str) -> None:
    print(f"{C.GREEN}[{tag}]{C.ENDC} {msg}", flush=True)


def _warn(tag: str, msg: str) -> None:
    print(f"{C.YELLOW}[{tag}]{C.ENDC} {msg}", flush=True)


def _err(tag: str, msg: str) -> None:
    print(f"{C.RED}[{tag}]{C.ENDC} {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return " --.-%"
    return f"{100.0 * num / den:5.1f}%"


try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")
try:
    import msgpack
except ImportError:
    raise SystemExit("msgpack required")

# Self-contained lib imports -- single source of truth for the .dat
# dtype patterns and the kernel signature.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_V3 = os.path.dirname(_THIS_DIR)
if _PIPELINE_V3 not in sys.path:
    sys.path.insert(0, _PIPELINE_V3)
from lib import dat  # noqa: E402
from lib.arch import canonical_arch as _canonical_arch  # noqa: E402
from lib.dat import (
    mx_block_size_for_scale_mode,  # noqa: E402
    parse_dtype_layout as _parse_dtype_layout,
    parse_scale_mode as _parse_scale_mode,
)
from lib.origami import _DTYPE_TO_ORIGAMI  # noqa: E402
from lib.origami import _rank_configs_compat  # noqa: E402

try:
    import origami

    _ORIGAMI_OK = True
except ImportError:
    _ORIGAMI_OK = False
    origami = None  # type: ignore


# ── log regexes ───────────────────────────────────────────────────────────────

SOL_IDX_RE = re.compile(r"--Solution index:\s+(\d+)")
SOL_NAME_RE = re.compile(r"--Solution name:\s+(\S+)")
HEADER_RE = re.compile(r"^\[(\d+)\]:")
SKIP_RE = re.compile(
    r"Skip solution:\s+(\d+)\s*\("
    r"\s*best warm-up\s*=\s*[\d.]+\s*us\s*,\s*"
    r"warm-up\s*=\s*([\d.]+)\s*us"
)
# The bench now prints the skipped solution's runtime global index directly, so
# its identity no longer has to be replayed through origami's ranking. Absent
# in logs from benches built before that change, which still take the rank path.
SKIP_IDX_RE = re.compile(r"solution index\s*=\s*(\d+)")
# A handful of tolerated mismatches before we abort: a correct index
# resolution yields exactly zero.
_MT_MISMATCH_FATAL_AFTER = 10
_MT_NAME_RE = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")

# LDS pad parsing and kernel signature
# all live in lib/dat.py -- imported at the top of this file. The
# legacy module-local copies (parse_lds_from_kernel_name,
# _origami_lookup_signature, _config_t_signature) have been removed
# in favour of the single source of truth.

# Full 39 fields hipblaslt-bench writes per data row, in log order.
_RAW_DATA_FIELDS = [
    "transA",
    "transB",
    "grouped_gemm",
    "batch_count",
    "m",
    "n",
    "k",
    "alpha",
    "lda",
    "stride_a",
    "beta",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "ldd",
    "stride_d",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
    "scaleC",
    "scaleD",
    "amaxD",
    "swizzle_a",
    "swizzle_b",
    "activation_type",
    "bias_vector",
    "bias_type",
    "aux_type",
    "rotating_buffer",
    "flush",
    "use_gpu_timer",
    "hipblaslt-Gflops",
    "hipblaslt-GB/s",
    "us",
]
# Columns to keep from the bench data row.
DATA_FIELDS = [
    "transA",
    "transB",
    "grouped_gemm",
    "batch_count",
    "m",
    "n",
    "k",
    "alpha",
    "lda",
    "stride_a",
    "beta",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "ldd",
    "stride_d",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
    "rotating_buffer",
    "flush",
    "use_gpu_timer",
    "hipblaslt-Gflops",
    "hipblaslt-GB/s",
    "us",
]
PROBLEM_FIELDS = [
    f for f in DATA_FIELDS if f not in ("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
]

CONFIG_FIELDS = [
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
]
# NOTE: the former config_ml_t columns (cache_hints_c/d/e,
# prefetch_global_read/local_read, lds_read_vector_width, local_split_u) and
# the LDS-pad columns (lds_pad_a/b, lds_block_size_per_pad_a/b) were removed.
# Origami's analytical scorer never read them and the ML recommender's
# corresponding features were dropped (see lib/features.py): they were
# populated only on un-benched skip rows and empty on every benched candidate,
# i.e. a dead, train/serve-skew-prone set. The kept fields are exactly the
# inputs the `tile` feature group + occupancy consume.
OUT_FIELDS = (
    ["is_winner", "is_skip", "rank", "sol_idx_global", "sol_idx_local"]
    + DATA_FIELDS
    + CONFIG_FIELDS
)
EMPTY_CFG = {k: "" for k in CONFIG_FIELDS}


# ── MatchTable ────────────────────────────────────────────────────────────────

_MT_LINE_RE = re.compile(r"^(\d+)\s*:\s*\[\s*([^,]+?)\s*,\s*(\d+)\s*\]\s*$")
_MT_OPEN_RE = re.compile(r"^(\d+)\s*:\s*\[\s*([^,]+?)\s*,\s*$")
_MT_CLOSE_RE = re.compile(r"^\s*(\d+)\s*\]\s*$")


def parse_match_table(path: str) -> Dict[int, Tuple[str, int]]:
    table: Dict[int, Tuple[str, int]] = {}
    pending_key: Optional[int] = None
    pending_path: Optional[str] = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("---") or not line.strip() or line.strip() == "...":
                continue
            m1 = _MT_LINE_RE.match(line)
            if m1:
                table[int(m1.group(1))] = (m1.group(2), int(m1.group(3)))
                continue
            m2 = _MT_OPEN_RE.match(line)
            if m2:
                pending_key = int(m2.group(1))
                pending_path = m2.group(2)
                continue
            m3 = _MT_CLOSE_RE.match(line)
            if m3 and pending_key is not None and pending_path is not None:
                table[pending_key] = (pending_path, int(m3.group(1)))
                pending_key = None
                pending_path = None
    return table


# ── library .dat ──────────────────────────────────────────────────────────────

_YAML_SOLS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
# Prefer the libyaml C loader -- the Tensile solution YAMLs run to tens of
# thousands of lines, where the pure-Python loader is much slower.
_HAVE_CSAFE_LOADER = hasattr(yaml, "CSafeLoader")


def _yaml_sols(yaml_path: str) -> List[Dict[str, Any]]:
    if yaml_path in _YAML_SOLS_CACHE:
        return _YAML_SOLS_CACHE[yaml_path]
    try:
        with open(yaml_path) as f:
            # Each branch names its safe loader explicitly: the bandit B506
            # gate cannot see through a loader held in a variable and treats
            # it as arbitrary-object deserialization.
            if _HAVE_CSAFE_LOADER:
                data = yaml.load(f, Loader=yaml.CSafeLoader)
            else:
                data = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception:
        _YAML_SOLS_CACHE[yaml_path] = []
        return []
    # Two on-disk shapes:
    #
    #  * LEGACY list form -- the classic Tensile logic yaml is a LIST of
    #    documents and one element is the solution list. Handled below.
    #
    #  * MAPPING form -- the gfx1250 Origami (LibraryType: Prediction) yamls are
    #    a top-level MAPPING with a `Solutions:` key:
    #        MinimumRequiredVersion / ScheduleName / ArchitectureName / CUCount
    #        DeviceNames / ProblemType / DefaultSolution / Solutions
    #    A list-only `isinstance(data, list)` gate rejects these outright, so
    #    EVERY Origami solution would come back as an enrichment miss -- i.e.
    #    nearly every tested row, the entire candidate pool this pipeline
    #    exists to rank. GridBased yamls are the legacy form, which is why
    #    they resolve under such a gate and Origami ones do not.
    if isinstance(data, dict):
        sols = data.get("Solutions")
        if isinstance(sols, list) and sols and isinstance(sols[0], dict):
            _YAML_SOLS_CACHE[yaml_path] = sols
            return sols
        _YAML_SOLS_CACHE[yaml_path] = []
        return []
    if not isinstance(data, list):
        _YAML_SOLS_CACHE[yaml_path] = []
        return []
    for elem in data:
        if (
            isinstance(elem, list)
            and elem
            and isinstance(elem[0], dict)
            and "MatrixInstruction" in elem[0]
        ):
            _YAML_SOLS_CACHE[yaml_path] = elem
            return elem
    _YAML_SOLS_CACHE[yaml_path] = []
    return []


def _yaml_kernel_cfg(yaml_path: str, local_idx: int) -> Optional[Dict[str, Any]]:
    """Read a kernel's structural params straight from the source-of-truth
    Tensile solution YAML, indexed by the MatchTable's local index.

    This is the AUTHORITATIVE param source. The prior path resolved params
    from a .dat solution looked up by kernel `name`, but Custom (and other)
    kernel names are DUPLICATED across the base / Origami_nta4 / Origami_ntb4
    libraries -- which live in different YAML files and differ in
    NonTemporalA/B. Name-keyed lookup therefore returned the wrong variant's
    cache_hints_a/b for any colliding kernel. Indexing the correct YAML by
    (yaml_path, local_idx) -- exactly the (yaml, local) pair the MatchTable
    maps each global solution index to -- is collision-free.

    Occupancy is intentionally NOT read here: the YAML stores `CUOccupancy: -1`
    (the 'unset' sentinel), whereas hipBLASLt's runtime SizeMapping carries the
    real value. The caller fills `occupancy` from the .dat (see
    `build_dat_occupancy_index`). Returns None when the solution can't be
    located so the caller can record a miss."""
    sols = _yaml_sols(yaml_path)
    if not sols or local_idx < 0 or local_idx >= len(sols):
        return None
    s = sols[local_idx]
    mi = s.get("MatrixInstruction") or [0, 0, 0, 0]
    while len(mi) < 3:
        mi = list(mi) + [0]

    def _gi(key: str, default: int) -> int:
        v = s.get(key)
        try:
            return int(v) if v is not None else int(default)
        except (TypeError, ValueError):
            return int(default)

    return {
        "mt_m": _gi("MacroTile0", _gi("MacroTileA", 0)),
        "mt_n": _gi("MacroTile1", _gi("MacroTileB", 0)),
        "mt_k": _gi("DepthU", _gi("_DepthU", 0)),
        "mi_m": _gi("MatrixInstM", int(mi[0])),
        "mi_n": _gi("MatrixInstN", int(mi[1])),
        "mi_k": _gi("MatrixInstK", int(mi[2])),
        "cache_hints_a": _gi("NonTemporalA", 0),
        "cache_hints_b": _gi("NonTemporalB", 0),
        "grvw_a": _gi("GlobalReadVectorWidthA", 1),
        "grvw_b": _gi("GlobalReadVectorWidthB", 1),
        # gwvw_d == YAML `StoreVectorWidth` (NOT `GlobalWriteVectorWidth`):
        # StoreVectorWidth matches the runtime's derived sizeMapping.gwvwD on
        # nearly every kernel, far more often than GlobalWriteVectorWidth does
        # (verified by tests/test_yaml_vs_origami_params.py). Clamp to >=1
        # (gwvwD is always a positive resolved width; some YAMLs store the -1
        # 'auto' sentinel).
        "gwvw_d": max(_gi("StoreVectorWidth", 1), 1),
    }


def build_dat_config_index(dat_dir: str, quiet: bool) -> Dict[int, Dict[str, Any]]:
    """sol_idx_global -> FULL kernel config, read from the .dat.

    The .dat's top-level `index` IS the runtime solution index that
    hipblaslt-bench prints as `--Solution index:` (verified against raw bench
    logs: every index agrees, none missing).

    This replaces the MatchTable lookup, which can address a DIFFERENT index
    space. When MatchTable.yaml's numbering is offset from the runtime's,
    every enriched row carries the tile geometry of the kernel a few slots
    away. Timings stay correct; only the kernel identity attached to them is
    wrong, on nearly every tested row. A merely printed `MT mismatches: ...`
    count is easy to overlook, so this is now FATAL (see _mt_mismatches).
    """
    out: Dict[int, Dict[str, Any]] = {}
    n_files = 0
    t0 = time.time()
    for fname in sorted(os.listdir(dat_dir)):
        if not dat.is_tensile_contraction_logic(fname):
            continue
        data = dat.read_tensile_logic(os.path.join(dat_dir, fname))
        if data is None:
            if not quiet:
                _warn("load", f"skip {fname}: unreadable Tensile logic")
            continue
        n_files += 1
        for sol in data.get("solutions", []):
            gi = sol.get("index")
            if gi is None:
                continue
            out[int(gi)] = dat.kernel_dat_info(sol)
    if not quiet:
        _info(
            "load",
            f"dat config index: {len(out)} solutions from "
            f"{n_files} .dat files in {time.time()-t0:.1f}s",
        )
    return out


def build_dat_occupancy_index(dat_dir: str, quiet: bool) -> Dict[int, int]:
    """Walk all TensileLibrary*.dat files; map sol_idx_global (the solution's
    top-level `index`, which IS the hipblaslt-bench MatchTable global key) ->
    CUOccupancy (clamped >= 1).

    Occupancy is the ONLY field sourced from the .dat: the source YAML stores
    the CUOccupancy=-1 sentinel, while hipBLASLt's runtime SizeMapping carries
    the real value. Every other kernel param (incl. gwvw_d == StoreVectorWidth)
    comes from the YAML by index. Keyed by the GLOBAL index (unique,
    collision-free) -- names collide across the NonTemporal library variants
    and resolved to the wrong variant under the old name index."""
    out: Dict[int, int] = {}
    n_files = 0
    t0 = time.time()
    for fname in sorted(os.listdir(dat_dir)):
        if not dat.is_tensile_contraction_logic(fname):
            continue
        # `read_tensile_logic` inflates `.dat.zlib` transparently. Reading
        # these with a bare msgpack.unpack throws on every compressed file
        # and leaves the occupancy index silently EMPTY on trees (gfx1250v0)
        # that ship only the compressed spelling.
        data = dat.read_tensile_logic(os.path.join(dat_dir, fname))
        if data is None:
            if not quiet:
                _warn("load", f"skip {fname}: unreadable Tensile logic")
            continue
        n_files += 1
        for sol in data.get("solutions", []):
            gi = sol.get("index")
            if gi is None:
                continue
            sm = sol.get("sizeMapping", {}) or {}
            out[int(gi)] = max(int(sm.get("CUOccupancy", 1) or 1), 1)
    if not quiet:
        _ok(
            "load",
            f"dat occupancy index: {_fmt_int(len(out))} solutions "
            f"from {n_files} .dat files ({_fmt_dur(time.time()-t0)})",
        )
    return out


# ── origami (for skip rows) ───────────────────────────────────────────────────
#
# .dat dtype/layout patterns and the kernel signature are imported
# from lib/dat.py at the top of this file (single source of truth).
# This block kept only the origami config_t builder, which uses
# the same patterns.

# `_DTYPE_TO_ORIGAMI` is imported from lib/origami.py at the top of this
# file. It used to be re-declared here; the two copies drifted (this one
# never got the 'f8_r' key), so f8 problems were silently built as bf16.


def _origami_config_from_dat_solution(sm: Dict[str, Any], cfg_position: int) -> Any:
    """Build an origami.config_t from a .dat sizeMapping. `cfg_position` is
    the index into our per-(dtype/layout) cfg_list -- we stash it on
    config_t.index (exposed by the binding) so that after origami's
    rank_configs returns a ranked permutation, we can recover the original
    .dat position via `ranked[K].config.index`.

    Note: pipeline_v3 drops `workGroupMapping` from the feature set
    entirely; we still set `c.workgroup_mapping = 0` because origami's
    predicate evaluation reads the field. `workspace_size` is hardcoded
    to UINT64_MAX in hipblaslt's PredictionLibrary.hpp, so it is NOT a
    kernel feature -- we just mirror that hardcode here for origami's
    predicate evaluation, but it never reaches training."""
    c = origami.config_t()
    c.index = cfg_position
    mt = sm.get("macroTile") or [0, 0, 0]
    c.mt = origami.dim3_t(int(mt[0]), int(mt[1]), int(sm.get("depthU", 0) or 0))
    mi = sm.get("matrixInstruction") or [0, 0, 0, 0]
    if not mi or (mi[0] == 0 and mi[1] == 0 and mi[2] == 0):
        c.mi = origami.dim3_t(1, 1, 64)
    else:
        c.mi = origami.dim3_t(int(mi[0]), int(mi[1]), int(mi[2]))
    cms = sm.get("customMainLoopScheduling", 0)
    try:
        c.hand_optimized_main_loop = int(cms) > 0
    except (TypeError, ValueError):
        c.hand_optimized_main_loop = False
    c.occupancy = max(int(sm.get("CUOccupancy", 1) or 1), 1)
    c.cache_hints_a = int(sm.get("nonTemporalA", 0) or 0)
    c.cache_hints_b = int(sm.get("nonTemporalB", 0) or 0)
    c.workspace_size = (2**64) - 1
    c.workspace_size_per_elem_c = (2**64) - 1
    c.grvw_a = int(sm.get("grvwA", 1) or 1)
    c.grvw_b = int(sm.get("grvwB", 1) or 1)
    c.gwvw_d = int(sm.get("gwvwD", 1) or 1)
    c.workgroup_mapping = 0  # pipeline_v3 dropped this from the feature set
    return c


def load_origami_configs_by_dtype(
    dat_dir: str, quiet: bool, scale_mode: Optional[int] = None
) -> Tuple[
    Dict[Tuple[str, str, str, str, str], List[Any]],
    Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]],
]:
    """Walk .dat files and build, per (a_type, d_type, compute_type,
    transA, transB) key, two parallel lists:

      configs[key][i]      = origami.config_t with `c.index = i`
      dat_info[key][i]     = dict with the dat-side recovery keys for skip
                              rows: 'sol_idx_global' (top-level .index),
                              'sol_idx_local' (libraryLogicIndex), 'kernel_name'.

    For skip-row enrichment, origami.rank_configs returns ranked entries whose
    `.config.index` survives the round-trip, so we recover sol_idx_global via
    `dat_info[key][ranked[K].config.index]`, then re-read the kernel params
    from the source YAML (see Enricher.enrich_skip).

    `scale_mode` is the run's hipblaslt `scaleA` int. None (default) =
    today's behavior: every library that decodes is loaded. An int keeps
    ONLY the libraries whose `lib.dat.parse_scale_mode` matches -- the
    gfx1250 MX and non-MX F8 TN libraries share an identical
    (a_type, d_type, compute_type, transA, transB) key, so without this
    filter their kernels would merge into one pool."""
    configs_out: Dict[Tuple[str, str, str, str, str], List[Any]] = {}
    info_out: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    n_files = 0
    n_sols = 0
    n_skipped = 0
    t0 = time.time()
    for fname in sorted(os.listdir(dat_dir)):
        # Accept both plain `.dat` and compressed `.dat.zlib` (productized
        # hipBLASLt deflates the Tensile logic); dat.read_tensile_logic
        # transparently decompresses.
        if not dat.is_tensile_contraction_logic(fname):
            continue
        # Decode via lib/dat.py -- this used to be an inline re-implementation
        # of the same pattern walk, which is exactly how the two copies drifted
        # (only one of them ever learned about the MX F8 library).
        decoded = _parse_dtype_layout(fname)
        if decoded is None:
            continue
        a_type, d_type, compute_type, transA, transB = decoded
        if scale_mode is not None and _parse_scale_mode(fname) != int(scale_mode):
            n_skipped += 1
            continue
        data = dat.read_tensile_logic(os.path.join(dat_dir, fname))
        if data is None:
            continue
        n_files += 1
        key = (a_type, d_type, compute_type, transA, transB)
        cfg_list = configs_out.setdefault(key, [])
        info_list = info_out.setdefault(key, [])
        for sol in data.get("solutions", []):
            sm = sol.get("sizeMapping", {}) or {}
            cfg_position = len(cfg_list)
            cfg_list.append(_origami_config_from_dat_solution(sm, cfg_position))
            n_sols += 1
            # `kernel_dat_info` carries the recovery keys (sol_idx_global /
            # _local / kernel_name) plus the base config params. enrich_skip
            # only consumes the indices from here -- it re-reads the kernel
            # params from the source YAML so skip rows stay byte-consistent
            # with tested rows.
            info_list.append(dat.kernel_dat_info(sol))
    if not quiet:
        _ok(
            "load",
            f"origami configs: {_fmt_int(n_sols)} kernels "
            f"({len(configs_out)} dtype/layout combos) from {n_files} "
            f".dat files; dat→solIdx index built "
            f"({_fmt_dur(time.time()-t0)})",
        )
        if scale_mode is not None:
            _info(
                "load",
                f"scale_mode={int(scale_mode)} filter skipped "
                f"{n_skipped} .dat file(s)",
            )
    return configs_out, info_out


def _origami_hardware(device: int, quiet: bool, arch: str = "gfx950") -> Any:
    """Prefer origami.get_hardware_for_device(device), which auto-detects
    actual GPU properties (CU count, L2, MALL, clock). Fall back to the
    arch-only descriptor with no hardcoded numeric overrides.

    `arch` only selects the *fallback* descriptor; the device probe already
    reports whatever silicon is actually present. It used to be hardcoded
    to gfx950, which meant a gfx1250 host that failed the device probe got
    silently enriched with gfx950 hardware constants."""
    if not _ORIGAMI_OK:
        return None
    try:
        hw = origami.get_hardware_for_device(device)
        if not quiet:
            _ok(
                "load",
                f"origami hardware: {arch} (device {device}), " f"N_CU={hw.N_CU}",
            )
        return hw
    except Exception as e:
        if not quiet:
            _warn(
                "load",
                f"get_hardware_for_device({device}) failed: {e}; "
                f"falling back to arch-only descriptor",
            )
    # Resolve the arch name against the binding rather than assuming it
    # exists: origami's architecture_t is generated per build, so a name
    # this pipeline supports may simply be absent from the .so on hand.
    # Say which name failed instead of dying on a bare AttributeError.
    # origami enumerates COMPILER TARGETS, not ASIC revisions: there is an
    # `architecture_t.gfx1250` but no `gfx1250v0`. Canonicalise so a
    # revisioned config arch (which is what selects the kernel tree and the
    # deploy path) still resolves to real hardware constants here.
    arch_enum = getattr(origami.architecture_t, _canonical_arch(arch), None)
    if arch_enum is None:
        if not quiet:
            _warn(
                "load",
                f"origami.architecture_t has no member {arch!r} "
                f"(binding built without it?); no hardware descriptor",
            )
        return None
    # get_hardware_for_arch is NOT a one-argument call: the binding's signature
    # is (arch, N_CU, lds_capacity, rf_capacity, L2_capacity, compute_clock_khz,
    # pci_chip_id=None). Calling it with the enum alone raises TypeError, which
    # the old bare `except Exception: return None` swallowed -- so this whole
    # fallback silently returned None and the "arch-only descriptor" it claims
    # to provide never existed. Probe another device first (the common real
    # cause is one bad device, not a bad build), and only then give up with a
    # message that says why.
    for alt in range(4):
        if alt == device:
            continue
        try:
            hw = origami.get_hardware_for_device(alt)
            if not quiet:
                _ok(
                    "load",
                    f"origami hardware (fallback): {arch} via device "
                    f"{alt}, N_CU={hw.N_CU}",
                )
            return hw
        except Exception:
            continue
    if not quiet:
        _warn(
            "load",
            f"no origami hardware for {arch}: device {device} failed "
            f"and no other device answered. get_hardware_for_arch() is not a "
            f"usable fallback here -- it needs the full hardware geometry "
            f"(N_CU, lds/rf/L2 capacity, clock), which we do not have "
            f"without a device. Features that need hardware will be skipped.",
        )
    return None


def _make_problem(
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
    """Mirror of `lib.origami.make_problem` (kept local so stage04 can be
    imported without the lib origami wrapper). `mx_block_size` is the
    elements-per-scale of an MX format -- 32 for MXFP8 / hipblaslt
    `scaleA: 3` -- and 0 for every non-MX problem."""
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = batch
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


def _cfg_from_origami_config(c: Any) -> Dict[str, Any]:
    """Map origami config_t to our CONFIG_FIELDS. Used as the FALLBACK for
    skip rows when the YAML-by-index lookup misses; the primary skip path
    mirrors tested rows and reads params from the source YAML."""
    return {
        "mt_m": int(getattr(c.mt, "m", 0)),
        "mt_n": int(getattr(c.mt, "n", 0)),
        "mt_k": int(getattr(c.mt, "k", 0)),
        "mi_m": int(getattr(c.mi, "m", 0)),
        "mi_n": int(getattr(c.mi, "n", 0)),
        "mi_k": int(getattr(c.mi, "k", 0)),
        "occupancy": max(int(getattr(c, "occupancy", 1)), 1),
        "cache_hints_a": int(getattr(c, "cache_hints_a", 0)),
        "cache_hints_b": int(getattr(c, "cache_hints_b", 0)),
        "grvw_a": int(getattr(c, "grvw_a", 1)),
        "grvw_b": int(getattr(c, "grvw_b", 1)),
        "gwvw_d": int(getattr(c, "gwvw_d", 1)),
    }


# ── log parser + enricher ─────────────────────────────────────────────────────


class Enricher:
    """Holds caches: dat-by-name, yaml-by-path (for swizzle/lds), origami
    per-(yaml-set, problem) ranking."""

    def __init__(
        self,
        dat_dir: str,
        match_table_path: str,
        quiet: bool,
        device: int = 0,
        arch: str = "gfx950",
        scale_mode: Optional[int] = None,
    ):
        self.quiet = quiet
        # The run's hipblaslt scaleA mode. Selects WHICH .dat libraries
        # feed the skip-row config pool (MX vs non-MX F8 share a key) and
        # whether the ranked problems are built as MX problems.
        self.scale_mode = scale_mode
        self.mx_block_size = mx_block_size_for_scale_mode(scale_mode)
        t0 = time.time()
        self.match_table = parse_match_table(match_table_path)
        if not quiet:
            _ok(
                "load",
                f"MatchTable: {_fmt_int(len(self.match_table))} "
                f"entries ({_fmt_dur(time.time()-t0)})",
            )
        # .dat is consulted ONLY for occupancy (the YAML stores the -1
        # sentinel). Keyed by global index, so collision-free; all other kernel
        # params come from the source YAML by (yaml_path, local_idx).
        self.dat_occ_by_global = build_dat_occupancy_index(dat_dir, quiet)
        # Authoritative kernel identity, keyed by the RUNTIME global index.
        self.dat_cfg_by_global = build_dat_config_index(dat_dir, quiet)

        # Per-yaml kernel-param cache, keyed by (yaml_path, local_idx).
        self._yaml_sub_cache: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
        # Counter of MT mismatches between log sol_name and .dat sizeMapping.
        self._mt_mismatches = 0

        # origami skip-row enrichment: load configs by (dtype/layout) and
        # a parallel `dat_info` list giving sol_idx_global, sol_idx_local
        # and LDS pad per .dat position. After rank_configs returns a
        # ranked permutation, we recover the .dat position via
        # `ranked[K].config.index` and index dat_info to fill the row.
        self.origami_hw = None
        self.origami_configs_by_dtype: Dict[
            Tuple[str, str, str, str, str], List[Any]
        ] = {}
        self.origami_dat_info_by_dtype: Dict[
            Tuple[str, str, str, str, str], List[Dict[str, Any]]
        ] = {}
        self._rank_cache: Dict[Tuple, List[Any]] = {}
        if _ORIGAMI_OK:
            (self.origami_configs_by_dtype, self.origami_dat_info_by_dtype) = (
                load_origami_configs_by_dtype(dat_dir, quiet, scale_mode=scale_mode)
            )
            self.origami_hw = _origami_hardware(device, quiet, arch)
            if self.origami_hw is None and not quiet:
                _warn(
                    "load",
                    "no origami hardware available; skip rows "
                    "will have blank kernel params",
                )
        elif not quiet:
            _warn(
                "load",
                "origami python binding not importable; skip "
                "rows will have blank kernel params",
            )

    # -- tested-row helpers ---------------------------------------------------

    def _yaml_sub(self, yaml_path: str, local_idx: int) -> Optional[Dict[str, Any]]:
        key = (yaml_path, local_idx)
        if key not in self._yaml_sub_cache:
            self._yaml_sub_cache[key] = _yaml_kernel_cfg(yaml_path, local_idx)
        return self._yaml_sub_cache[key]

    def enrich_tested(
        self, sol_idx_global: int, sol_name: str
    ) -> Optional[Dict[str, Any]]:
        """Return enriched dict (CONFIG_FIELDS + sol_idx_local) or None on miss.

        Kernel params (mt/mi/cache_hints_a/b/grvw/gwvw) are read from the
        SOURCE YAML by (yaml_path, local_idx) via the MatchTable -- the
        authoritative, collision-free source (the old .dat-by-name lookup
        returned the wrong NonTemporal variant for kernels whose names collide
        across the base / Origami_nta4 / Origami_ntb4 libraries). Occupancy is
        the one field taken from the .dat (keyed by global index), mirroring
        hipBLASLt's runtime SizeMapping. Validates the MT tile parsed from the
        log's sol_name against the YAML MT; on mismatch records a counter."""
        # Resolve by the RUNTIME global index straight out of the .dat. Do NOT
        # go through MatchTable.yaml: its numbering can be a different index
        # space (offset from the runtime's), which silently attaches the wrong
        # kernel's tile to every row.
        info = self.dat_cfg_by_global.get(sol_idx_global)
        if info is None:
            return None
        cfg = {
            k: v for k, v in info.items() if k not in ("sol_idx_global", "kernel_name")
        }
        cfg["occupancy"] = self.dat_occ_by_global.get(
            sol_idx_global, cfg.get("occupancy", 1)
        )

        # MT validation: log's sol_name encodes the macro tile geometry as
        # `_MT<m>x<n>x<k>_`. The YAML solution must agree on these three
        # numbers for the same kernel.
        m_match = _MT_NAME_RE.search(sol_name)
        if m_match:
            try:
                nm_mt = (
                    int(m_match.group(1)),
                    int(m_match.group(2)),
                    int(m_match.group(3)),
                )
                yaml_mt = (cfg["mt_m"], cfg["mt_n"], cfg["mt_k"])
                if nm_mt != yaml_mt:
                    self._mt_mismatches += 1
                    if self._mt_mismatches <= 5 and not self.quiet:
                        _warn(
                            "mt",
                            f"sol_idx_global={sol_idx_global}: "
                            f"log_name MT={nm_mt} dat MT={yaml_mt}",
                        )
                    # FATAL. The kernel name in the bench log is ground truth;
                    # disagreeing with it means we are attaching the wrong
                    # kernel's geometry to a measured time, which poisons every
                    # downstream model. A counter printed only at the end is
                    # easy to miss, and whole training runs can be built on
                    # wrong rows before anyone notices.
                    if self._mt_mismatches > _MT_MISMATCH_FATAL_AFTER:
                        raise RuntimeError(
                            f"kernel identity resolution is broken: "
                            f"{self._mt_mismatches} MT mismatches between the "
                            f"bench log's kernel name and the .dat entry "
                            f"(e.g. sol_idx_global={sol_idx_global}: "
                            f"log={nm_mt} dat={yaml_mt}). Refusing to emit a "
                            f"dataset whose kernel identities are wrong."
                        )
            except (TypeError, ValueError):
                pass
        return cfg

    # -- skip-row helpers -----------------------------------------------------

    def enrich_skip(
        self,
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
        rank: int,
    ) -> Optional[Dict[str, Any]]:
        """Fill kernel params for a skipped solution via
        origami.rank_configs(problem, hw, configs). configs are the .dat
        solutions for the matching (a_type, d_type, compute_type, transA,
        transB) — same key offline_eval.py uses.

        The .dat solution dict carries top-level `index` (== sol_idx_global,
        the hipblaslt-bench MatchTable key) and `libraryLogicIndex`
        (== sol_idx_local, the yaml-local kernel index). We stash a
        `cfg_position` on `config_t.index` at load time, so after
        rank_configs returns a ranked permutation we recover sol_idx_global
        via `ranked[K].config.index`. To keep skip rows BYTE-CONSISTENT with
        tested rows, the kernel params are then re-read from the source YAML
        by (yaml_path, local_idx) -- the same authoritative path tested rows
        use -- with occupancy from the .dat; if that lookup misses we fall
        back to the origami config_t values.
        """
        if not _ORIGAMI_OK or self.origami_hw is None:
            return None
        # The bench log's `compute_type` is the un-prefixed form (`f32_r`);
        # offline_eval.py keys by the `c_`-prefixed form that the .dat
        # filename pattern decodes to (`c_f32_r`).
        ct = (compute_type or "").strip()
        if not ct.startswith("c_"):
            ct = "c_" + ct
        key = (a_type, d_type, ct, transA, transB)
        configs = self.origami_configs_by_dtype.get(key)
        info_list = self.origami_dat_info_by_dtype.get(key)
        if not configs or not info_list:
            return None
        rk = (key, m, n, k, batch)
        ranked = self._rank_cache.get(rk)
        if ranked is None:
            try:
                problem = _make_problem(
                    m,
                    n,
                    k,
                    batch,
                    a_type,
                    b_type,
                    d_type,
                    ct,
                    transA,
                    transB,
                    mx_block_size=self.mx_block_size,
                )
                ranked = _rank_configs_compat(
                    origami, problem, self.origami_hw, configs
                )
            except Exception:
                ranked = []
            self._rank_cache[rk] = ranked
        if not ranked or rank < 0 or rank >= len(ranked):
            return None
        ranked_cfg = ranked[rank].config
        pos = int(getattr(ranked_cfg, "index", -1))
        cfg: Optional[Dict[str, Any]] = None
        sg = sl = None
        if 0 <= pos < len(info_list):
            info = info_list[pos]
            sg = info.get("sol_idx_global")
            sl = info.get("sol_idx_local")
            # Primary: read params from the source YAML by (yaml_path,
            # local_idx) -- identical to tested rows -- with occupancy from
            # the .dat. Keeps skip / tested rows byte-consistent and free of
            # the .dat-by-name NonTemporal-variant collision.
            if sg is not None:
                # Same authoritative source as tested rows: the .dat keyed by
                # the RUNTIME global index. (Was MatchTable, whose numbering is
                # a different index space -- see build_dat_config_index.)
                info_dat = self.dat_cfg_by_global.get(int(sg))
                if info_dat is not None:
                    cfg = {
                        k: v
                        for k, v in info_dat.items()
                        if k not in ("sol_idx_global", "kernel_name")
                    }
                    cfg["occupancy"] = self.dat_occ_by_global.get(
                        int(sg), cfg.get("occupancy", 1)
                    )
        if cfg is None:
            # Fallback: origami config_t values (already collision-free, since
            # the origami cfg list is indexed positionally, not by name).
            cfg = _cfg_from_origami_config(ranked_cfg)
        cfg["sol_idx_global"] = "" if sg is None else int(sg)
        cfg["sol_idx_local"] = "" if sl is None else int(sl)
        return cfg


# ── log streaming ─────────────────────────────────────────────────────────────


# ── equivalent-kernel collapse (TemporalHint triples) ────────────────────────
#
# HISTORICAL (TH-DISABLED, upstream 24145512bf): PR #11588 briefly shipped each
# gfx1250 tuned kernel THREE times -- the base plus `Origami_tha1` /
# `Origami_thb1` TemporalHint variants -- then reverted them for kernel-load
# overhead. While they existed the hint never reached the .dat metadata (no
# `temporalHint*` key and `nonTemporalA/B` are 0 for all of them), so all three
# collapse to ONE kernel signature
# `(MT0, MT1, depthU, MI0, MI1, MI2, cache_hints_a, cache_hints_b)`.
#
# Left alone that hands the ranker three rows with IDENTICAL features and
# DIFFERENT labels for the same problem -- pure contradictory supervision --
# and makes smart_K signature lists collide 3-to-1 at deploy time.
#
# We keep the MEDIAN-`us` row of each group, deliberately NOT the fastest.
# In-sweep, the variants of one signature can come back far apart, while the
# same kernels measured in ISOLATION agree closely. The fast outlier is a
# warm-cache ordering artifact (identical kernels run back-to-back on the same
# buffers), so best-of-group would teach the model a latency this kernel does
# not really achieve. The median rejects that outlier.
#
# Where signatures are already unique (every gfx950 library), this is a no-op.

_COLLAPSE_STATS = {"rows_dropped": 0, "problems": 0}


def _collapse_equivalent_rows(
    problem_rows: List[Dict[str, Any]],
    us_ok,
) -> Tuple[List[Dict[str, Any]], int]:
    """Keep one row per kernel signature for this problem. Returns
    (rows, n_dropped)."""
    from lib.dat import sig_from_row  # noqa: E402

    groups: "OrderedDict[Any, List[Dict[str, Any]]]" = OrderedDict()
    unkeyed: List[Dict[str, Any]] = []
    for r in problem_rows:
        # An unresolved row (enrichment miss) has EMPTY kernel params, so every
        # such row in a problem yields the SAME signature and would be merged
        # into one -- silently destroying the problem. Only ever collapse rows
        # whose params actually resolved.
        if not str(r.get("mt_m", "")).strip():
            unkeyed.append(r)
            continue
        try:
            key = sig_from_row(r)
        except Exception:
            unkeyed.append(r)
            continue
        if key is None:
            unkeyed.append(r)
            continue
        groups.setdefault(key, []).append(r)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for _key, rows in groups.items():
        if len(rows) == 1:
            kept.append(rows[0])
            continue
        measured = [r for r in rows if us_ok(r)]
        if measured:
            measured.sort(key=lambda r: float(r["us"]))
            kept.append(measured[len(measured) // 2])  # lower median
        else:
            kept.append(rows[0])  # all skips: keep one
        dropped += len(rows) - 1
    return unkeyed + kept, dropped


def _finalize_problem(
    problem_rows: List[Dict[str, Any]], enricher: Enricher
) -> List[Dict[str, Any]]:
    """Stamp is_winner=True on the min-us tested row, then enrich skip rows
    via origami.rank_configs() using the .dat-derived config list for this
    problem's (a_type, d_type, compute_type, transA, transB) key."""
    if not problem_rows:
        return problem_rows

    # `us == ""` (no measurement) and `us == 0.0` (kernel ran but the
    # bench tool reported zero -- some unsupported configs do this and
    # they must NEVER be the winner) both have to be excluded.
    def _us_ok(r: Dict[str, Any]) -> bool:
        if r["sol_idx_global"] == "" or r["us"] == "":
            return False
        try:
            return float(r["us"]) > 0.0
        except (TypeError, ValueError):
            return False

    # Collapse TemporalHint (and any other) duplicate signatures BEFORE picking
    # the winner -- otherwise the winner can be an artifact row from a triple.
    problem_rows, _n_collapsed = _collapse_equivalent_rows(problem_rows, _us_ok)
    if _n_collapsed:
        _COLLAPSE_STATS["rows_dropped"] += _n_collapsed
        _COLLAPSE_STATS["problems"] += 1
    tested = [r for r in problem_rows if _us_ok(r)]
    if tested:
        min_row = min(tested, key=lambda r: float(r["us"]))
        min_row["is_winner"] = "True"

    # ---- validate the skip-row rank reconstruction before trusting it ------
    # Skip rows carry only "Skip solution: K" -- a position in the bench's
    # origami-ranked candidate order -- so their kernel identity is REBUILT by
    # replaying origami.rank_configs(). That replay can disagree with the bench:
    # the bench may rank a slightly different number of candidates than stage04
    # rebuilds, and any length difference shifts every rank.
    #
    # Tested rows are the ground truth: they carry BOTH `rank` and the true
    # `sol_idx_global`. Replay a few of them; if the reconstruction disagrees on
    # any, the mapping is shifted for this problem and every skip row it would
    # produce is mislabelled -- so leave them un-enriched (blank
    # sol_idx_global), which stage05 already drops, rather than feeding the
    # model a kernel identity we know to be wrong.
    _probe = []
    for r in tested:
        try:
            _probe.append(
                (
                    int(r["rank"]),
                    int(r["sol_idx_global"]),
                    int(r["m"]),
                    int(r["n"]),
                    int(r["k"]),
                    int(r["batch_count"]),
                    r,
                )
            )
        except (ValueError, TypeError):
            continue
        if len(_probe) >= 24:
            break
    # Solve for the rank offset. hipblaslt-bench with TENSILE_PREDICTION_LIB=1
    # calls solutions in ORIGAMI RANK ORDER, so `Skip solution: K` is a position
    # in that order -- but the bench also calls a small number of NON-Origami
    # solutions (Equality/MAF-MAB entries, which bypass rank_configs) first, so
    # the sequence is origami's list shifted by a constant -- e.g. one leading
    # non-Origami solution gives bench[r] == origami_ranked[r - 1].
    #
    # Rather than hardcode 1, solve for the shift per problem against the tested
    # rows -- they carry BOTH the rank and the true sol_idx_global, so the
    # offset is verifiable, not assumed. If no offset explains every probe, the
    # mapping is genuinely unknown and we leave the skip rows un-enriched
    # instead of guessing a kernel identity.
    _rank_off = None
    if _probe:
        # Score every candidate offset and take the best, rather than demanding
        # unanimity: `dup_rows_merged` collapses equivalent-signature kernels to
        # their MEDIAN row, so a probe row can legitimately carry a different
        # sol_idx_global than the ranked list even when the offset is right.
        # Requiring every probe to agree rejects many problems whose offset is
        # correct; a strong majority keeps the proof while tolerating
        # collapsed rows. An offset is accepted only if it explains most probes
        # AND clears an absolute floor, so a 1-of-2 fluke cannot win.
        _best = (0, None)
        for cand in (1, 0, 2, 3, 4, 5, -1, -2, -3, -4, -5):
            hits = 0
            for rk, true_g, m_, n_, k_, bc_, r in _probe:
                if rk - cand < 0:
                    continue
                chk = enricher.enrich_skip(
                    m_,
                    n_,
                    k_,
                    bc_,
                    r["a_type"],
                    r["b_type"],
                    r["d_type"],
                    r["compute_type"],
                    r["transA"],
                    r["transB"],
                    rk - cand,
                )
                got = None if chk is None else chk.get("sol_idx_global")
                if got not in (None, "") and int(got) == true_g:
                    hits += 1
            if hits > _best[0]:
                _best = (hits, cand)
            if hits == len(_probe):
                break
        # Denominator must count only probes that COULD match. The bench calls a
        # couple of Equality/MAF-MAB solutions that are not in the Origami pool
        # at all, so they can never match any offset -- and on problems with few
        # tested rows they dominate the probe set and sink an otherwise correct
        # calibration -- e.g. a problem whose few tested rows are mostly
        # Equality solutions would be rejected and lose every one of its skip
        # rows. Count a probe as matchable if ANY candidate offset explains it.
        _matchable = 0
        for rk, true_g, m_, n_, k_, bc_, r in _probe:
            for cand in (1, 0, 2, 3, 4, 5, -1, -2, -3, -4, -5):
                if rk - cand < 0:
                    continue
                chk = enricher.enrich_skip(
                    m_,
                    n_,
                    k_,
                    bc_,
                    r["a_type"],
                    r["b_type"],
                    r["d_type"],
                    r["compute_type"],
                    r["transA"],
                    r["transB"],
                    rk - cand,
                )
                got = None if chk is None else chk.get("sol_idx_global")
                if got not in (None, "") and int(got) == true_g:
                    _matchable += 1
                    break
        _need = max(1, int(0.7 * _matchable)) if _matchable else 10**9
        if _best[1] is not None and _best[0] >= _need:
            _rank_off = _best[1]
        _COLLAPSE_STATS["rank_off_checked"] = (
            _COLLAPSE_STATS.get("rank_off_checked", 0) + 1
        )
        if _rank_off is None:
            _COLLAPSE_STATS["rank_off_unsolved"] = (
                _COLLAPSE_STATS.get("rank_off_unsolved", 0) + 1
            )
        else:
            _COLLAPSE_STATS[f"rank_off_{_rank_off}"] = (
                _COLLAPSE_STATS.get(f"rank_off_{_rank_off}", 0) + 1
            )

    for row in problem_rows:
        if row["sol_idx_global"] != "" or not row.get("_is_skip"):
            row.pop("_is_skip", None)
            continue
        # Preferred path: the bench told us the solution index outright, so the
        # identity is exact and independent of how hipBLASLt ordered candidates
        # against origami's ranking.
        direct_idx = row.pop("_skip_sol_idx", None)
        if direct_idx is not None:
            cfg = enricher.enrich_tested(direct_idx, "")
            if cfg is not None:
                row["sol_idx_global"] = str(direct_idx)
                sl = cfg.get("sol_idx_local", "")
                row["sol_idx_local"] = "" if sl in (None, "") else str(sl)
                for kk in CONFIG_FIELDS:
                    row[kk] = cfg.get(kk, "")
            elif direct_idx not in enricher.dat_cfg_by_global:
                # Not in the Origami pool. TENSILE_PREDICTION_LIB=1 only skips
                # EqualityMatching and RangeMatching rows, so GridBased and
                # FreeSize solutions still get benched -- typically a few per
                # problem. They are excluded from training deliberately, so
                # count them apart from identities we simply failed to resolve.
                row["_non_origami"] = True
            row.pop("_is_skip", None)
            continue
        try:
            rank = int(row["rank"])
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
            bc = int(row["batch_count"])
        except (ValueError, TypeError):
            row.pop("_is_skip", None)
            continue
        if _rank_off is None:
            # no offset explains the tested rows -- do not guess an identity
            row.pop("_is_skip", None)
            continue
        rank = rank - _rank_off
        if rank < 0:
            row.pop("_is_skip", None)
            continue
        cfg = enricher.enrich_skip(
            m,
            n,
            k,
            bc,
            row["a_type"],
            row["b_type"],
            row["d_type"],
            row["compute_type"],
            row["transA"],
            row["transB"],
            rank,
        )
        if cfg is not None:
            sg = cfg.get("sol_idx_global", "")
            row["sol_idx_global"] = "" if sg in (None, "") else str(sg)
            sl = cfg.get("sol_idx_local", "")
            row["sol_idx_local"] = "" if sl in (None, "") else str(sl)
            for kk in CONFIG_FIELDS:
                row[kk] = cfg.get(kk, "")
        row.pop("_is_skip", None)
    return problem_rows


def process_log_file(
    log_path: str,
    enricher: Enricher,
    counters: Dict[str, int],
    max_gemms_remaining: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return all enriched rows for one block log file plus the number of
    GEMMs consumed (so callers tracking `--max-gemms` can stop)."""
    with open(log_path) as f:
        lines = f.readlines()

    out: List[Dict[str, Any]] = []
    cur_problem: List[Dict[str, Any]] = []
    cur_fields: Dict[str, str] = {}
    cur_entry: Optional[Dict[str, Any]] = None
    cur_sol_idx_global: Optional[int] = None
    cur_rank: Optional[int] = None
    in_winner = False
    gemms_done = 0

    def _flush_problem():
        nonlocal cur_problem, cur_fields, gemms_done
        if cur_problem:
            _finalize_problem(cur_problem, enricher)
            out.extend(cur_problem)
            gemms_done += 1
        cur_problem = []
        cur_fields = {}

    for line in lines:
        s = line.strip()

        if s.startswith("Rotating buffer"):
            _flush_problem()
            cur_entry = None
            cur_sol_idx_global = None
            cur_rank = None
            in_winner = False
            if max_gemms_remaining is not None and gemms_done >= max_gemms_remaining:
                return out, gemms_done
            continue

        if (
            not s
            or s.startswith("hipBLASLt")
            or s.startswith("Query device")
            or s.startswith("Device ID")
            or s.startswith("with ")
            or s.startswith("maxGrid")
        ):
            continue

        if s.startswith("Is supported"):
            in_winner = False
            continue

        if s == "Winner:":
            in_winner = True
            cur_entry = None
            cur_sol_idx_global = None
            cur_rank = None
            continue

        sm = SKIP_RE.match(s)
        if sm and not in_winner:
            rank = int(sm.group(1))
            warm_up = sm.group(2)
            if cur_fields:
                row: Dict[str, Any] = dict(cur_fields)
                row["us"] = warm_up
                row["hipblaslt-Gflops"] = ""
                row["hipblaslt-GB/s"] = ""
                row["rank"] = rank
                row["is_winner"] = "False"
                row["is_skip"] = "True"
                row["sol_idx_global"] = ""
                row["sol_idx_local"] = ""
                row["_is_skip"] = True
                sidx = SKIP_IDX_RE.search(s)
                if sidx:
                    row["_skip_sol_idx"] = int(sidx.group(1))
                for kk in CONFIG_FIELDS:
                    row[kk] = ""
                cur_problem.append(row)
                counters["skip_rows"] += 1
            continue

        hm = HEADER_RE.match(s)
        if hm:
            cur_rank = int(hm.group(1))
            continue

        im = SOL_IDX_RE.match(s)
        if im:
            if cur_entry is not None and not in_winner:
                cur_sol_idx_global = int(im.group(1))
                cur_entry["sol_idx_global"] = cur_sol_idx_global
            continue

        nm = SOL_NAME_RE.match(s)
        if nm:
            if (
                cur_entry is not None
                and not in_winner
                and cur_sol_idx_global is not None
            ):
                sol_name = nm.group(1).strip()
                cfg = enricher.enrich_tested(cur_sol_idx_global, sol_name)
                if cfg is None:
                    cur_entry["sol_idx_local"] = ""
                    for kk in CONFIG_FIELDS:
                        cur_entry[kk] = ""
                    counters["miss_tested"] += 1
                else:
                    cur_entry["sol_idx_local"] = cfg["sol_idx_local"]
                    for kk in CONFIG_FIELDS:
                        cur_entry[kk] = cfg.get(kk, "")
                    counters["enriched_tested"] += 1
                cur_problem.append(cur_entry)
                counters["tested_rows"] += 1
                cur_entry = None
                cur_sol_idx_global = None
            continue

        if s.startswith("--kernel name:") or s.startswith("--"):
            continue

        if s.startswith("T,") or s.startswith("N,"):
            if in_winner:
                continue
            fields = [f.strip() for f in s.split(",")]
            if len(fields) < len(_RAW_DATA_FIELDS):
                continue
            raw = {_RAW_DATA_FIELDS[i]: fields[i] for i in range(len(_RAW_DATA_FIELDS))}
            row = {f: raw[f] for f in DATA_FIELDS}
            row["rank"] = cur_rank if cur_rank is not None else -1
            row["is_winner"] = "False"
            row["is_skip"] = "False"
            row["sol_idx_global"] = ""
            row["sol_idx_local"] = ""
            for kk in CONFIG_FIELDS:
                row[kk] = ""
            if not cur_fields:
                cur_fields = {k: row[k] for k in PROBLEM_FIELDS}
            cur_entry = row

    _flush_problem()
    return out, gemms_done


# ── chunked CSV writer ────────────────────────────────────────────────────────


class ChunkWriter:
    def __init__(self, out_dir: str, rows_per_chunk: int, prefix: str = "chunk_"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.rows_per_chunk = rows_per_chunk
        self.prefix = prefix
        self._n_chunks = 0
        self._rows_in_chunk = 0
        self._total_rows = 0
        self._writer: Optional[csv.DictWriter] = None
        self._fh = None

    def _open(self):
        path = os.path.join(self.out_dir, f"{self.prefix}{self._n_chunks:04d}.csv")
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=OUT_FIELDS, extrasaction="ignore"
        )
        self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._open()
        self._writer.writerow(row)
        self._rows_in_chunk += 1
        self._total_rows += 1
        if self._rows_in_chunk >= self.rows_per_chunk:
            assert self._fh is not None
            self._fh.close()
            self._fh = None
            self._writer = None
            self._n_chunks += 1
            self._rows_in_chunk = 0

    def close(self) -> Tuple[int, int]:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
            self._n_chunks += 1
        return self._n_chunks, self._total_rows


# ── main ──────────────────────────────────────────────────────────────────────


def _print_progress(state: Dict[str, Any]) -> None:
    files_done = state["files_done"]
    files_total = state["files_total"]
    gemms_done = state["gemms_done"]
    gemms_total_est = state["gemms_total_est"]
    rows_total = state["rows_total"]
    counters = state["counters"]
    elapsed = time.time() - state["t0"]
    if files_done > 0:
        avg = elapsed / files_done
        eta = avg * (files_total - files_done)
        eta_s = _fmt_dur(eta)
    else:
        eta_s = "calculating..."

    print(f"\n{C.BOLD}{'-'*80}{C.ENDC}", flush=True)
    print(
        f"{C.BOLD}Files  : {files_done:>4d}/{files_total:<4d} "
        f"({_pct(files_done, files_total)})    "
        f"GEMMs  : {_fmt_int(gemms_done):>10s}/{_fmt_int(gemms_total_est):<10s} "
        f"({_pct(gemms_done, gemms_total_est)}){C.ENDC}",
        flush=True,
    )
    print(
        f"{C.GREEN}Elapsed: {_fmt_dur(elapsed):<10s}{C.ENDC} "
        f"{C.CYAN}ETA: {eta_s:<10s}{C.ENDC} "
        f"{C.BLUE}Rate: {(gemms_done/elapsed) if elapsed > 0 else 0:>6.1f} gemm/s{C.ENDC}",
        flush=True,
    )
    skip_done = counters["enriched_skip"]
    skip_blank = counters["skip_blank"]
    print(
        f"{C.GREY}Rows   : total={_fmt_int(rows_total):>11s}  "
        f"tested={_fmt_int(counters['enriched_tested']):>11s}  "
        f"skip-enriched={_fmt_int(skip_done):>9s}  "
        f"skip-blank={_fmt_int(skip_blank):>7s}{C.ENDC}",
        flush=True,
    )
    if counters["miss_tested"] or state["enricher"]._mt_mismatches:
        print(
            f"{C.YELLOW}Warnings: "
            f"tested-miss={counters['miss_tested']}  "
            f"MT-mismatches={state['enricher']._mt_mismatches}{C.ENDC}",
            flush=True,
        )
    print(f"{C.BOLD}{'-'*80}{C.ENDC}\n", flush=True)


# Core kernel-param columns that drive the `tile` feature group + occupancy.
# A tested row with any of these empty means enrichment failed to resolve the
# kernel from the source YAML (a real bug); we warn on empties/constants.
_CORE_CONFIG_FIELDS = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
)


def validate_enriched_dataset(out_dir: str, quiet: bool = False) -> None:
    """Sanity-check the enriched CSVs after writing. For the core kernel-param
    columns, count how many TESTED rows leave them empty (a YAML-resolution
    miss) and warn on constant columns."""
    import csv as _csv
    import glob as _glob
    import collections as _c

    chunks = sorted(_glob.glob(os.path.join(out_dir, "*.csv")))
    if not chunks:
        _warn("validate", f"no enriched CSV chunks in {out_dir}; skip check")
        return
    n_tested = 0
    empty_on_tested: "_c.Counter[str]" = _c.Counter()
    distinct: Dict[str, set] = _c.defaultdict(set)
    cols = list(_CORE_CONFIG_FIELDS)
    for cf in chunks:
        with open(cf) as fh:
            for row in _csv.DictReader(fh):
                # Only tested rows are expected to carry kernel params; skip
                # rows beyond origami's ranked count legitimately stay blank.
                if str(row.get("is_skip", "")).strip() == "True":
                    continue
                n_tested += 1
                for col in cols:
                    v = str(row.get(col, "")).strip()
                    if v == "":
                        empty_on_tested[col] += 1
                    elif len(distinct[col]) < 8:
                        distinct[col].add(v)
    if n_tested == 0:
        _warn("validate", "enriched CSVs have 0 tested rows; skip check")
        return
    empty = [c for c in cols if empty_on_tested[c] == n_tested]
    const = [c for c in cols if len(distinct[c]) == 1 and empty_on_tested[c] < n_tested]
    if not quiet:
        _ok(
            "validate",
            f"enriched dataset: {_fmt_int(n_tested)} tested rows; "
            f"core kernel cols checked={len(cols)} all-empty={len(empty)} "
            f"constant={len(const)}",
        )
    for c in const:
        _warn(
            "validate",
            f"kernel column '{c}' is CONSTANT "
            f"(value={next(iter(distinct[c]))}) across {_fmt_int(n_tested)} "
            f"tested rows -- verify this is expected for this dtype/layout.",
        )
    if empty:
        _err(
            "validate",
            f"core kernel column(s) {empty} are ENTIRELY EMPTY "
            f"across {_fmt_int(n_tested)} tested rows -- enrichment failed to "
            f"resolve kernels from the source YAML; aborting.",
        )
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="hipblaslt-bench block logs -> chunked enriched CSV folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--log-dir", required=True, help="Directory with block_*.log files")
    ap.add_argument(
        "--out-dir", required=True, help="Output directory for chunk_NNNN.csv files"
    )
    ap.add_argument(
        "--dat-dir",
        required=True,
        help="Library .dat dir (e.g. <build>/Tensile/library/gfx950)",
    )
    ap.add_argument(
        "--match-table",
        required=True,
        help="MatchTable.yaml from <build>/device-library/",
    )
    ap.add_argument("--rows-per-chunk", type=int, default=10_000)
    ap.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index for origami.get_hardware_for_device",
    )
    # Default gfx950 preserves the previous hardcoded behaviour for callers
    # that don't pass the flag; run_orchestrator.py always passes the arch
    # from the run config.
    ap.add_argument(
        "--arch",
        default="gfx950",
        help="Arch name for the origami arch-only hardware "
        "fallback (must match an origami.architecture_t "
        "member, e.g. gfx950, gfx1250)",
    )
    # Default None == "no filter" == exactly the pre-MX behavior. The
    # orchestrator passes the run config's `hipblaslt.scaleA` when set.
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA/scaleB mode of this run "
        "(0=none, 1=Scalar, 3=Block_32_UE8M0/MX). When "
        "set, only .dat libraries expecting that mode are "
        "loaded -- required on gfx1250 where the MX and "
        "non-MX F8 TN libraries share one dtype/layout key",
    )
    ap.add_argument(
        "--max-gemms",
        type=int,
        default=None,
        help="Stop after this many GEMMs (small-subset test)",
    )
    ap.add_argument(
        "--progress-every",
        type=float,
        default=10.0,
        help="Seconds between progress reports",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    if not args.quiet:
        _banner("hipblaslt-bench logs  ->  enriched chunked CSV", C.HEADER)
        _info("cfg", f"log_dir        : {args.log_dir}")
        _info("cfg", f"out_dir        : {args.out_dir}")
        _info("cfg", f"dat_dir        : {args.dat_dir}")
        _info("cfg", f"match_table    : {args.match_table}")
        _info("cfg", f"device         : {args.device}")
        _info("cfg", f"arch           : {args.arch}")
        _info("cfg", f"scale_mode     : {args.scale_mode}")
        _info("cfg", f"rows_per_chunk : {args.rows_per_chunk:,}")
        if args.max_gemms is not None:
            _info("cfg", f"max_gemms      : {args.max_gemms} (subset test)")

    log_files = sorted(
        [
            f
            for f in os.listdir(args.log_dir)
            if f.endswith(".log") and (f.startswith("block_") or f.startswith("line"))
        ]
    )
    if not log_files:
        _err("err", f"no block/line .log files in {args.log_dir}")
        raise SystemExit(1)
    if not args.quiet:
        _info("cfg", f"log files      : {len(log_files)}")

    # Stage 1: load indices ----------------------------------------------------
    if not args.quiet:
        _banner("Stage 1/3  Loading library indices", C.BLUE)
    enricher = Enricher(
        args.dat_dir,
        args.match_table,
        args.quiet,
        device=args.device,
        arch=args.arch,
        scale_mode=args.scale_mode,
    )

    # Stage 2: parse + enrich + write -----------------------------------------
    if not args.quiet:
        _banner("Stage 2/3  Parse + enrich + write", C.BLUE)
    writer = ChunkWriter(args.out_dir, args.rows_per_chunk)
    counters: Dict[str, int] = defaultdict(int)

    # Estimate total GEMMs from the file-name line ranges in log filenames so
    # progress %/ETA are meaningful even before the run finishes.
    gemms_total_est = 0
    for fname in log_files:
        m = re.search(r"lines(\d+)-(\d+)_", fname)
        if m:
            gemms_total_est += int(m.group(2)) - int(m.group(1)) + 1
    if args.max_gemms is not None:
        gemms_total_est = min(gemms_total_est, args.max_gemms)

    progress_state = {
        "files_done": 0,
        "files_total": len(log_files),
        "gemms_done": 0,
        "gemms_total_est": gemms_total_est,
        "rows_total": 0,
        "counters": counters,
        "enricher": enricher,
        "t0": t0,
    }
    last_progress = time.time()

    total_gemms_done = 0
    for i, fname in enumerate(log_files, 1):
        if args.max_gemms is not None and total_gemms_done >= args.max_gemms:
            if not args.quiet:
                _info("main", f"hit --max-gemms={args.max_gemms}; stopping")
            break
        remaining = (
            args.max_gemms - total_gemms_done if args.max_gemms is not None else None
        )
        t_file = time.time()
        rows, gemms_done = process_log_file(
            os.path.join(args.log_dir, fname),
            enricher,
            counters,
            max_gemms_remaining=remaining,
        )
        for r in rows:
            writer.write(r)

        # Track skip-row enrichment in counters. Skip rows that resolved carry
        # sol_idx_global / sol_idx_local / kernel params / LDS pad from the
        # .dat lookup; identify them via the is_skip flag.
        for r in rows:
            if r.get("is_skip") == "True":
                if r.get("mt_m", "") != "":
                    counters["enriched_skip"] += 1
                elif r.pop("_non_origami", False):
                    counters["skip_non_origami"] += 1
                else:
                    counters["skip_blank"] += 1

        total_gemms_done += gemms_done
        progress_state["files_done"] = i
        progress_state["gemms_done"] = total_gemms_done
        progress_state["rows_total"] += len(rows)

        if not args.quiet:
            print(
                f"  {C.GREY}[file {i:>3d}/{len(log_files)}]{C.ENDC} "
                f"{fname}  gemms={gemms_done:>5d}  "
                f"rows={_fmt_int(len(rows)):>10s}  "
                f"({_fmt_dur(time.time()-t_file)})",
                flush=True,
            )

        # Periodic progress block.
        if not args.quiet and (time.time() - last_progress) >= args.progress_every:
            _print_progress(progress_state)
            last_progress = time.time()

    n_chunks, total = writer.close()
    elapsed = time.time() - t0

    # Guardrail: fail fast if enrichment left core kernel-param columns empty
    # on tested rows (YAML resolution miss). Runs in quiet mode too.
    validate_enriched_dataset(args.out_dir, quiet=args.quiet)

    # Stage 3: summary --------------------------------------------------------
    if not args.quiet:
        _banner("Stage 3/3  Done", C.GREEN)
        skip_done = counters["enriched_skip"]
        skip_blank = counters["skip_blank"]
        skip_total = skip_done + skip_blank
        print(f"  {C.BOLD}total_rows     : {_fmt_int(total):>14s}{C.ENDC}", flush=True)
        print(
            f"  {C.BOLD}chunks         : {_fmt_int(n_chunks):>14s}{C.ENDC}", flush=True
        )
        print(
            f"  {C.GREEN}tested_rows    : {_fmt_int(counters['enriched_tested']):>14s}  "
            f"(miss={counters['miss_tested']}){C.ENDC}",
            flush=True,
        )
        print(
            f"  {C.GREEN}skip_enriched  : {_fmt_int(skip_done):>14s}  "
            f"({_pct(skip_done, max(skip_total, 1))} of skip rows){C.ENDC}",
            flush=True,
        )
        skip_non_orig = counters.get("skip_non_origami", 0)
        if skip_non_orig:
            print(
                f"  {C.GREEN}skip_non_origami: {_fmt_int(skip_non_orig):>13s}  "
                f"({_pct(skip_non_orig, max(skip_total, 1))} of skip rows -- "
                f"GridBased/FreeSize, excluded by design){C.ENDC}",
                flush=True,
            )
        # Don't attribute a cause here. A fixed label such as "rank beyond
        # origami's ranked count" reads as a diagnosis and can mislead an
        # investigation; unresolved identity also covers an unsolved rank
        # offset and a .dat index miss.
        print(
            f"  {C.YELLOW}skip_blank     : {_fmt_int(skip_blank):>14s}  "
            f"({_pct(skip_blank, max(skip_total, 1))} -- identity "
            f"unresolved){C.ENDC}",
            flush=True,
        )
        if _COLLAPSE_STATS["rows_dropped"]:
            # Visible on purpose: this is the TemporalHint triple collapse. If
            # rows_dropped is ~2/3 of what the sweep measured, the library is
            # shipping each kernel 3x (gfx1250 Origami base + tha1 + thb1) and
            # the model is now seeing ONE row per signature instead of three
            # contradictory ones. Zero here on a library with unique signatures
            # (e.g. every gfx950 one) is expected, not a failure.
            print(
                f"  {C.CYAN}dup_rows_merged: "
                f"{_fmt_int(_COLLAPSE_STATS['rows_dropped']):>14s}  "
                f"(equivalent-signature kernels collapsed to their MEDIAN "
                f"across {_fmt_int(_COLLAPSE_STATS['problems'])} problems)"
                f"{C.ENDC}",
                flush=True,
            )
        if enricher._mt_mismatches:
            print(
                f"  {C.YELLOW}MT mismatches  : "
                f"{_fmt_int(enricher._mt_mismatches):>14s}  "
                f"(log sol_name vs .dat){C.ENDC}",
                flush=True,
            )
        print(
            f"  {C.CYAN}elapsed        : {_fmt_dur(elapsed):>14s}{C.ENDC}", flush=True
        )
        print(
            f"  {C.CYAN}avg gemm/s     : "
            f"{(total_gemms_done/elapsed) if elapsed > 0 else 0:>14.1f}"
            f"{C.ENDC}",
            flush=True,
        )
        print(flush=True)


if __name__ == "__main__":
    main()
