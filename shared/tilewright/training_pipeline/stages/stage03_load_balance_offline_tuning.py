# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03_load_balance_offline_tuning -- self-contained per-GPU bench
launcher with a dynamic work queue + per-block crash recovery.

Naming, on-disk layout, and behaviour all mirror
`benchmark_load_balancer.py` from origami_fine_grained_heuristics so
the two are interchangeable from a user-experience point of view.

Reads ONE bench yaml -- preferably `<in-dir>/shapes_probed.yaml`
(stage02's calibrated output), otherwise falls back to
`<in-dir>/shapes.yaml`. Splits it into many small BLOCKS in memory,
drops them into a queue.Queue, and spawns one worker thread per GPU.
Workers pull the next block off the queue when they finish their
current one -- so faster GPUs naturally take more blocks (dynamic
load balancing) and all GPUs finish at nearly the same time (no
fat-tail wall-clock).

On a per-block crash (hipblaslt-bench non-zero exit, e.g. illegal
memory access on one bad shape):

  1. Count successful GEMMs from `    --kernel name:` markers in the
     captured stdout.
  2. Identify the crashing shape (the one right after the last
     successful GEMM).
  3. Skip it and resume the block with the rest of the shapes.
  4. Repeat until the block fully drains or every shape is either
     successful or skipped.

Outputs in `<out-dir>/`:

  logs/block_NNNN_linesX-Y_gpuD.log     one log per block, named with
                                        the block id, its 1-based
                                        source-yaml line range, and
                                        the GPU that ran it.
                                        (No bench_split/ on disk --
                                        block yamls live in temp files
                                        that are deleted post-bench.)

  logs/warmup_NNNN_gpuD.log             the DISCARDED warm-up shape's
                                        output (see `render_warmup_line`).
                                        Deliberately NOT named `block_*` /
                                        `line*`: stage04 globs exactly
                                        `f.startswith("block_") or
                                        f.startswith("line")`, so this
                                        prefix keeps warm-up rows out of
                                        the training set.

Skipped GEMMs (the ones hipblaslt-bench crashed on and got re-run-around)
are appended to:

  skipped_gemms.txt                     one CSV line per skipped GEMM:
                                          yaml_line,M,N,K,B,rc

(Only created if at least one GEMM crashed.)

CLI:

  python stage03_load_balance_offline_tuning.py \\
      --in-dir       <run>/round_<R>/stage02 \\
      --out-dir      <run>/round_<R>/stage03 \\
      --bench-binary <hipblaslt build>/clients/hipblaslt-bench \\
      --devices 0,1,2,3,4,5,6,7 \\
      [--blocks-per-gpu 20]          # default: target ~20 blocks/GPU
      [--no-warmup-pass]             # disable the per-block warm-up shape
      [--config-yaml <run cfg>]      # read bench.warmup_pass from it
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_V3 = THIS_DIR.parent
if str(PIPELINE_V3) not in sys.path:
    sys.path.insert(0, str(PIPELINE_V3))

from lib import ui  # noqa: E402


# ── work block ───────────────────────────────────────────────────────────────


@dataclass
class TaskBlock:
    block_id: int
    start_line: int  # 1-based line number in the source yaml
    end_line: int  # inclusive 1-based
    lines: List[str]  # bench yaml lines for this block
    # Results -- filled in by the worker:
    device: Optional[int] = None
    wall_s: float = 0.0
    rc: int = -1
    n_gemms: int = 0  # total shapes attempted
    n_successful: int = 0  # GEMMs that produced output before any crash
    n_skipped: int = 0  # GEMMs that crashed hipblaslt-bench
    skipped: List[dict] = field(default_factory=list)


def _read_shape_lines(in_yaml: Path) -> List[str]:
    out: List[str] = []
    with in_yaml.open() as f:
        for line in f:
            if line.strip().startswith("-"):
                out.append(line if line.endswith("\n") else line + "\n")
    return out


_MNKB_RE = re.compile(r"M:\s*(\d+).*?N:\s*(\d+).*?K:\s*(\d+).*?batch_count:\s*(\d+)")


def _shape_tuple_from_line(line: str):
    m = _MNKB_RE.search(line)
    return (
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        if m
        else None
    )


def _build_blocks(lines: List[str], block_size: int) -> List[TaskBlock]:
    """Slice `lines` into TaskBlocks of up to `block_size` lines each.
    No disk I/O -- block yamls are materialised in temp files inside
    `_run_block_with_recovery` and removed after the run."""
    blocks: List[TaskBlock] = []
    for bi, start in enumerate(range(0, len(lines), block_size)):
        chunk = lines[start : start + block_size]
        blocks.append(
            TaskBlock(
                block_id=bi,
                start_line=start + 1,
                end_line=start + len(chunk),
                lines=chunk,
                n_gemms=len(chunk),
            )
        )
    return blocks


# ── crash-recovery bench runner (per block, per device) ──────────────────────

_KERNEL_NAME_MARKER = "    --kernel name:"

# Per-SHAPE completion marker. hipblaslt-bench prints "Winner:" once, AFTER it
# has benched every candidate for a shape.
#
# The crash-recovery below must count SHAPES COMPLETED, not kernels. Counting
# `_KERNEL_NAME_MARKER` is wrong: it is one line per CANDIDATE, and under
# stage03's `requested_solution_num: -1` a single shape emits one per candidate
# kernel, overcounting shapes by the size of the candidate set. The count then
# always clamps to the block size, `crash_idx >= n_total` is always true, and
# EVERY crash is misclassified as a benign teardown fault -- so a block is
# marked fully complete having benched only a handful of shapes, and reports
# skipped_gemms=0. That silently truncates the training set, which is far
# worse than a visible failure.
#
# (The old marker was correct for the rsn=1 probe, where kernels == shapes; it
# was never right for the full sweep.)
_SHAPE_DONE_MARKER = "Winner:"


# Markers that mean the GPU queue is DEAD and the process will never make
# progress again. Detecting these lets us kill in seconds instead of waiting out
# the stall timeout, which every faulting shape would otherwise pay in full as
# pure waiting.
#
# Verified discriminating on real logs: absent from healthy blocks, repeated in
# every stalled one.
#
# DELIBERATELY NOT fatal: "hipModuleUnload failed: ... illegal memory access".
# That fires during TEARDOWN on some large/large-batch shapes AFTER every GEMM
# has been benched and logged, and the recovery path below already treats it as
# success. Killing on it would throw away a complete block's data.
# Consecutive sub-runs producing zero output before a block is abandoned.
# Low on purpose: the failure this guards against is self-amplifying.
def _log_has_marker(path, marker: str, since: int = 0) -> bool:
    """True once `marker` appears in `path` AT OR AFTER byte offset `since`.

    `since` is not optional in practice: stage03 APPENDS every sub-run to the
    same log, so scanning from byte 0 finds winners produced by a PREVIOUS
    sub-run. A respawned bench would then be treated as "already producing
    output" and held to the short stall timeout while it was still doing its
    lengthy lazy kernel-module load -- so every fault recovery would be killed
    mid-reload and respawned into the same trap. Pass the log size captured at
    sub-run start."""
    try:
        with open(path, "r", errors="ignore") as fh:
            if since:
                fh.seek(since)
            for line in fh:
                if marker in line:
                    return True
    except OSError:
        pass
    return False


_MAX_ZERO_PROGRESS_SUBRUNS = 3

_FATAL_GPU_MARKERS = (
    "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
    "aborting with error",
)


def _log_has_fatal_gpu_error(path, since: int = 0) -> bool:
    """True when `path` shows an unrecoverable queue abort AT OR AFTER byte
    offset `since`.

    `since` is mandatory in spirit: after a fault is recovered, the marker from
    the PREVIOUS sub-run stays in the file forever. Scanning the whole file (or
    a fixed tail) would make the very next sub-run test positive the instant it
    starts, kill it, and loop -- skipping every remaining shape in the block.
    Callers pass the log size captured just before launching the sub-run."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, int(since)))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(m in chunk for m in _FATAL_GPU_MARKERS)


# ── warm-up pass ─────────────────────────────────────────────────────────────
#
# WHY IT EXISTS. Every block is a FRESH hipblaslt-bench process, and Tensile
# kernel modules load LAZILY -- on a candidate's first use. On the block's
# FIRST shape that means `skip_slow_solution_ratio` (which decides from ONE
# cold call: drop when `warm_up * ratio > best_warm_up_seen`) is comparing
# MODULE LOAD TIMES, not kernel speeds. The bench logs each prune as
#
#   Skip solution: <i> (best warm-up = <us> us , warm-up = <us> us,
#                       skip ratio = <ratio>)
#
# and on a block's first shape both readings -- even the reference `best
# warm-up` -- can include a module load that is orders of magnitude longer
# than any real kernel time. Genuinely fast kernels get pruned for being slow
# to LOAD. Shapes 2..N in the same process are clean, because by then every
# module is resident.
#
# THE FIX. PREPEND one throwaway shape to the block's yaml so the cost of
# lazy loading is paid by a shape whose numbers nobody keeps. It has to be in
# the SAME PROCESS as the real shapes -- module residency does not survive
# process exit, so a separate warm-up invocation would buy nothing. Hence the
# prepend-and-split design below rather than a second bench call.
#
# The warm-up reuses the block's own first shape: same dtype/layout/scale mode,
# therefore the same candidate set gets loaded. It is stamped:
#   requested_solution_num: -1   REQUIRED -- it must touch EVERY candidate;
#                                that is the entire point of the pass.
#   skip_slow_solution_ratio: 0  no pruning -- a contaminated screen would
#                                suppress the very loads we are forcing.
#   iters/cold_iters: 1, rotating: 0, print_kernel_info: 0, adaptive dropped
#                                cheapest timing available; we want the loads,
#                                not the measurements.
#
# `adaptive` and iters/cold_iters are mutually exclusive in the bench ("Not
# compatible with --iters / --cold_iters"), so the adaptive keys are REMOVED
# rather than left alongside the iter knobs.
_WARMUP_DROP_KEYS = frozenset(
    (
        "adaptive",
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
)

_WARMUP_FORCE = (
    ("requested_solution_num", "-1"),
    ("skip_slow_solution_ratio", "0"),
    ("iters", "1"),
    ("cold_iters", "1"),
    ("rotating", "0"),
    ("print_kernel_info", "0"),
)


def render_warmup_line(line: str) -> Optional[str]:
    """Return `line` restamped as a throwaway warm-up shape, or None if it
    is not a flow-mapping bench line we can rewrite (caller then skips the
    warm-up rather than feeding the bench something malformed).

    Bench lines are a single-level yaml flow mapping --
    `- {function: matmul, ..., batch_count: 1, iters: 50, ...}` -- with no
    nested braces and no commas inside values, so a split on "," is exact.
    Key ORDER is preserved; forced keys not already present are appended."""
    s = line.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    forced = dict(_WARMUP_FORCE)
    kept: List[tuple] = []
    seen = set()
    for part in s[i + 1 : j].split(","):
        k, sep, v = part.partition(":")
        if not sep:
            return None
        k = k.strip()
        if k in _WARMUP_DROP_KEYS:
            continue
        if k in forced:
            v = forced[k]
            seen.add(k)
        kept.append((k, v.strip()))
    for k, v in _WARMUP_FORCE:
        if k not in seen:
            kept.append((k, v))
    return s[:i] + "{" + ", ".join(f"{k}: {v}" for k, v in kept) + "}"


# A hipblaslt-bench GEMM's output opens with a short preamble --
#     Rotating buffer <N> MiB. Needed Size: <N> MiB. Needed block count: <N>
#     Adaptive timing: warmup <T>ms. ...
#     Is supported <N> / Total solutions: <N>
# -- and stage04 uses the `Rotating buffer` line as its per-problem flush
# boundary (stage04_convert_to_enriched_dataset.py, `s.startswith("Rotating
# buffer")`). So splitting the stream at the SECOND preamble group leaves the
# block log starting exactly where stage04 expects a new problem to start.
# The other two markers are fallbacks in case a config turns rotation or
# adaptive timing off; consecutive markers collapse into ONE group.
_SHAPE_PREAMBLE_MARKERS = ("Rotating buffer", "Adaptive timing:", "Is supported")


def _stream_split(proc_stdout, warmup_f, block_f) -> float:
    """Pump the bench's stdout, sending the FIRST shape's output to
    `warmup_f` and everything from the second shape on to `block_f`.

    The per-process header (hipBLASLt version / device banner) precedes the
    first shape and is written to BOTH, so the block log stays a complete,
    self-describing record even though its first shape's output was discarded.

    Returns the wall seconds spent before the switch, i.e. what the warm-up
    actually cost. Reads with readline() rather than iterating the file object
    so a slow block still streams line-by-line (the read-ahead buffer of
    `for line in f` was what made an earlier stage03 look hung)."""
    t0 = time.time()
    groups = 0  # completed shape-preamble groups seen so far
    in_preamble = False
    warmup_s = 0.0
    while True:
        line = proc_stdout.readline()
        if not line:
            break
        if line.startswith(_SHAPE_PREAMBLE_MARKERS):
            if not in_preamble:
                in_preamble = True
                groups += 1
                if groups == 2:
                    warmup_s = time.time() - t0
        elif line.startswith("["):
            in_preamble = False
        if groups == 0:
            warmup_f.write(line)  # process header -> both logs
            block_f.write(line)
        elif groups == 1:
            warmup_f.write(line)  # the discarded warm-up shape
        else:
            block_f.write(line)  # the real shapes
    if groups < 2:
        # The bench never reached a second shape (crash during warm-up, or a
        # single-shape sub-run). warmup_s stays 0.0 and the block log holds
        # only the header -> zero real GEMMs counted, which is exactly what
        # the crash-recovery bookkeeping should conclude.
        warmup_s = time.time() - t0
    return warmup_s


def _annotate(log_path: Path, msg: str) -> None:
    """Append a pipeline annotation (sub-run boundary, skipped GEMM, abort)
    to the block log. Written immediately so the log stays a faithful,
    in-order record even if the process is killed mid-block."""
    try:
        with log_path.open("a") as f:
            f.write(msg)
    except OSError:
        pass


# ── stall watchdog ───────────────────────────────────────────────────────────
#
# A GPU memory fault does NOT reliably kill hipblaslt-bench. When a candidate
# kernel faults, the runtime prints
#     HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION ... Callback: Queue ... aborting
# and then the PROCESS CAN HANG instead of exiting. (The error may name
# `flush_icache` as the kernel, but that kernel is pure `s_icache_inv`/`s_nop`
# and touches no memory -- it is simply what happened to be in flight, so the
# real faulting kernel is one of the GEMMs.)
#
# Without a watchdog those blocks sit alive forever, producing no shapes while
# the other GPUs finish whole blocks. `last_rc` never arrives, so the
# crash-recovery path below -- which is perfectly capable of skipping the
# offending shape -- is never reached.
#
# We watch the log for GROWTH rather than imposing a fixed wall-clock budget: a
# healthy block can legitimately run for a long time, but it always emits
# output. No growth for `stall_s` means the process is wedged.


def _wait_with_stall_watchdog(
    proc,
    probe_size,
    stall_s: float,
    poll_s: float = 15.0,
    probe_fatal=None,
    startup_grace_s: float = 0.0,
    probe_progress=None,
):
    """Wait for `proc`, killing it if `probe_size()` stops growing for
    `stall_s` seconds, or as soon as `probe_fatal()` reports an unrecoverable
    GPU queue abort. Returns (returncode, reason) where reason is None on a
    normal exit, "fatal" when a queue abort was seen in the log (fast path),
    or "stall" when the log simply stopped growing. Naming the mechanism
    matters: without it both kills logged the same line and there was no way to
    tell whether the fast path was working or every fault was silently paying
    the full stall budget.

    `startup_grace_s` is the budget for the FIRST byte of output, which is a
    different quantity from `stall_s` and must not be conflated with it. A
    freshly spawned hipblaslt-bench emits nothing while it lazily loads kernel
    modules, which can take longer than the stall timeout -- and longer still
    when several GPUs load at once. Holding startup to `stall_s` therefore
    kills every process before it can print its first line, marks an innocent
    GEMM as crashed, and respawns -- which pays the module load again and is
    killed again, so even tiny shapes get skipped without any real fault.
    Worse, a bench killed mid-hipModuleLoad lands in uninterruptible D state
    holding GPU resources and cannot be reaped, so the stuck processes pile
    up. That is a GPU-wedging cascade, not a safety net.

    So: no output yet -> allow startup_grace_s. Output seen -> the process is
    alive and stall_s applies to gaps between writes."""
    if stall_s <= 0:
        return proc.wait(), None
    grace_s = max(float(startup_grace_s or 0.0), stall_s)
    started = time.time()
    seen_output = False
    last_size = -1
    last_change = time.time()
    while True:
        try:
            rc = proc.wait(timeout=poll_s)
            return rc, None
        except subprocess.TimeoutExpired:
            pass
        # Fast path: the fault is already in the log, so do not wait out the
        # full stall budget for a process that can never recover.
        if probe_fatal is not None:
            try:
                if probe_fatal():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        rc = proc.wait(timeout=30)
                    except Exception:
                        rc = -9
                    return (rc if rc not in (None, 0) else -9), "fatal"
            except Exception:
                pass
        try:
            sz = probe_size()
        except Exception:
            sz = last_size
        now = time.time()
        if not seen_output and probe_progress is not None:
            # "Started working" means a SHAPE COMPLETED, not merely that some
            # bytes appeared. hipBLASLt prints "Is supported N" and only THEN
            # lazily hipModuleLoads the N candidate kernels on first launch --
            # silently, and for longer the larger the code object. Treating
            # the first byte as "running" switches the watchdog to the short
            # stall timeout while the process is still loading, so it gets
            # killed mid-load: every block dead at offset 0 with no winners.
            try:
                if probe_progress():
                    seen_output = True
            except Exception:
                pass
        if sz != last_size:
            last_size, last_change = sz, now
            continue
        # Before the first byte the process is loading kernel modules, not
        # stalling: hold it to the (larger) startup grace measured from spawn.
        if not seen_output:
            if now - started >= grace_s:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=30)
                except Exception:
                    rc = -9
                return (rc if rc not in (None, 0) else -9), "stall"
            continue
        if now - last_change >= stall_s:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=30)
            except Exception:
                rc = -9
            return (rc if rc not in (None, 0) else -9), "stall"


def _run_block_with_recovery(
    block: TaskBlock,
    device: int,
    bench_binary: Path,
    log_path: Path,
    warmup: bool = False,
    warmup_log_path: Optional[Path] = None,
    on_warmup: Optional[Callable[[int, float], None]] = None,
    stall_s: float = 900.0,
    startup_grace_s: float = 900.0,
) -> None:
    """Execute one TaskBlock on `device`. If hipblaslt-bench crashes
    mid-block, identify the crashing shape from the stdout markers,
    skip it, and re-run with the remaining shapes. Loop until done.
    Mirrors benchmark_load_balancer.py's in-block recovery.

    When `warmup` is set, each sub-run's yaml gets ONE throwaway shape
    prepended (see `render_warmup_line`) so the block's real shapes are
    screened against RESIDENT kernel modules. Its output is split off into
    `warmup_log_path` and never reaches `log_path` -- so stage04, which
    globs `block_*` / `line*`, never sees it. The warm-up is re-run on every
    sub-run because a sub-run is a NEW process with an empty module cache.
    `on_warmup(subrun, seconds)` is called after each warm-up completes."""
    block.device = device
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = block.lines
    n_total = len(lines)
    if warmup and (warmup_log_path is None or n_total < 2):
        # < 2 shapes: the "warm-up" would be the only shape, so there is
        # nothing left for it to warm up. Nothing to gain, a module load to lose.
        warmup = False
    log_buf: List[str] = []
    next_offset = 0
    subrun = 0
    zero_progress_subruns = 0
    completed = 0
    last_rc = 0
    t0 = time.time()
    while next_offset < n_total:
        remaining = lines[next_offset:]
        warmup_line = (
            render_warmup_line(remaining[0])
            if (warmup and len(remaining) >= 2)
            else None
        )
        # Write the remaining shapes to a temp file (deleted after), with the
        # warm-up shape PREPENDED so it loads the modules in this very process.
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            tmp_yaml = Path(f.name)
            if warmup_line:
                f.write(warmup_line + "\n")
            for ln in remaining:
                f.write(ln)
        if subrun > 0:
            _annotate(
                log_path,
                f"\n*** SUB-RUN #{subrun + 1}: resume at "
                f"block-offset {next_offset} with "
                f"{len(remaining)} GEMMs remaining ***\n",
            )
        # Byte offset where THIS sub-run's output starts. The log is appended
        # across sub-runs, so the "has it completed a shape yet" probe must not
        # see a previous sub-run's winners (that made every fault recovery look
        # like a running process and got it killed during its module reload).
        try:
            _subrun_start = log_path.stat().st_size
        except OSError:
            _subrun_start = 0
        try:
            from lib.bench_env import bench_env_with  # noqa: E402

            # Stream to the log file instead of buffering the whole block.
            # Two reasons this matters:
            #   * VISIBILITY. With stdout=PIPE the block_*.log does not exist
            #     until the block completes, so a stage03 that is merely slow
            #     is indistinguishable from one that is hung (an empty logs/
            #     dir and blocks_done=0 for the whole run).
            #   * MEMORY. A full rsn=-1 sweep is one kernel record per
            #     candidate per shape; over a whole block that is gigabytes
            #     held in a list.
            # We still keep the text for crash analysis, but read it back from
            # the file per sub-run rather than accumulating every sub-run.
            sub_start = log_path.stat().st_size if log_path.exists() else 0
            _warm_start = (
                warmup_log_path.stat().st_size
                if (warmup_log_path and warmup_log_path.exists())
                else 0
            )
            if warmup_line:
                # stdout has to come back through a PIPE here: it is ONE
                # process producing two logs, so the split cannot be done with
                # a plain fd redirect. `_stream_split` still writes every line
                # as it arrives, so the visibility property is preserved.
                with warmup_log_path.open("a", buffering=1) as wf, log_path.open(
                    "a", buffering=1
                ) as lf:
                    wf.write(
                        f"*** WARM-UP (discarded) sub-run #{subrun + 1}, "
                        f"block {block.block_id}, gpu {device} ***\n"
                    )
                    proc = subprocess.Popen(
                        [
                            str(bench_binary),
                            "--device",
                            str(device),
                            "--yaml",
                            str(tmp_yaml),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=bench_env_with(),
                    )
                    # _stream_split BLOCKS until stdout closes, so it cannot
                    # be called on this thread: the hang we are guarding
                    # against happens WHILE the warm-up is streaming (a
                    # faulting kernel wedges the process with the pipe still
                    # open), and the watchdog would never get to run. Drain on
                    # a worker thread and watch the process from here.
                    _warm = {"s": 0.0}

                    def _drain():
                        try:
                            _warm["s"] = _stream_split(proc.stdout, wf, lf)
                        except Exception:
                            pass

                    _t = threading.Thread(target=_drain, daemon=True)
                    _t.start()
                    last_rc, _stalled = _wait_with_stall_watchdog(
                        proc,
                        lambda: (
                            log_path.stat().st_size + warmup_log_path.stat().st_size
                        ),
                        stall_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=lambda: _log_has_marker(
                            log_path, _SHAPE_DONE_MARKER, since=_subrun_start
                        ),
                        probe_fatal=lambda: (
                            _log_has_fatal_gpu_error(warmup_log_path, _warm_start)
                            or _log_has_fatal_gpu_error(log_path, sub_start)
                        ),
                    )
                    _t.join(timeout=30)
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                    warm_s = _warm["s"]
                    if _stalled:
                        _annotate(
                            log_path,
                            f"\n*** KILLED during WARM-UP [{_stalled}]: no output for"
                            f" {stall_s:.0f}s -- killed. The warm-up shape"
                            f" is a copy of this block's first GEMM, so"
                            f" the crash-recovery below will skip it and"
                            f" resume. ***\n",
                        )
                if on_warmup is not None:
                    on_warmup(subrun + 1, warm_s)
            else:
                with log_path.open("a", buffering=1) as lf:
                    proc = subprocess.Popen(
                        [
                            str(bench_binary),
                            "--device",
                            str(device),
                            "--yaml",
                            str(tmp_yaml),
                        ],
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=bench_env_with(),
                    )
                    last_rc, _stalled = _wait_with_stall_watchdog(
                        proc,
                        lambda: log_path.stat().st_size,
                        stall_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=lambda: _log_has_marker(
                            log_path, _SHAPE_DONE_MARKER, since=_subrun_start
                        ),
                        probe_fatal=lambda: _log_has_fatal_gpu_error(
                            log_path, sub_start
                        ),
                    )
                    if _stalled:
                        _annotate(
                            log_path,
                            f"\n*** KILLED [{_stalled}]: "
                            + (
                                "unrecoverable GPU queue abort seen in the log"
                                if _stalled == "fatal"
                                else f"no output for {stall_s:.0f}s"
                            )
                            + " -- treating as a crash so the offending"
                            " GEMM is skipped ***\n",
                        )
            # Re-read only THIS sub-run's output for the crash-recovery
            # bookkeeping below (which counts kernel markers).
            with log_path.open("r", errors="replace") as rf:
                rf.seek(sub_start)
                log_buf.append(rf.read())
        finally:
            try:
                tmp_yaml.unlink()
            except OSError:
                pass
        subrun += 1
        if last_rc == 0:
            completed += len(remaining)
            next_offset = n_total
            break
        # Sub-run crashed: count GEMMs that finished before the crash.
        out_text = log_buf[-1] if log_buf else ""
        ok_in_subrun = out_text.count(_SHAPE_DONE_MARKER)
        if ok_in_subrun > len(remaining):
            ok_in_subrun = len(remaining)
        completed += ok_in_subrun
        # A sub-run that produced NOTHING was killed before it finished loading
        # kernel modules -- respawning just pays the load and dies again, and
        # each corpse is stuck in uninterruptible D state holding GPU
        # resources. Three of those in a row means the block is unrunnable on
        # this device right now; grinding on wedges the GPU (unreapable
        # processes accumulate until the node needs a reboot).
        zero_progress_subruns = (zero_progress_subruns + 1) if ok_in_subrun == 0 else 0
        crash_idx = next_offset + ok_in_subrun
        if crash_idx >= n_total:
            # Process exited non-zero but every remaining shape produced
            # output -> the fault was during TEARDOWN (e.g. the
            # `hipModuleUnload: illegal memory access (700)` we see on some
            # large / large-batch shapes), AFTER every GEMM in this block
            # was benched and its timing captured to the log. No data is
            # lost, so treat it as SUCCESS -- don't let a benign teardown
            # fault count as a crash and trip the round's tolerance gate.
            last_rc = 0
            completed = n_total
            break
        crash_line = lines[crash_idx]
        shape = _shape_tuple_from_line(crash_line)
        yaml_line = block.start_line + crash_idx
        _annotate(
            log_path,
            f"\n*** SKIPPED CRASHED GEMM at block-offset "
            f"{crash_idx} (yaml line {yaml_line}): "
            f"M={shape[0] if shape else '?'} "
            f"N={shape[1] if shape else '?'} "
            f"K={shape[2] if shape else '?'} "
            f"(rc={last_rc}) ***\n",
        )
        block.skipped.append(
            {
                "yaml_line": yaml_line,
                "M": (shape[0] if shape else None),
                "N": (shape[1] if shape else None),
                "K": (shape[2] if shape else None),
                "B": (shape[3] if shape else None),
                "rc": last_rc,
                "line": crash_line,  # full bench line, for the end-of-run retry
            }
        )
        next_offset = crash_idx + 1
        if zero_progress_subruns >= _MAX_ZERO_PROGRESS_SUBRUNS:
            _annotate(
                log_path,
                f"\n*** ABORT: {zero_progress_subruns} consecutive sub-runs "
                f"produced no output (kernel-module load being killed, or "
                f"the device is wedged). Giving up on this block instead of "
                f"respawning -- see startup_grace_s. ***\n",
            )
            break
        if subrun > n_total + 1:
            _annotate(
                log_path,
                f"\n*** ABORT: too many sub-runs ({subrun}); " f"giving up ***\n",
            )
            break
    block.wall_s = time.time() - t0
    block.rc = last_rc
    block.n_successful = completed
    block.n_skipped = len(block.skipped)
    # NOTE: the bench output is already in `log_path` -- it was streamed there
    # as each sub-run ran. Do NOT rewrite the file from `log_buf` here: that
    # used to be the only write, but now it would truncate the streamed log
    # and then duplicate it. Annotations are appended inline via `_annotate`.


# ── device worker ────────────────────────────────────────────────────────────


class _DeviceWorker(threading.Thread):
    """Pull blocks off `q`, run each one on `device`, append the
    finished TaskBlock to `done`."""

    def __init__(
        self,
        device: int,
        q: queue.Queue,
        done: List[TaskBlock],
        done_lock: threading.Lock,
        bench_binary: Path,
        logs_dir: Path,
        warmup: bool = True,
        quiet: bool = False,
    ) -> None:
        super().__init__(name=f"bench-d{device}", daemon=True)
        self.device = device
        self.q = q
        self.done = done
        self.done_lock = done_lock
        self.bench_binary = bench_binary
        self.logs_dir = logs_dir
        self.warmup = warmup
        self.quiet = quiet
        self.n_blocks_completed = 0
        self.wall_s = 0.0
        self.stall_s = 300.0
        self.startup_grace_s = 900.0
        self.warmup_s = 0.0  # total warm-up seconds paid on this GPU

    def run(self) -> None:
        t0 = time.time()
        while True:
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                break
            # Filename mirrors benchmark_load_balancer.py exactly:
            #   block_NNNN_linesX-Y_gpuD.log
            log_path = self.logs_dir / (
                f"block_{block.block_id:04d}_"
                f"lines{block.start_line}-{block.end_line}_"
                f"gpu{self.device}.log"
            )
            # NOTE the `warmup_` prefix: stage04 globs
            # `f.startswith("block_") or f.startswith("line")`, so this file
            # is invisible to it and the discarded rows can never become
            # training data.
            warmup_log_path = self.logs_dir / (
                f"warmup_{block.block_id:04d}_gpu{self.device}.log"
            )

            def _on_warmup(
                subrun: int, secs: float, _b=block, _w=warmup_log_path
            ) -> None:
                self.warmup_s += secs
                if not self.quiet:
                    ui.grey(
                        "warm",
                        f"gpu{self.device} block {_b.block_id:04d} "
                        f"sub-run #{subrun}: warm-up loaded all "
                        f"candidates in {ui.fmt_dur(secs)} "
                        f"(discarded -> {_w.name})",
                    )

            try:
                _run_block_with_recovery(
                    block,
                    self.device,
                    self.bench_binary,
                    log_path,
                    warmup=self.warmup,
                    warmup_log_path=warmup_log_path,
                    stall_s=self.stall_s,
                    startup_grace_s=self.startup_grace_s,
                    on_warmup=_on_warmup,
                )
            except Exception as exc:
                block.rc = -1
                with log_path.open("a") as f:
                    f.write(f"\n*** ERROR: {exc} ***\n")
            with self.done_lock:
                self.done.append(block)
            self.q.task_done()
            self.n_blocks_completed += 1
        self.wall_s = time.time() - t0


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage03 -- self-contained per-GPU bench launcher "
        "(work queue + crash recovery)."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="dir from stage02 (shapes_probed.yaml) or "
        "stage01 (shapes.yaml) as fallback.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    ap.add_argument(
        "--startup-grace-s",
        type=float,
        default=900.0,
        help="Budget for the FIRST byte of output from a freshly "
        "spawned bench, which is kernel-module load time and "
        "NOT a stall. That load can exceed the stall timeout, "
        "more so with several GPUs loading at once, so using "
        "the stall timeout here kills every process before it "
        "prints, skips an innocent GEMM, and respawns into "
        "the same trap -- leaving unreapable D-state "
        "processes behind. "
        "Must exceed the CONTENDED module-load time.",
    )
    ap.add_argument(
        "--stall-timeout-s",
        type=float,
        default=300.0,
        help="Kill a bench sub-run whose log has not grown for "
        "this many seconds and treat it as a crash, so the "
        "offending GEMM is skipped. A GPU memory fault does "
        "NOT reliably kill hipblaslt-bench -- it can wedge "
        "with the pipe open and hold the device forever. A "
        "healthy block streams output continuously, so a "
        "long silence is always abnormal; this is a bound on "
        "the cost of each fault, not a budget for real work. "
        "0 disables the watchdog.",
    )
    ap.add_argument(
        "--blocks-per-gpu",
        type=int,
        default=20,
        help="target number of blocks per GPU; block_size "
        "= ceil(n_shapes / (n_gpus * blocks_per_gpu)). "
        "Smaller blocks => finer dynamic balancing.",
    )
    ap.add_argument("--progress-every", type=float, default=30.0)
    # Warm-up pass (see the `render_warmup_line` block above). Default ON;
    # `bench.warmup_pass: false` in the run config, or --no-warmup-pass on the
    # command line, turns it off. The CLI flag wins over the config.
    ap.add_argument(
        "--warmup-pass",
        dest="warmup_pass",
        action="store_true",
        default=None,
        help="prepend one discarded warm-up shape per block so "
        "skip_slow_solution_ratio screens RESIDENT kernels "
        "(default: on, or bench.warmup_pass from "
        "--config-yaml)",
    )
    ap.add_argument(
        "--no-warmup-pass",
        dest="warmup_pass",
        action="store_false",
        help="disable the per-block warm-up shape",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config; only `bench.warmup_pass` is read from "
        "it, and only when neither --warmup-pass nor "
        "--no-warmup-pass was given",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # bench.warmup_pass resolution: CLI flag > config > default true.
    warmup = True
    warmup_src = "default"
    if args.warmup_pass is not None:
        warmup = bool(args.warmup_pass)
        warmup_src = "cli"
    elif args.config_yaml is not None and args.config_yaml.exists():
        try:
            import yaml  # noqa: E402

            with args.config_yaml.open() as f:
                _cfg = yaml.safe_load(f) or {}
            _v = (_cfg.get("bench") or {}).get("warmup_pass")
            if _v is not None:
                warmup = bool(_v)
                warmup_src = "config"
        except Exception as exc:  # noqa: BLE001
            ui.warn(
                "cfg",
                f"could not read bench.warmup_pass from "
                f"{args.config_yaml}: {exc}; using default (on)",
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    devices = [int(d) for d in args.devices.split(",") if d.strip()]

    probed = args.in_dir / "shapes_probed.yaml"
    raw = args.in_dir / "shapes.yaml"
    if probed.exists():
        in_yaml = probed
    elif raw.exists():
        in_yaml = raw
        if not args.quiet:
            ui.warn(
                "yaml",
                f"shapes_probed.yaml not found; using "
                f"uncalibrated {raw.name}. Run stage02 first for "
                "stable bench measurements.",
            )
    else:
        ui.err("err", f"neither {probed} nor {raw} exists")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"hipblaslt-bench not found: {args.bench_binary}")
        return 1

    lines = _read_shape_lines(in_yaml)
    n_shapes = len(lines)
    if n_shapes == 0:
        ui.err("err", f"no shape lines in {in_yaml}")
        return 1

    # Block size: target ~blocks_per_gpu blocks per device.
    n_gpus = len(devices)
    target_blocks = max(n_gpus, n_gpus * int(args.blocks_per_gpu))
    block_size = max(1, math.ceil(n_shapes / target_blocks))
    blocks = _build_blocks(lines, block_size)

    if not args.quiet:
        ui.banner(f"stage03 bench launcher  ({len(devices)} devices)")
        ui.info("cfg", f"yaml         : {in_yaml}")
        ui.info("cfg", f"devices      : {devices}")
        ui.info("cfg", f"bench_binary : {args.bench_binary}")
        ui.info("cfg", f"out_dir      : {args.out_dir}")
        ui.info("cfg", f"n_shapes     : {n_shapes:,}")
        ui.info(
            "cfg",
            f"block_size   : {block_size}  "
            f"(target ~{args.blocks_per_gpu} blocks/GPU)",
        )
        ui.info("cfg", f"n_blocks     : {len(blocks):,}")
        ui.info(
            "cfg",
            f"warmup_pass  : {warmup}  ({warmup_src})"
            + (
                ""
                if warmup
                else "  -- WARNING: the first shape of every block will be "
                "screened on module-load times, not kernel times"
            ),
        )

    # Fill the work queue.
    q: queue.Queue = queue.Queue()
    for b in blocks:
        q.put(b)
    done: List[TaskBlock] = []
    done_lock = threading.Lock()

    if not args.quiet:
        ui.banner("Bench launch (work queue + crash recovery)", ui.C.BLUE)
    workers = [
        _DeviceWorker(
            dev,
            q,
            done,
            done_lock,
            args.bench_binary,
            logs_dir,
            warmup=warmup,
            quiet=args.quiet,
        )
        for dev in devices
    ]
    # Stall watchdog budget (see --stall-timeout-s). Set after construction so
    # the worker signature stays as upstream has it.
    for _w in workers:
        _w.stall_s = float(args.stall_timeout_s)
        _w.startup_grace_s = float(args.startup_grace_s)
    t0 = time.time()
    for w in workers:
        w.start()
    last_print = 0.0
    while any(w.is_alive() for w in workers):
        time.sleep(0.5)
        if not args.quiet and time.time() - last_print >= args.progress_every:
            with done_lock:
                n_done = len(done)
                n_skip = sum(b.n_skipped for b in done)
            alive = [w.device for w in workers if w.is_alive()]
            elapsed = time.time() - t0
            ui.grey(
                "bench",
                f"blocks_done={n_done:>4d}/{len(blocks):<4d}  "
                f"alive_gpus={alive}  "
                f"skipped_gemms={n_skip}  "
                f"elapsed={ui.fmt_dur(elapsed)}",
            )
            last_print = time.time()
    for w in workers:
        w.join()
    elapsed = time.time() - t0

    # ── Retry pass ───────────────────────────────────────────────────────
    # In-block recovery SKIPS a shape that crashes hipblaslt-bench. Those
    # crashes are typically TRANSIENT HIP faults under heavy concurrent
    # multi-GPU load (the same shape benches fine in isolation), so a single
    # lightly-loaded re-bench recovers them -- avoiding any lost data. Only
    # shapes that crash AGAIN stay skipped. A large skip count means a
    # systematic fault (bad build / dead GPU), not transient -- don't waste
    # time re-benching it; the tolerance gate below handles that.
    retried_recovered = 0
    skipped_lines = [s["line"] for b in done for s in b.skipped if s.get("line")]
    if (
        skipped_lines
        and not getattr(args, "no_retry_skipped", False)
        and len(skipped_lines) <= 200
    ):
        if not args.quiet:
            ui.banner(
                f"Retry pass: re-benching {len(skipped_lines)} "
                f"skipped GEMM(s) on device {devices[0]}",
                ui.C.BLUE,
            )
        retry_block = TaskBlock(
            block_id=99999,
            start_line=1,
            end_line=len(skipped_lines),
            lines=skipped_lines,
            n_gemms=len(skipped_lines),
        )
        # No warm-up here: the retry pass exists to recover shapes that
        # CRASHED, not to produce clean rankings, and it is a single
        # lightly-loaded process.
        _run_block_with_recovery(
            retry_block,
            devices[0],
            args.bench_binary,
            logs_dir / "retry_skipped.log",
            warmup=False,
        )
        still = {(s["M"], s["N"], s["K"], s["B"]) for s in retry_block.skipped}
        retried_recovered = len(skipped_lines) - len(retry_block.skipped)
        for b in done:  # drop now-recovered shapes from skips
            b.skipped = [
                s for s in b.skipped if (s["M"], s["N"], s["K"], s["B"]) in still
            ]
            b.n_skipped = len(b.skipped)
        if not args.quiet:
            ui.ok(
                "retry",
                f"recovered {retried_recovered}/{len(skipped_lines)} "
                f"skipped GEMM(s) on re-bench; "
                f"{len(retry_block.skipped)} still failing",
            )

    # Summary.
    n_crash_blocks = sum(1 for b in done if b.rc != 0 and b.n_skipped == 0)
    n_recovered_blocks = sum(1 for b in done if b.n_skipped > 0)
    n_skipped_gemms = sum(b.n_skipped for b in done)
    total_attempts = sum(b.n_gemms for b in done)
    total_successful = sum(b.n_successful for b in done)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN if n_crash_blocks == 0 else ui.C.YELLOW)
        for w in workers:
            ui.ok(
                "dev ",
                f"  device {w.device}  "
                f"blocks={w.n_blocks_completed:>4d}  "
                f"wall={ui.fmt_dur(w.wall_s)}  "
                f"warmup={ui.fmt_dur(w.warmup_s)}",
            )
        if warmup:
            _tot_warm = sum(w.warmup_s for w in workers)
            ui.ok(
                "done",
                f"warm-up overhead : {ui.fmt_dur(_tot_warm)} across "
                f"{len(devices)} GPU(s) -- one lazy-load pass per block "
                f"process; discarded rows are in logs/warmup_*.log",
            )
        ui.ok("done", f"total wall       : {ui.fmt_dur(elapsed)}")
        ui.ok("done", f"blocks completed : {len(done)}/{len(blocks)}")
        ui.ok("done", f"GEMMs successful : {total_successful:,}/{total_attempts:,}")
        ui.ok(
            "done",
            f"GEMMs skipped    : {n_skipped_gemms}  "
            f"(from {n_recovered_blocks} crash-recovered blocks)",
        )
        ui.ok(
            "done",
            f"unrecovered      : {n_crash_blocks} block(s) "
            f"(other than per-shape skips)",
        )
        ui.ok("done", f"logs dir         : {logs_dir}")

    # If any GEMM crashed hipblaslt-bench, drop a flat
    # skipped_gemms.txt for visibility (one CSV line per skipped GEMM).
    if n_skipped_gemms > 0:
        skip_path = args.out_dir / "skipped_gemms.txt"
        with skip_path.open("w") as f:
            f.write("yaml_line,M,N,K,B,rc\n")
            for b in sorted(done, key=lambda x: x.block_id):
                for s in b.skipped:
                    f.write(
                        f"{s['yaml_line']},{s['M']},{s['N']},"
                        f"{s['K']},{s['B']},{s['rc']}\n"
                    )
        if not args.quiet:
            ui.warn(
                "skip",
                f"{n_skipped_gemms} GEMMs hipblaslt-bench "
                f"crashed on -- see {skip_path}",
            )

    # Tolerance: a HANDFUL of blocks can hit an unrecoverable GPU fault
    # (HIP 700 "illegal memory access") on the most extreme envelope
    # shapes -- the fault poisons the device context so the in-block
    # recovery can't skip-and-continue (n_skipped stays 0). Those blocks
    # still flushed the data they benched BEFORE the fault, and they're
    # a tiny fraction of a many-thousand-shape round. Failing the entire
    # long-running pipeline on <=5% such blocks is the wrong call -- it
    # just costs a re-bench of work that's already on disk. So we return
    # success when crashed blocks are within tolerance (still LOUDLY
    # warned + recorded above), and only hard-fail on a widespread crash
    # (e.g. a bad build / dead GPU), which the fraction gate catches.
    crash_frac = n_crash_blocks / max(1, len(done))
    tol = max(int(getattr(args, "max_crash_blocks", 0) or 0), int(0.05 * len(blocks)))
    if n_crash_blocks == 0:
        return 0
    if n_crash_blocks <= tol:
        if not args.quiet:
            ui.warn(
                "tol",
                f"{n_crash_blocks} unrecovered block(s) "
                f"({crash_frac*100:.1f}%) within tolerance ({tol}); "
                f"benched data kept, continuing. (raise --max-crash-blocks "
                f"or lower lib.shapes._MAX_FLOPS if this grows)",
            )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
