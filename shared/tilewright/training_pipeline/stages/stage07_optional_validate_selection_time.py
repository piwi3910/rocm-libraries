# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage07_optional_validate_selection_time -- measure end-to-end
SELECTION TIME on real GPU AND check kernel-pick parity between the
deployed C++ runtime and the Python `models.pt` student.

(1) SELECTION TIME mirrors what `MasterSolutionLibrary::findTopSolutions`
actually times (see comment in the source there): one
`std::chrono::steady_clock`-based span per problem, printed as a
single line of the form

    Solution selection time: N us

That single line is the call's TOTAL selection time -- everything
from the problem-level `findTopSolutions` entry to the chosen
solution being returned. There's no per-stage breakdown printed by
this code path; each shape produces exactly one line per call.

(2) PICK PARITY (optional; enabled when `--train-dir` is passed)
verifies, per problem, that the kernel hipblaslt-bench's ML path
selects matches the kernel `models.pt` would select via the
deployed-pipeline scoring in Python. With TILEWRIGHT_PICK_LOG=1 the C++
ml_recommender prints one stderr line per `rank_configs` call,
e.g.

    [TILEWRIGHT_PICK] m=4096 n=4096 k=4096 b=1 leaf=Large|Large|LargeK|Bnone
             top1_sig=(mt_m=128,mt_n=256,mt_k=64,mi_m=32,mi_n=32,
                       mi_k=16,cha=0,chb=0) top1_score=0.4321
             n_configs=512

and stage 7 computes the SAME pick from `models.pt` for every
problem in the yaml (Python mirror of `ml_recommender::rank_configs`
-- see `compute_deployed_top1_picks` in stage05). Mismatch on any
problem means either the runtime didn't load the staged `.bin`,
features drifted between C++ and Python, or the rebuild used stale
weights -- exactly the deploy-time bug class this check exists to
catch (hipblaslt-bench must show that ML mode actually ran).

Toggling the ML recommender:
  ML ON  : TENSILE_PREDICTION_LIB=1 TENSILE_USE_TILEWRIGHT=1 TILEWRIGHT_PICK_LOG=1
  ML OFF : TENSILE_PREDICTION_LIB=1      (the recommender opts out
                                          unless the env var is set)
Both runs target the SAME hipblaslt-bench from the deployed
checkout. `TENSILE_DB=0x10000` enables the selection-time print on
both runs; `TILEWRIGHT_DIAG=1` makes the runtime print a one-shot weights-
load banner so a missing `.bin` is loud rather than silent.

Inputs:
  --rocm-libraries-to-deploy <repo>     rocm-libraries checkout that
                                        already has stage06_deploy
                                        deployed + was rebuilt
                                        (hipblaslt-bench AND origami).
  --hipblaslt-bench-yaml <path>         bench yaml of shapes to time.
                                        Repeatable (one timing run per
                                        yaml). Each yaml is run twice,
                                        once for ML-ON and once for
                                        ML-OFF.
  --out-dir <where>                     where to write logs +
                                        timings_summary.json.
  --train-dir <stage05>                 OPTIONAL. Path to the trained
                                        `<run>/round_<R>/stage05`. When
                                        set, stage 7 runs the pick-
                                        parity check (Python <-> C++)
                                        per problem on top of the
                                        timing comparison.
  --prior-round-dir <round>             OPTIONAL, repeatable. Round
                                        dirs whose stage04b/splits.json
                                        feeds into the parity check's
                                        cumulative split tree. Pass
                                        every prior round when the
                                        models you're validating
                                        emerged from active learning.
  [--devices D0,D1,...]                 GPU devices (default 0); only
                                        the FIRST is used.
  [--iters N]                           bench --iters (default 1 so
                                        selection time dominates).
  [--cold-iters N]                      default 0.
  [--drop-first-call] / [--no-drop-first-call]
                                        the first call to a given
                                        (problem, library) pair carries
                                        the LIBRARY-INIT cost (mmap +
                                        first-touch -- far above the
                                        steady-state cost). When
                                        true, that first call is
                                        excluded from the summary stats.
                                        Default: true.

Outputs under `--out-dir`:
  <yaml_name>/ml_on/bench.log          hipblaslt-bench stdout with
                                       TENSILE_DB=0x10000 + TILEWRIGHT_DIAG=1
                                       + TILEWRIGHT_PICK_LOG=1 enabled
  <yaml_name>/ml_off/bench.log
  <yaml_name>/parity.json              per-problem pick comparison
                                       (only when --train-dir is set)
  timings_summary.json                 per-yaml mean / p50 / p99 +
                                       ML overhead deltas + parity
                                       match rate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from lib import ui  # noqa: E402


# The Tensile code path that emits selection times is gated by
# `Debug::printSolutionSelectionTime()` (TENSILE_DB=0x10000) and uses a
# single chrono::steady_clock span around `library->findTopSolutions`,
# printing `Solution selection time: <us> us`. That single line is the
# WHOLE call's selection time -- there's no per-stage decomposition in
# this code path. Number is in microseconds; scientific notation
# unlikely but we accept it for safety.
_RE_SEL_TIME = re.compile(r"Solution selection time:\s*([\d\.eE\+\-]+)\s*us")

# Emitted by origami's `ml_recommender::diag_load` when TILEWRIGHT_DIAG=1 and
# the weights .bin was successfully loaded. Stage 7 uses this as the
# ground-truth marker that the ML recommender path is actually live
# in the bench process -- a ML-on log MISSING this line means origami
# silently fell back to its analytical scorer.
# The per-library loader prefixes the marker with `handle=N path=...`
# before `arch=` (e.g. `[TILEWRIGHT_DIAG FILE] handle=0 path=./foo.bin
# arch=...`), whereas the legacy singleton loader emits `arch=` right
# after `]`. The `.*?` tolerates that optional prefix so both formats
# are recognised as "ML weights loaded".
_RE_TILEWRIGHT_DIAG_LOAD = re.compile(
    r"\[TILEWRIGHT_DIAG\s+\S+\].*?\barch=(\S+)\s+qhash=(\S+)\s+"
    r"qdim=(\d+)\s+idim=(\d+)\s+xdim=(\d+)\s+"
    r"n_cells=(\d+)\s+n_splits=(\d+)"
)
# Origami emits this when no weights file could be opened.
_RE_ML_NO_WEIGHTS = re.compile(r"\[origami::ml_recommender\] no weights loaded")

# `TILEWRIGHT_PICK_LOG=1` makes ml_recommender::rank_configs print one
# `[TILEWRIGHT_PICK]` stderr line per problem (the SAME problem hipblaslt-bench
# is timing in the next breath). We parse it back into a structured
# dict and join against the Python deployed-pick from
# `compute_deployed_top1_picks` to produce the parity table.
#
# Format produced by the C++ side (see
#   shared/origami/src/origami/ml_recommender.cpp:rank_configs):
#   [TILEWRIGHT_PICK] m=4096 n=4096 k=4096 b=1 leaf=Large|Large|LargeK|Bnone
#            top1_sig=(mt_m=128,mt_n=256,mt_k=64,mi_m=32,mi_n=32,
#                      mi_k=16,cha=0,chb=0) top1_score=0.4321
#            n_configs=512
_RE_ML_PICK = re.compile(
    r"\[TILEWRIGHT_PICK\]\s+"
    r"m=(\d+)\s+n=(\d+)\s+k=(\d+)\s+b=(\d+)\s+"
    # `tA=` / `tB=` are emitted by the latest runtime to disambiguate
    # the per-call transpose. The group is optional so we still parse
    # older bench logs (runtimes that predate tA/tB) cleanly.
    r"(?:tA=([NT])\s+tB=([NT])\s+)?"
    r"leaf=(\S+)\s+"
    r"top1_sig=\(mt_m=(\d+),mt_n=(\d+),mt_k=(\d+),"
    r"mi_m=(\d+),mi_n=(\d+),mi_k=(\d+),"
    r"cha=(-?\d+),chb=(-?\d+)\)\s+"
    r"top1_score=([\d\.eE\+\-]+)\s+"
    r"n_configs=(\d+)"
)


def _run_bench(
    bench_binary: Path,
    yaml_path: Path,
    log_path: Path,
    device: int,
    env_extra: Dict[str, str],
    iters: int,
    cold_iters: int,
    quiet: bool,
    ml_weights_path: Optional[Path] = None,
    tensile_libpath: Optional[Path] = None,
    stall_timeout_s: float = 600.0,
    startup_grace_s: float = 1800.0,
) -> int:
    """Invoke hipblaslt-bench with TENSILE_DB=0x10000 (selection-time
    logging) on top of the pipeline-wide bench env. Tee stdout+stderr
    to log_path.

    Two extra env vars are always set so the operator can verify the
    ML recommender path is actually running (vs origami silently
    falling back to its analytical scorer):
      * `TILEWRIGHT_DIAG=1` -- makes origami's `ml_recommender::ensure_weights`
        emit one `[TILEWRIGHT_DIAG FILE] arch=... n_cells=N ...` line per
        weights load. If this line is missing from a ML-on bench
        log, the deployed library did not pick up the .bin.
      * `TILEWRIGHT_WEIGHTS=<.bin>` -- explicit override of the
        weights-search path so we don't depend on `./` or
        `/opt/rocm/share/origami/` happening to contain a copy.
        Skipped when `ml_weights_path` is None (legacy callers).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from lib.bench_env import bench_env_with  # noqa: E402

    # We deliberately KEEP the bench env minimal for the ML-OFF
    # comparison run -- if we also set `TILEWRIGHT_WEIGHTS` /
    # `TILEWRIGHT_DIAG=1` there, origami's eager-init loads the .bin and
    # prints `[TILEWRIGHT_DIAG FILE] ...` on the ML-off log too. That makes
    # the log misleading (the operator can no longer tell at a
    # glance which side is which). All ML-side env knobs flow in
    # via `env_extra` instead, so the caller for the ML-on run
    # passes them explicitly and the ML-off caller doesn't.
    extra = {"TENSILE_DB": "0x10000"}
    # F6: we no longer set `TILEWRIGHT_WEIGHTS`. The deployed runtime
    # discovers the weights through the `tilewright_index` beside the
    # Tensile library .dat when the ML path is active
    # (TENSILE_USE_TILEWRIGHT=1), so the env override is unnecessary. ML-on/ML-off is
    # selected purely by TENSILE_USE_TILEWRIGHT (set in `env_extra` by the
    # ML-on caller; absent for the ML-off baseline) -- and with the override
    # gone, the runtime's eager-init self-disables, so the ML-off baseline
    # stays clean without ML_RECOMMENDER_NO_EAGER_INIT either (F7).
    extra.update(env_extra or {})
    env = bench_env_with(extra)
    # gfx1250 builds the v0 GEMM tree into its OWN directory
    # (build/release/Tensile-gfx1250v0/library/gfx1250v0) which is NOT merged
    # under Tensile/library/ until install. Without an explicit libpath the
    # runtime picks the library by implicit promotion and can land on the v1
    # tree -- wrong kernels, and no tilewright weights (they are discovered
    # ONLY in the directory the .dat was loaded from), so the ML path silently
    # never engages and the [TILEWRIGHT_DIAG FILE] marker never appears.
    #
    # Point HIPBLASLT_TENSILE_LIBPATH at the v0 tree explicitly. NOTE the
    # distinction: aiming it at the merged PARENT (Tensile/library) is the
    # known trap that serves v1 on v0 silicon; aiming it at the concrete
    # per-arch directory is correct.
    if tensile_libpath is not None:
        env["HIPBLASLT_TENSILE_LIBPATH"] = str(tensile_libpath)
    # Drop any inherited LD_LIBRARY_PATH so the binary resolves through its own
    # RPATH. An inherited path from another checkout silently loads that
    # checkout's libhipblaslt -- the failure is invisible because it succeeds.
    env.pop("LD_LIBRARY_PATH", None)
    # NB: do NOT pass `--use_gpu_timer 1` as a CLI arg -- the deploy
    # build's hipblaslt-bench treats `--use_gpu_timer` as a BOOL
    # flag (no value), so the `1` would be parsed as a stray
    # positional ("Option 1 is not defined"). The bench yaml carries
    # `use_gpu_timer: 1` per problem line, which is the right place
    # to set it.
    cmd = [
        str(bench_binary),
        "--yaml",
        str(yaml_path),
        "--device",
        str(device),
        "--iters",
        str(iters),
        "--cold_iters",
        str(cold_iters),
    ]
    if not quiet:
        ui.info("run", " ".join(cmd))
        # Only the bench-specific extras are interesting; printing the
        # whole env would be noise.
        ui.info("env", "  " + " ".join(f"{k}={v}" for k, v in extra.items()))
    t0 = time.time()
    # buffering=1 makes the log file line-buffered; explicit flush()
    # covers any partial writes (no trailing newline). We do NOT
    # mirror the bench's raw stdout to stage 7's stdout -- per-call
    # kernel dumps (hundreds of lines) would drown stage07.log and
    # make the orchestrator-level summary unreadable. The per-yaml
    # bench.log under `<out>/<yaml>/<ml_on|ml_off>/bench.log` still
    # captures everything for drill-down.
    # F6: run the bench from the deployed .bin's directory so the runtime's
    # built-in `./tilewright_weights.bin` search auto-finds the weights
    # (no TILEWRIGHT_WEIGHTS needed). The deployed .bin lives at
    # <deploy>/shared/tilewright/backends/hipblaslt/data/ -- not one of the binary's libOrigami-
    # relative search paths -- so we cd into it. --yaml/output paths are
    # absolute, so changing cwd is safe. ML-off (ml_weights_path None) keeps
    # the default cwd; with TENSILE_USE_TILEWRIGHT unset it won't load anyway.
    run_cwd = (
        str(ml_weights_path.parent)
        if ml_weights_path is not None and ml_weights_path.parent.is_dir()
        else None
    )
    # Stall watchdog. A single runaway kernel can pin the GPU at 100% and emit
    # nothing further; `for line in proc.stdout` then blocks forever, because
    # the process is alive and the pipe is simply idle. Without a watchdog one
    # hung shape stalls the stage indefinitely and every shape after it goes
    # unmeasured. Drain stdout on a thread and watch the log for growth, so a
    # hang costs one shape instead of the whole stage. Mirrors stage03's
    # startup-grace / stall-timeout split: the first shape also pays library
    # load, so it gets a much longer budget than steady state.
    stall_reason: Optional[str] = None
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=run_cwd,
        )
        assert proc.stdout is not None

        def _drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()

        pump = threading.Thread(target=_drain, daemon=True)
        pump.start()

        last_size, last_change = -1, time.time()
        while proc.poll() is None:
            time.sleep(5.0)
            try:
                size = log_path.stat().st_size
            except OSError:
                size = last_size
            now = time.time()
            if size != last_size:
                last_size, last_change = size, now
                continue
            # No new bytes. Before the first output the budget is the library
            # load; after it, a single shape should never take this long.
            budget = startup_grace_s if last_size <= 0 else stall_timeout_s
            if now - last_change > budget:
                stall_reason = (
                    f"no output for {int(now - last_change)}s "
                    f"(budget {int(budget)}s); killing bench"
                )
                ui.err("bench", stall_reason)
                logf.write(f"\n*** KILLED [stall]: {stall_reason}\n")
                logf.flush()
                proc.kill()
                break
        rc = proc.wait()
        pump.join(timeout=30.0)
    el = time.time() - t0
    if not quiet:
        msg = f"completed in {ui.fmt_dur(el)}"
        (ui.ok if rc == 0 else ui.err)("bench", f"{msg} (rc={rc})")
    return rc


def _parse_selection_times(log_path: Path) -> List[float]:
    """Pull every `Solution selection time: N us` line out of a bench
    log -- one per findTopSolutions call. Returned in source order so
    callers can drop the cold first call if they want."""
    out: List[float] = []
    if not log_path.exists():
        return out
    with log_path.open() as f:
        for line in f:
            m = _RE_SEL_TIME.search(line)
            if not m:
                continue
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
    return out


def _parse_ml_picks(log_path: Path) -> List[Dict[str, Any]]:
    """Pull every `[TILEWRIGHT_PICK] ...` line out of a bench log. Returns one
    dict per call, in source order (= the order hipblaslt-bench ran
    the problems). The 8-tuple in `top1_sig` matches the Python sig
    produced by `lib.dat.sig_from_config_t` so a direct == compares
    them.

    NOTE on duplicate (m,n,k,b): if the bench yaml lists the same
    problem twice OR `--iters > 1`, multiple [TILEWRIGHT_PICK] lines share
    the same problem key but should have IDENTICAL picks (the model
    is deterministic). We keep them all and let the parity check
    dedup by problem key (first occurrence wins) to compare against
    Python's per-problem pick.
    """
    out: List[Dict[str, Any]] = []
    if not log_path.exists():
        return out
    with log_path.open() as f:
        for line in f:
            m = _RE_ML_PICK.search(line)
            if not m:
                continue
            out.append(
                {
                    "m": int(m.group(1)),
                    "n": int(m.group(2)),
                    "k": int(m.group(3)),
                    "batch": int(m.group(4)),
                    # transA / transB are optional in the regex; older
                    # bench logs (pre-tA/tB runtimes) leave these None.
                    # When None, the parity join falls back to the
                    # (m,n,k,b) key like before.
                    "transA": m.group(5),
                    "transB": m.group(6),
                    "cell": m.group(7),
                    "top1_sig": (
                        int(m.group(8)),
                        int(m.group(9)),
                        int(m.group(10)),
                        int(m.group(11)),
                        int(m.group(12)),
                        int(m.group(13)),
                        int(m.group(14)),
                        int(m.group(15)),
                    ),
                    "top1_score": float(m.group(16)),
                    "n_configs": int(m.group(17)),
                }
            )
    return out


def _problem_key(
    p: Dict[str, Any],
) -> Tuple[int, int, int, int, Optional[str], Optional[str]]:
    """Stable key used to join C++ [TILEWRIGHT_PICK] lines against Python
    picks against bench yaml entries. transA / transB ARE in the key
    so the join survives a multi-layout yaml (one BBS_TN problem and
    one BBS_NT problem at the same (m,n,k,b) shape WOULD collide on
    just the size tuple). Newer C++ runtimes emit `tA=` / `tB=`;
    bench logs from older runtimes leave them None and the
    join falls back to the size-only tuple (legacy behavior)."""
    return (
        int(p["m"]),
        int(p["n"]),
        int(p["k"]),
        int(p.get("batch", p.get("batch_count", 1))),
        p.get("transA"),
        p.get("transB"),
    )


def _normalize_bench_yaml_for_seltime(
    src: Path,
    dst: Path,
    workload_filter: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    """Rewrite a hipblaslt-bench yaml so each problem runs with the
    selection-time-friendly knobs:
        iters: 1, cold_iters: 0, requested_solution_num: -1.

    Why these specifically:
      * `iters=1, cold_iters=0`: stage 7 measures `findTopSolutions`
        wall-clock, not kernel benchmark throughput. Extra iters
        just inflate the bench wall-clock without changing the
        selection-time line we parse.
      * `requested_solution_num=-1`: materialises every supported
        solution, which makes ml_on/bench.log large. That
        is deliberate -- `1` skips ranking entirely (see the inline note
        at the assignment), so the ML path under test never runs.

    `workload_filter` is an optional dict of bench-yaml fields that
    MUST match for the entry to survive. Used to keep only the
    in-distribution slice (matching the trained model's dtype +
    transpose layout) before the bench runs at all. Without this
    pre-filter the bench would invoke `findTopSolutions` on
    out-of-distribution shapes (e.g. m=1 GEMV with xf32_r compute
    type, where hipBLASLt's `xf32_r -> f32_r` library fallback can
    trigger a SECOND rank_configs call -- inflating the call count
    above the yaml entry count and dragging in problems the model
    was never trained for. Filtering at the yaml level keeps the
    bench, the parity check, and the timing summary all on the
    same set of problems.

    A handful of fields that only make sense at iters>1 are also
    dropped: skip_slow_solution_ratio (a multi-solution filter),
    print_kernel_info (just noise here -- we have parity.json for
    pick auditing). Everything else (M/N/K/B, dtypes, transposes,
    ld*, stride*, alpha/beta, rotating, use_gpu_timer, ...) is
    passed through unchanged so the bench still sees the same GEMM.

    Output is written in flow style (one entry per line, inline
    dict), matching the format hipblaslt-bench's gentest parser
    expects. Returns `(n_before_filter, n_after_filter)` for the
    caller's logging.
    """
    import yaml  # local import; lazy load

    with src.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"bench yaml must be a LIST of problem dicts; got "
            f"{type(data).__name__} ({src})"
        )

    # Normalize the filter values so we can compare against the
    # yaml entries. hipBLASLt yamls write `compute_type` with the
    # `c_` prefix (e.g. `c_f32_r`), and the config carries the same
    # prefixed form -- they should compare directly. transA / transB
    # are single chars (`T` / `N`). We strip leading/trailing
    # whitespace and compare case-sensitively (yaml is consistent).
    wf: Dict[str, str] = {}
    if workload_filter:
        for k, v in workload_filter.items():
            if v is None or str(v) == "":
                continue
            wf[k] = str(v).strip()

    def _entry_matches(e: Dict[str, Any]) -> bool:
        for fk, fv in wf.items():
            if str(e.get(fk, "")).strip() != fv:
                return False
        return True

    _DROP_KEYS = ("skip_slow_solution_ratio", "print_kernel_info")
    out_entries: List[Dict[str, Any]] = []
    n_before = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        n_before += 1
        if not _entry_matches(entry):
            continue
        e = dict(entry)
        e["iters"] = 1
        e["cold_iters"] = 0
        # MUST be -1. `1` does NOT mean "return the top of
        # findTopSolutions": it takes a short-circuit path that yields a
        # single solution WITHOUT ranking, so ProblemPredictionLibrary ->
        # rank_configs is never called, tilewright never loads its weights,
        # and the [TILEWRIGHT_DIAG FILE] marker never appears -- so stage 7
        # aborts with "the deployed origami isn't ml_recommender-linked".
        # With `1` the log carries no DIAG and no PICK lines; with `-1`
        # both appear. Bigger logs are the price of actually exercising
        # the path this stage exists to validate.
        e["requested_solution_num"] = -1
        for k in _DROP_KEYS:
            e.pop(k, None)
        out_entries.append(e)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for e in out_entries:
            # Flow-style inline dict, one per line. `width=99999`
            # stops yaml.dump from line-wrapping mid-entry (gentest
            # wants the whole dict on one line).
            line = yaml.dump(
                e, default_flow_style=True, width=99999, sort_keys=False
            ).strip()
            f.write("- " + line + "\n")
    return n_before, len(out_entries)


def _parse_full_shapes_yaml(path: Path) -> List[Dict[str, Any]]:
    """Load a hipblaslt-bench yaml as full per-problem dicts.

    `lib.bench_yaml.parse_shapes_yaml` only returns (M, N, K, B)
    tuples (regex per line, fast for huge OOB seeds). The parity
    check needs dtype + layout per problem too, so we go through
    `yaml.safe_load`. Stage 7 yamls have at most a few hundred
    entries -- the overhead is negligible.

    hipblaslt-bench yamls use UPPERCASE `M`/`N`/`K` (the standalone
    bench tool's convention). The pipeline's feature/training code
    uses lowercase. We normalize here so downstream consumers
    (`compute_deployed_top1_picks`, `_problem_key`, the C++
    `[TILEWRIGHT_PICK]` join, etc.) see one consistent shape.
    """
    import yaml  # local import; lazy load

    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"bench yaml must be a LIST of problem dicts; got "
            f"{type(data).__name__} ({path})"
        )
    # Map bench-yaml key -> pipeline-internal key. Everything not
    # listed here is passed through unchanged so downstream code
    # that reads e.g. `function`, `initialization`, `iters` still
    # works.
    _RENAME = {
        "M": "m",
        "N": "n",
        "K": "k",
    }
    out: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        norm: Dict[str, Any] = {}
        for k, v in entry.items():
            nk = _RENAME.get(k, k)
            # If both cases are present (defensive), the lowercase
            # value wins so a user who already lowercased the yaml
            # is honored.
            norm.setdefault(nk, v)
        out.append(norm)
    return out


def _compute_parity(
    yaml_path: Path,
    ml_on_log: Path,
    train_dir: Path,
    prior_round_dirs: List[Path],
    rocm_libraries_to_deploy: Path,
    arch: str,
    device: str,
    smart_k: int,
    origami_python_path: Optional[Path],
    origami_device: int,
    quiet: bool,
    workload_filter: Optional[Dict[str, str]] = None,
    weight_dtype: str = "fp32",
    scale_mode: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the C++ <-> Python pick parity check for one bench yaml.

    Returns:
        n_problems        problems in the yaml
        n_cpp_picks       [TILEWRIGHT_PICK] lines we parsed out of the ML-on
                          bench log (deduped by problem key; first
                          occurrence per problem wins)
        n_py_picks        problems we successfully scored in Python
        n_compared        problems where BOTH sides produced a pick
        n_match           subset of n_compared where the 8-tuple sig
                          is identical
        match_rate        n_match / n_compared (NaN if n_compared==0)
        per_problem       list of dicts -- one row per problem in
                          file order -- with both picks side by side
                          plus a `match`/`reason` tag.
    """
    # Order matters: origami python path must be in sys.path BEFORE
    # we import stage04/stage05, because those modules do a top-level
    # `import origami` and freeze the result in their own namespace.
    # If we add the path AFTER, their `origami` stays `None` and any
    # `origami.config_t()` call inside them blows up with
    # `AttributeError: 'NoneType' object has no attribute 'config_t'`.
    if origami_python_path and str(origami_python_path) not in sys.path:
        sys.path.insert(0, str(origami_python_path))
    try:
        import origami as _ori_mod  # type: ignore
    except ImportError as e:
        return {
            "reason": f"origami_import_failed:{e!r}",
            "n_problems": 0,
            "n_cpp_picks": 0,
            "n_py_picks": 0,
            "n_compared": 0,
            "n_match": 0,
            "match_rate": float("nan"),
            "per_problem": [],
        }

    # NOW it's safe to pull in stage04 / stage05.
    import torch  # noqa: F401  -- guards CUDA init order

    sys.path.insert(0, str(PIPELINE_ROOT / "stages"))
    import stage05_train as _train  # type: ignore

    sys.path.insert(0, str(PIPELINE_ROOT))
    from lib import subcells as sc  # noqa: E402
    from stage04_convert_to_enriched_dataset import (  # type: ignore
        load_origami_configs_by_dtype as _load_cfgs,
        _origami_hardware as _ori_hw,
    )

    # Even with the path-first ordering above, stage04/stage05 may
    # have been imported earlier in THIS process for other reasons
    # (orchestrator runs both stages in-process). Explicitly bind
    # the live `origami` module so stale `_ORIGAMI_OK=False` / None
    # state from an earlier import order doesn't bite us.
    import stage04_convert_to_enriched_dataset as _enrich  # type: ignore

    _enrich.origami = _ori_mod
    _enrich._ORIGAMI_OK = True
    _train.origami = _ori_mod
    _train._ORIGAMI_OK = True

    # Origami configs index. The deploy checkout's dat_dir holds the
    # SAME kernels the bench runtime sees (these are what the C++ runs
    # rank_configs against), so we point at the deploy repo.
    # Same per-arch resolution as the bench: the v0 tree lives in
    # Tensile-<arch>/library/<arch> until install merges it under
    # Tensile/library/. Reading the wrong one indexes the WRONG kernels.
    _root = rocm_libraries_to_deploy / "projects/hipblaslt/build/release"
    dat_dir = _root / f"Tensile-{arch}/library/{arch}"
    if not dat_dir.is_dir():
        dat_dir = _root / "Tensile/library" / arch
    if not dat_dir.exists():
        # Deploy checkout might not have built kernels (the bench
        # binary won't exist either, but we'd have skipped stage 7
        # earlier in that case). Fail gracefully here.
        return {
            "reason": f"dat_dir_missing:{dat_dir}",
            "n_problems": 0,
            "n_cpp_picks": 0,
            "n_py_picks": 0,
            "n_compared": 0,
            "n_match": 0,
            "match_rate": float("nan"),
            "per_problem": [],
        }
    # `scale_mode` narrows the pool to the libraries the run's
    # scaleA/scaleB actually selects. Without it the gfx1250 MX and
    # non-MX F8 TN kernels land in ONE pool under the same key, and the
    # Python side would score candidates the C++ runtime can't pick --
    # guaranteeing a parity mismatch that isn't a real bug.
    loaded = _load_cfgs(str(dat_dir), quiet=quiet, scale_mode=scale_mode)
    configs_by_key = loaded[0] if isinstance(loaded, tuple) else loaded
    info_by_key = loaded[1] if isinstance(loaded, tuple) else None
    # Pass the run's arch: _origami_hardware reports it and uses it for the
    # fallback path. Omitting it leaves the gfx950 default, which is exactly
    # the silent wrong-hardware case the arch parameter was added to remove.
    hw = _ori_hw(int(origami_device), quiet=quiet, arch=arch)
    if hw is None:
        return {
            "reason": "no_origami_hardware",
            "n_problems": 0,
            "n_cpp_picks": 0,
            "n_py_picks": 0,
            "n_compared": 0,
            "n_match": 0,
            "match_rate": float("nan"),
            "per_problem": [],
        }

    # Trained bundle + cumulative split tree.
    bundle = torch.load(
        str(train_dir / "models.pt"), map_location="cpu", weights_only=False
    )
    split_tree = sc.load_cumulative_split_tree(prior_round_dirs)

    # ── Python picks ──────────────────────────────────────────────
    problems = _parse_full_shapes_yaml(yaml_path)
    # Restrict the parity check to in-distribution problems (same
    # dtype + transpose the model was trained on). Without this
    # filter, a multi-dtype / mixed-layout external trace
    # (production-OOB yamls routinely carry the full BBS_TN + BBS_NT
    # + HHS_NN + ... mix) pumps OUT-OF-DISTRIBUTION problems through
    # the C++ recommender. Routing still lands on the same cell-key
    # M/N/K/B tiers, but `is_kernel_feasible`'s per-problem NTA/NTB
    # branch flips with the transpose layout -- so a model whose
    # smart_K was greedily covered on TN layout gets every K-list
    # signature rejected when scored against an NT problem and the
    # runtime falls back. Filtering ensures we only measure parity
    # on problems the model was actually trained to serve.
    n_problems_before_filter = len(problems)
    if workload_filter:

        def _norm(k, v):
            s = str(v).strip()
            if k == "compute_type" and s.lower().startswith("c_"):
                s = s[2:]
            return s

        wf = {
            k: _norm(k, v)
            for k, v in workload_filter.items()
            if v is not None and str(v) != ""
        }

        def _keep(p):
            for fk, fv in wf.items():
                if _norm(fk, p.get(fk, "")) != fv:
                    return False
            return True

        problems = [p for p in problems if _keep(p)]
        if not quiet:
            ui.info(
                "filter",
                f"workload_filter={wf}  "
                f"problems: {n_problems_before_filter} -> "
                f"{len(problems)}",
            )
    from lib.dat import mx_block_size_for_scale_mode as _mx_bs

    py_picks = _train.compute_deployed_top1_picks(
        problems,
        bundle,
        split_tree,
        configs_by_key,
        hw,
        arch,
        device,
        smart_k=smart_k,
        info_by_key=info_by_key,
        weight_dtype=weight_dtype,
        mx_block_size=_mx_bs(scale_mode),
    )
    py_by_key: Dict[Tuple, Dict[str, Any]] = {}
    # Also index NON-ok picks so the `no_py_pick` row can surface
    # WHY the Python side returned nothing (helps the operator
    # debug schema drift, missing models, etc.).
    py_skipped_by_key: Dict[Tuple, Dict[str, Any]] = {}
    n_py_ok = 0
    n_py_fallback = 0
    for pr in py_picks:
        key = _problem_key(pr)
        if pr.get("reason") != "ok":
            py_skipped_by_key.setdefault(key, pr)
            continue
        py_by_key.setdefault(key, pr)
        n_py_ok += 1
        if pr.get("used_smart_k_fallback"):
            n_py_fallback += 1

    # ── C++ picks ─────────────────────────────────────────────────
    cpp_picks = _parse_ml_picks(ml_on_log)
    cpp_by_key: Dict[Tuple, Dict[str, Any]] = {}
    for cp in cpp_picks:
        cpp_by_key.setdefault(_problem_key(cp), cp)

    # ── join in file order ────────────────────────────────────────
    per_problem: List[Dict[str, Any]] = []
    n_match = n_compared = 0
    for p in problems:
        key = _problem_key(p)
        cpp = cpp_by_key.get(key)
        py = py_by_key.get(key)
        row: Dict[str, Any] = {
            "m": key[0],
            "n": key[1],
            "k": key[2],
            "batch": key[3],
            "transA": key[4],
            "transB": key[5],
        }
        if cpp is None and py is None:
            row["match"] = False
            row["reason"] = "no_pick_either_side"
        elif cpp is None:
            row["match"] = False
            row["reason"] = "no_cpp_pick"
            row["py_sig"] = list(py["top1_sig"])
            row["py_cell"] = py.get("cell")
        elif py is None:
            row["match"] = False
            row["reason"] = "no_py_pick"
            row["cpp_sig"] = list(cpp["top1_sig"])
            row["cpp_cell"] = cpp.get("cell")
            # Surface the underlying py reason so the operator can
            # see WHY the python helper returned nothing
            # (no_survivors / no_model / no_origami / etc).
            py_skipped = py_skipped_by_key.get(key)
            if py_skipped is not None:
                row["py_reason"] = py_skipped.get("reason")
        else:
            row["cpp_sig"] = list(cpp["top1_sig"])
            row["cpp_cell"] = cpp.get("cell")
            row["cpp_score"] = cpp.get("top1_score")
            row["py_sig"] = list(py["top1_sig"])
            row["py_cell"] = py.get("cell")
            row["py_score"] = py.get("top1_score")
            row["match"] = tuple(cpp["top1_sig"]) == tuple(py["top1_sig"])
            row["reason"] = "ok" if row["match"] else "sig_mismatch"
            n_compared += 1
            if row["match"]:
                n_match += 1
        per_problem.append(row)

    return {
        "n_problems": len(problems),
        "n_cpp_picks": len(cpp_by_key),
        "n_py_picks": n_py_ok,
        # How many of the n_py_ok picks fell back to "LDS+SC only"
        # because the cell's smart_K signature list didn't include
        # any kernel that satisfies is_kernel_feasible for that
        # problem. High fallback rate = the K-list isn't picking
        # kernels with diverse-enough NTA/NTB combinations.
        "n_py_smart_k_fallback": n_py_fallback,
        "n_compared": n_compared,
        "n_match": n_match,
        "match_rate": (n_match / n_compared if n_compared > 0 else float("nan")),
        "smart_k": smart_k,
        "per_problem": per_problem,
    }


def _parse_ml_diag(log_path: Path) -> Dict[str, Any]:
    """Return what we learned from the TILEWRIGHT_DIAG=1 instrumentation in
    the bench log:
      ml_diag_seen        bool   `[TILEWRIGHT_DIAG FILE] ...` line present
      no_weights_seen     bool   `[origami::ml_recommender] no
                                  weights loaded` line present
      arch / qhash / dims / n_cells / n_splits   parsed from the
                                  TILEWRIGHT_DIAG line when present.
    A ML-on bench with ml_diag_seen=False means the deployed
    runtime never opened the .bin -- the operator should treat the
    timings as 'plain Origami' regardless of what TENSILE_ML_
    RECOMMENDER was set to."""
    out: Dict[str, Any] = {
        "ml_diag_seen": False,
        "no_weights_seen": False,
    }
    if not log_path.exists():
        return out
    with log_path.open() as f:
        for line in f:
            m = _RE_TILEWRIGHT_DIAG_LOAD.search(line)
            if m:
                out["ml_diag_seen"] = True
                out["arch"] = m.group(1)
                out["qhash"] = m.group(2)
                out["q_dim"] = int(m.group(3))
                out["i_dim"] = int(m.group(4))
                out["x_dim"] = int(m.group(5))
                out["n_cells"] = int(m.group(6))
                out["n_splits"] = int(m.group(7))
            elif _RE_ML_NO_WEIGHTS.search(line):
                out["no_weights_seen"] = True
    return out


def _summarize(name: str, calls: List[float], drop_first: bool) -> Dict[str, Any]:
    """Mean / median / p99 over the per-call selection times. Optionally
    drops the FIRST call (library-init cost; far above the
    steady-state cost the operator actually cares about)."""
    raw = list(calls)
    dropped = []
    if drop_first and raw:
        dropped = [raw[0]]
        raw = raw[1:]
    if not raw:
        return {
            "name": name,
            "n_calls_raw": len(calls),
            "n_calls": 0,
            "first_call_us": dropped[0] if dropped else None,
        }

    def _p99(xs: List[float]) -> float:
        s = sorted(xs)
        return s[max(0, int(len(s) * 0.99) - 1)]

    return {
        "name": name,
        "n_calls_raw": len(calls),
        "n_calls": len(raw),
        "first_call_us": dropped[0] if dropped else None,
        "sel_us_mean": statistics.fmean(raw),
        "sel_us_p50": statistics.median(raw),
        "sel_us_p99": _p99(raw),
        "sel_us_min": min(raw),
        "sel_us_max": max(raw),
    }


# ── direct rank_configs() call-time microbench ───────────────────────
#
# The GPU selection-time above (parsed from `Solution selection time:`)
# is END-TO-END -- it includes hipBLASLt's findTopSolutions dispatch,
# problem setup, library lookup, etc. To isolate the COST OF THE SCORER
# ITSELF we also run the CPU-only microbench in
# scripts/experiments/bench_selection_time.{py,cpp,sh}, which times
#   origami::rank_configs                  (analytical)
#   origami::ml_recommender::rank_configs  (ML two-tower)
# directly, over the deploy library's real candidate set. No GPU work,
# so this is safe to run alongside other GPU jobs. Fully guarded: if the
# deploy repo's liborigami.a isn't built (or no UserArgs.yaml / compiler
# is found) it SKIPS with a reason rather than failing stage 7.
# Columns: path n min median mean p90 p99 max. p90/p99 are optional so an
# older bench binary (min median mean max only) still parses.
_RE_RANK_ROW = re.compile(
    r"^(origami|ml_rec)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    r"(?:\s+([\d.]+)\s+([\d.]+))?\s+([\d.]+)\s*$"
)
_RE_RANK_SPEEDUP = re.compile(r"mean speedup ml_rec vs origami:\s*([\d.]+)x")
_RE_RANK_SPEEDUP_MED = re.compile(r"median speedup ml_rec vs origami:\s*([\d.]+)x")


def _find_userargs_yaml(
    repo: Path, arch: str, workload_filter: Optional[Dict[str, str]]
) -> Optional[Path]:
    """Best-effort locate a Tensile Origami UserArgs.yaml in the deploy
    repo to source the candidate-kernel set from. Prefers a file whose
    name matches the deployed dtype/layout (BBS == bf16, Alik_Bljk == TN)
    so the candidate count mirrors what the runtime dispatches from."""
    cands = sorted(repo.glob(f"projects/hipblaslt/**/Origami/{arch}_*UserArgs.yaml"))
    if not cands:
        cands = sorted(repo.glob("projects/hipblaslt/**/*UserArgs.yaml"))
    if not cands:
        return None
    wf = workload_filter or {}
    a_type = str(wf.get("a_type", "")).lower()
    # dtype token in the Origami liblogic yaml name. f32/xf32 (TF32) uses a
    # standalone "_S_" (single precision); bf16->BBS, f16->HHS, f8->B8.
    if a_type.startswith("bf16"):
        dtype_tok = "BBS"
    elif a_type.startswith("f16"):
        dtype_tok = "HHS"
    elif a_type.startswith(("f8", "fp8")):
        dtype_tok = "B8"
    elif a_type.startswith("f32") or "xf32" in a_type:
        dtype_tok = "_S_"
    else:
        dtype_tok = None
    transA = str(wf.get("transA", "")).upper()
    transB = str(wf.get("transB", "")).upper()
    # Tensile contraction layout: TN (T,N) -> Alik_Bljk, NT (N,T) -> Ailk_Bjlk.
    if transA == "T" and transB == "N":
        layout_tok = "Alik_Bljk"
    elif transA == "N" and transB == "T":
        layout_tok = "Ailk_Bjlk"
    else:
        layout_tok = None

    def _score(p: Path) -> int:
        s = 0
        if dtype_tok and dtype_tok in p.name:
            s += 2
        if layout_tok and layout_tok in p.name:
            s += 1
        return s

    best = max(cands, key=_score)
    if _score(best) == 0:
        # No dtype/layout match -- the timing would run on an arbitrary
        # library. Surface it instead of silently mis-targeting (the old bug).
        try:
            from lib import ui as _ui

            _ui.warn(
                "rank-time",
                f"no UserArgs.yaml matched dtype={a_type} "
                f"layout={transA}{transB}; microbench library may be "
                f"unrepresentative ({best.name}).",
            )
        except Exception:
            pass
    return best


def _resolve_deployed_weights_bin(
    repo: Path, arch: str, wfilter: Optional[Dict[str, str]]
) -> Optional[Path]:
    """Resolve the per-stem `.tilewright.bin` matching this config's dtype/layout
    from the deployed nested weights dir `weights/<arch>/<arch>/`, via its
    `tilewright_index` (productized layout). Falls back to the legacy flat global
    `tilewright_weights.bin` for older checkouts. Returns None if none found.

    The call-time microbench loads ONE model, so it needs the specific per-stem
    .bin for this workload -- not a global file (which no longer exists in the
    per-stem layout, hence the old "tilewright_weights.bin missing" skip)."""
    from lib.dat import parse_dtype_layout  # noqa: E402

    arch_base = str(arch).split(":")[0]
    wdir = repo / "shared/tilewright/backends/hipblaslt/weights" / arch_base / arch_base
    idx = wdir / "tilewright_index"
    wf = wfilter or {}
    want = (
        wf.get("a_type"),
        wf.get("d_type"),
        wf.get("compute_type"),
        wf.get("transA"),
        wf.get("transB"),
    )
    if idx.exists() and all(want):
        want_s = tuple(str(x) for x in want)
        for line in idx.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            stem, binname = line.split("\t", 1)
            key = parse_dtype_layout(stem)
            if key is not None and tuple(str(x) for x in key) == want_s:
                cand = wdir / binname.strip()
                if cand.exists():
                    return cand
    for rel in (
        "shared/tilewright/backends/hipblaslt/data/tilewright_weights.bin",
        "shared/tilewright/data/hipblaslt/ml_recommender_weights.bin",
        "shared/origami/data/ml_recommender_weights.bin",
    ):
        cand = repo / rel
        if cand.exists():
            return cand
    return None


def _measure_rank_call_time(
    *,
    origami_repo: Path,
    weights_path: Optional[Path],
    userargs_yaml: Optional[Path],
    arch: str,
    iters: int,
    warmup: int,
    out_dir: Path,
    quiet: bool,
) -> Dict[str, Any]:
    """Run scripts/experiments/bench_selection_time.sh (parse yaml ->
    compile -> run) against the deploy library and parse the per-path
    summary table. Returns {origami, ml_rec, speedup_*} on success, or
    {skipped: <reason>} when a prerequisite is missing. CPU-only."""
    script = PIPELINE_ROOT / "scripts/experiments/bench_selection_time.sh"
    lib = origami_repo / "projects/hipblaslt/build/release/origami/liborigami.a"
    tilewright_lib = (
        origami_repo / "projects/hipblaslt/build/release/tilewright/libtilewright.a"
    )
    inc = (
        origami_repo
        / "shared/tilewright/backends/hipblaslt/include/tilewright/hipblaslt/adapter.hpp"
    )
    if not script.exists():
        return {"skipped": f"missing {script}"}
    if not lib.exists():
        return {
            "skipped": f"liborigami.a not built at {lib} "
            f"(build origami in the deploy repo to enable the "
            f"rank_configs call-time microbench)"
        }
    if not tilewright_lib.exists():
        return {
            "skipped": f"libtilewright.a not built at {tilewright_lib} "
            f"(build tilewright in the deploy repo to enable the "
            f"rank_configs call-time microbench)"
        }
    if not inc.exists():
        return {"skipped": f"tilewright backend header missing at {inc}"}
    if weights_path is None or not Path(weights_path).exists():
        return {"skipped": f"tilewright_weights.bin missing " f"({weights_path})"}
    if userargs_yaml is None or not Path(userargs_yaml).exists():
        return {
            "skipped": "no Tensile Origami UserArgs.yaml found "
            "(pass --rank-time-userargs-yaml)"
        }
    env = os.environ.copy()
    env.update(
        {
            "ORIGAMI_REPO": str(origami_repo),
            "WEIGHTS_BIN": str(weights_path),
            "YAML": str(userargs_yaml),
            "ARCH": arch,
            "ITERS": str(iters),
            "WARMUP": str(warmup),
            "OUT_DIR": str(out_dir / "rank_call_time"),
        }
    )
    if not quiet:
        ui.info(
            "rank-time",
            f"microbench: origami::rank_configs vs "
            f"tilewright::hipblaslt::rank_configs  (yaml={userargs_yaml.name}, "
            f"iters={iters}, warmup={warmup})",
        )
    try:
        proc = subprocess.run(
            ["bash", str(script)], env=env, capture_output=True, text=True, timeout=900
        )
    except Exception as e:  # noqa: BLE001 -- never let this fail stage 7
        return {"skipped": f"bench_selection_time.sh did not run: {e!r}"}
    if proc.returncode != 0:
        return {
            "skipped": f"bench_selection_time.sh rc={proc.returncode}",
            "stderr_tail": proc.stderr[-600:],
        }
    rows: Dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        m = _RE_RANK_ROW.match(line.strip())
        if m:
            rows[m.group(1)] = {
                "n": int(m.group(2)),
                "min_us": float(m.group(3)),
                "median_us": float(m.group(4)),
                "mean_us": float(m.group(5)),
                # p90/p99 present only with the newer bench binary.
                "p90_us": float(m.group(6)) if m.group(6) else None,
                "p99_us": float(m.group(7)) if m.group(7) else None,
                "max_us": float(m.group(8)),
            }
    sp = _RE_RANK_SPEEDUP.search(proc.stdout)
    sp_med = _RE_RANK_SPEEDUP_MED.search(proc.stdout)
    return {
        "userargs_yaml": str(userargs_yaml),
        "iters": iters,
        "warmup": warmup,
        "origami": rows.get("origami"),
        "ml_rec": rows.get("ml_rec"),
        "speedup_mean_ml_vs_origami": float(sp.group(1)) if sp else None,
        "speedup_median_ml_vs_origami": (float(sp_med.group(1)) if sp_med else None),
        "stdout": proc.stdout,
    }


def _compare(on: Dict[str, Any], off: Dict[str, Any]) -> Dict[str, Any]:
    """ML overhead = on - off, on each summary statistic."""

    def _delta(k: str) -> Optional[float]:
        a, b = on.get(k), off.get(k)
        if a is None or b is None:
            return None
        try:
            return float(a) - float(b)
        except (TypeError, ValueError):
            return None

    return {
        "ml_on": on,
        "ml_off": off,
        "delta": {
            "sel_us_mean": _delta("sel_us_mean"),
            "sel_us_p50": _delta("sel_us_p50"),
            "sel_us_p99": _delta("sel_us_p99"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage07: measure ML-recommender vs Origami "
        "selection-time on real GPU."
    )
    ap.add_argument(
        "--rocm-libraries-to-deploy",
        type=Path,
        required=True,
        help="rocm-libraries checkout that already has "
        "stage06_deploy's artifacts installed AND was "
        "rebuilt. We read hipblaslt-bench from "
        "<repo>/projects/hipblaslt/build/release/"
        "clients/hipblaslt-bench.",
    )
    ap.add_argument(
        "--hipblaslt-bench-yaml",
        "--yaml",
        type=Path,
        action="append",
        required=True,
        dest="bench_yaml",
        help="bench yaml. Repeatable (one timing run per " "yaml).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="dir for per-yaml logs + timings summary.",
    )
    ap.add_argument(
        "--devices",
        default="0",
        help="comma-separated device IDs. Only the FIRST "
        "is used; multi-device runs are out of scope.",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="hipblaslt-bench --iters. Default 1 so the "
        "selection-time spans dominate.",
    )
    ap.add_argument(
        "--cold-iters", type=int, default=0, help="hipblaslt-bench --cold_iters."
    )
    ap.add_argument(
        "--drop-first-call",
        dest="drop_first_call",
        action="store_true",
        default=True,
        help="drop the first per-yaml selection-time call "
        "(library-init cost; default).",
    )
    ap.add_argument(
        "--no-drop-first-call",
        dest="drop_first_call",
        action="store_false",
        help="include the first call (cold).",
    )
    ap.add_argument(
        "--ml-weights-path",
        type=Path,
        default=None,
        help="explicit path to tilewright_weights.bin. "
        "Exported to hipblaslt-bench as "
        "`TILEWRIGHT_WEIGHTS=<path>`. Without it, "
        "origami's loader searches `./` and "
        "`/opt/rocm/share/origami/` only -- which "
        "usually means the just-staged .bin isn't "
        "found and the ML path silently falls back "
        "to analytical Origami. The orchestrator's "
        "`--mode validate` derives this automatically "
        "from `deploy.rocm_libraries_to_deploy`.",
    )
    ap.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="OPTIONAL trained-models dir (the LATEST "
        "round's `stage05/`, containing models.pt + "
        "cells.json). When set, stage 7 ALSO runs "
        "the C++<->Python pick PARITY check per "
        "problem (joins [TILEWRIGHT_PICK] lines from the "
        "ML-on bench log against Python "
        "`compute_deployed_top1_picks` from the "
        "same `models.pt`). A 100%% match rate is "
        "the strongest possible signal that the "
        "deployed runtime is honoring the staged "
        ".bin; anything less is a deploy bug.",
    )
    ap.add_argument(
        "--prior-round-dir",
        type=Path,
        action="append",
        default=None,
        help="OPTIONAL, repeatable. Used ONLY by the "
        "parity check to load the cumulative split "
        "tree (every entry contributes "
        "stage04b/splits.json). Pass every prior "
        "round when --train-dir came from active "
        "learning so the Python pick uses the same "
        "leaf-cell routing the deploy runtime does.",
    )
    ap.add_argument(
        "--arch",
        default="gfx950",
        help="arch label used by the parity check for "
        "feature extraction + dat-dir resolution.",
    )
    ap.add_argument(
        "--torch-device",
        default="cpu",
        help="torch device for the parity check's " "per-cell student inference.",
    )
    ap.add_argument(
        "--parity-smart-k",
        type=int,
        default=10,
        help="Candidate cap after the check_lds_capacity "
        "+ is_kernel_feasible pre-filter; walks "
        "configs in INPUT ORDER and keeps the first "
        "K. Match this to the deployed C++ "
        "runtime's ML_SMART_K (default 10) so the "
        "parity check scores the SAME K kernels "
        "the C++ scored. K=0 disables the cap "
        "(debug only).",
    )
    ap.add_argument(
        "--parity-weight-dtype",
        choices=("fp32", "bf16", "int8", "int4"),
        default="fp32",
        help="precision the deployed .bin stores its MLP "
        "`.weight` matrices in (stage06 "
        "--weight-dtype). When bf16/int8, the parity check "
        "rounds the Python student weights through the "
        "same grid too, so the C++<->Python pick parity "
        "stays exact. Must match what was deployed.",
    )
    ap.add_argument(
        "--origami-python-path",
        type=Path,
        default=None,
        help="dir containing the `origami` Python "
        "package (`<repo>/shared/origami/python/src`). "
        "Required for the parity check; ignored "
        "otherwise. The orchestrator derives it "
        "from deploy.rocm_libraries_to_deploy "
        "(falling back to runtime.rocm_libraries_root "
        "when the deploy build is incomplete).",
    )
    # None == no filter == the legacy unfiltered pool. See lib/dat.py.
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA/scaleB mode of the run under "
        "test (0=none, 1=Scalar, 3=Block_32_UE8M0/MX); "
        "restricts the parity config pool to the .dat "
        "libraries that mode actually selects",
    )
    ap.add_argument(
        "--origami-device",
        type=int,
        default=0,
        help="GPU device index passed to "
        "origami.get_hardware_for_device (parity-"
        "check only; bench timing uses the value "
        "passed via --devices). 0 is fine for the "
        "parity check because the hardware "
        "descriptor is per-arch, not per-board.",
    )
    ap.add_argument("--quiet", action="store_true")
    # ── direct rank_configs() call-time microbench (CPU-only) ─────────
    ap.add_argument(
        "--measure-rank-call-time",
        dest="measure_rank_call_time",
        action="store_true",
        default=True,
        help="ALSO run the CPU-only origami::rank_configs vs "
        "origami::ml_recommender::rank_configs call-time "
        "microbench (scripts/experiments/"
        "bench_selection_time.*). Isolates the SCORER cost "
        "from the end-to-end GPU selection time. CPU-only "
        "(safe alongside GPU jobs); skipped gracefully if "
        "the deploy repo's liborigami.a isn't built. "
        "Default on.",
    )
    ap.add_argument(
        "--no-measure-rank-call-time",
        dest="measure_rank_call_time",
        action="store_false",
    )
    ap.add_argument(
        "--rank-time-userargs-yaml",
        type=Path,
        default=None,
        help="Tensile Origami UserArgs.yaml for the call-time "
        "microbench's candidate set. Auto-found in the "
        "deploy repo (matching arch + dtype/layout) when "
        "omitted.",
    )
    ap.add_argument(
        "--rank-time-iters",
        type=int,
        default=200,
        help="median over this many timed rank_configs calls "
        "per shape (microbench).",
    )
    ap.add_argument("--rank-time-warmup", type=int, default=50)
    # Workload filter applied at parity-check time. Drops yaml
    # entries whose dtype + transpose layout don't match the model's
    # training distribution. WITHOUT this filter, a production
    # OOB yaml that mixes BBS_TN + BBS_NT + HSS_NN + ... pumps
    # out-of-distribution problems through the recommender; the
    # cell-key routing (M/N/K/B tiers only) still lands them on a
    # cell, but `is_kernel_feasible`'s per-problem NTA/NTB branch
    # flips with the transpose layout -- so an in-distribution
    # smart_K (chb=4-heavy for TN+K-aligned problems) gets every
    # signature rejected, runtime falls back, and the parity check
    # reports an artificially high fallback rate that's NOT a
    # training/deploy parity bug -- just a layout-mismatch artifact.
    # The orchestrator pulls the values from configs/<id>.yaml's
    # `hipblaslt:` block (a_type / b_type / transA / transB).
    ap.add_argument("--filter-a-type", default=None)
    ap.add_argument("--filter-b-type", default=None)
    ap.add_argument("--filter-c-type", default=None)
    ap.add_argument("--filter-d-type", default=None)
    ap.add_argument("--filter-compute-type", default=None)
    ap.add_argument("--filter-transA", default=None)
    ap.add_argument("--filter-transB", default=None)
    args = ap.parse_args()
    workload_filter = {
        k: v
        for k, v in {
            "a_type": args.filter_a_type,
            "b_type": args.filter_b_type,
            "c_type": args.filter_c_type,
            "d_type": args.filter_d_type,
            "compute_type": args.filter_compute_type,
            "transA": args.filter_transA,
            "transB": args.filter_transB,
        }.items()
        if v is not None
    } or None

    bench_bin = (
        args.rocm_libraries_to_deploy
        / "projects/hipblaslt/build/release/clients/hipblaslt-bench"
    )
    if not bench_bin.exists():
        ui.err("err", f"hipblaslt-bench not found at {bench_bin}")
        return 1

    # Explicit per-arch Tensile library dir. gfx1250 v0 builds into its own
    # tree (Tensile-<arch>/library/<arch>) that is only merged under
    # Tensile/library/ at install time, so prefer it when present.
    _build_root = args.rocm_libraries_to_deploy / "projects/hipblaslt/build/release"
    _tensile_libpath = _build_root / f"Tensile-{args.arch}/library/{args.arch}"
    if not _tensile_libpath.is_dir():
        _tensile_libpath = _build_root / f"Tensile/library/{args.arch}"
    if not _tensile_libpath.is_dir():
        _tensile_libpath = None

    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    if not devices:
        ui.err("err", "--devices is empty")
        return 1
    device = devices[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parity_enabled = args.train_dir is not None
    if not args.quiet:
        ui.banner("stage07  validate selection time")
        ui.info("cfg", f"rocm_libraries  : {args.rocm_libraries_to_deploy}")
        ui.info("cfg", f"bench_binary    : {bench_bin}")
        ui.info("cfg", f"out_dir         : {args.out_dir}")
        ui.info("cfg", f"device          : {device}")
        ui.info("cfg", f"iters/cold_iters: {args.iters} / {args.cold_iters}")
        ui.info("cfg", f"drop_first_call : {args.drop_first_call}")
        ui.info(
            "cfg",
            f"ml_weights_path : "
            f"{args.ml_weights_path or '(unset; runtime searches '
                                            './ and /opt/rocm/share/'
                                            'origami/ only)'}",
        )
        ui.info("cfg", f"yamls           : " f"{[str(p) for p in args.bench_yaml]}")
        if parity_enabled:
            ui.info("cfg", f"parity train-dir: {args.train_dir}")
            ui.info(
                "cfg",
                f"parity smart_k  : {args.parity_smart_k} "
                f"({'no cap (debug only)' if args.parity_smart_k == 0
                       else 'first-K survivors in input order; '
                            'mirrors ML_SMART_K in the deploy '
                            'runtime'})",
            )
            ui.info(
                "cfg",
                f"parity priors   : "
                f"{[str(p) for p in (args.prior_round_dir or [])]}",
            )
        else:
            ui.info("cfg", "parity check    : disabled (no --train-dir)")
    if args.ml_weights_path is not None and not args.ml_weights_path.exists():
        ui.err(
            "err",
            f"--ml-weights-path does not exist: {args.ml_weights_path} "
            f"(deploy with `--mode initial --deploy` or `--mode full` "
            f"to stage the .bin first)",
        )
        return 1
    if parity_enabled:
        if not (args.train_dir / "models.pt").exists():
            ui.err("err", f"--train-dir has no models.pt: {args.train_dir}")
            return 1

    summary: Dict[str, Any] = {"per_yaml": {}}
    for yp in args.bench_yaml:
        if not yp.exists():
            ui.err("err", f"yaml not found: {yp}")
            return 1
        tag = yp.stem
        ui.banner(f"timing :: {tag}", ui.C.BLUE)

        # Rewrite the yaml so each problem runs with iters=1,
        # cold_iters=0, requested_solution_num=-1. Stage 7's job is
        # to time `findTopSolutions`, not benchmark the picked
        # kernel; any iters>1 would inflate the per-problem
        # wall-clock without changing the selection-time number
        # we actually care about. requested_solution_num=-1 makes
        # the bench materialise every solution (slow, large logs),
        # but `1` would skip ranking entirely -- see
        # `_normalize_bench_yaml_for_seltime`. The orchestrator already
        # drops the first call to absorb library-init noise.
        norm_yaml = args.out_dir / tag / "shapes.normalized.yaml"
        # Filter the yaml to the trained workload (same dtype +
        # transpose layout the .bin was trained against) BEFORE we
        # invoke hipblaslt-bench. Stale out-of-distribution entries
        # in a multi-layout yaml would otherwise (a) pump
        # extra `findTopSolutions` calls for shapes the model was
        # never trained to serve and (b) on m=1 xf32 GEMV trigger
        # hipBLASLt's internal `xf32 -> f32` library fallback, which
        # silently emits a SECOND rank_configs call per such
        # problem -- inflating both the call count and the mean
        # selection time. Filtering at the yaml level keeps the
        # bench / parity / timing all on the SAME in-distribution
        # set of problems.
        n_before, n_after = _normalize_bench_yaml_for_seltime(
            yp, norm_yaml, workload_filter=workload_filter
        )
        if workload_filter and n_after == 0:
            ui.err(
                "err",
                f"workload_filter={workload_filter} dropped all "
                f"{n_before} entries from {yp}; nothing left to "
                f"bench. Check the filter values against the yaml's "
                f"dtype/transpose columns.",
            )
            return 1
        if workload_filter:
            ui.info(
                "yaml",
                f"normalized + filtered -> {norm_yaml.name}  "
                f"({n_after}/{n_before} entries match "
                f"{workload_filter})",
            )
        else:
            ui.info(
                "yaml",
                f"normalized -> {norm_yaml.name}  "
                f"({n_after}/{n_before} entries; "
                f"iters=1 cold_iters=0 requested_solution_num=-1)",
            )

        # ML ON: TENSILE_USE_TILEWRIGHT=1 enables the recommender code
        # path in the deployed origami.cpp (which then dispatches to
        # ml_recommender::rank_configs). TENSILE_PREDICTION_LIB=1 comes
        # for free from bench_env_with. TILEWRIGHT_DIAG=1 + TILEWRIGHT_PICK_LOG=1
        # are added ONLY on the ML-on side so they don't pollute
        # the ML-off log.
        ml_on_env: Dict[str, str] = {
            "TENSILE_USE_TILEWRIGHT": "1",
            "TILEWRIGHT_DIAG": "1",
            "TILEWRIGHT_PICK_LOG": "1",
        }
        log_on = args.out_dir / tag / "ml_on" / "bench.log"
        rc = _run_bench(
            bench_bin,
            norm_yaml,
            log_on,
            device,
            env_extra=ml_on_env,
            iters=args.iters,
            cold_iters=args.cold_iters,
            quiet=args.quiet,
            ml_weights_path=args.ml_weights_path,
            tensile_libpath=_tensile_libpath,
        )
        if rc != 0:
            ui.err("err", f"ML-on bench failed for {tag}")
            return rc

        # ML OFF: pure-Origami baseline. With TENSILE_USE_TILEWRIGHT unset
        # (not in this env) the runtime never takes the ML rank path, and
        # with TILEWRIGHT_WEIGHTS no longer set (F6) the eager-init
        # self-disables -- so the baseline log carries ONLY origami's
        # analytical timings, no ML-side prints. F7: ML_RECOMMENDER_NO_EAGER_
        # INIT is therefore no longer needed and is dropped.
        ml_off_env: Dict[str, str] = {}
        log_off = args.out_dir / tag / "ml_off" / "bench.log"
        rc = _run_bench(
            bench_bin,
            norm_yaml,
            log_off,
            device,
            env_extra=ml_off_env,
            iters=args.iters,
            cold_iters=args.cold_iters,
            quiet=args.quiet,
            ml_weights_path=None,
            tensile_libpath=_tensile_libpath,
        )
        if rc != 0:
            ui.err("err", f"ML-off bench failed for {tag}")
            return rc

        calls_on = _parse_selection_times(log_on)
        calls_off = _parse_selection_times(log_off)
        s_on = _summarize("ml_on", calls_on, args.drop_first_call)
        s_off = _summarize("ml_off", calls_off, args.drop_first_call)
        cmp_ = _compare(s_on, s_off)
        # TILEWRIGHT_DIAG marker check: did origami actually load the weights
        # in the ML-on run? If `ml_diag_seen=False` the ML-on log was
        # produced by plain Origami, and any "delta" reading is
        # meaningless. We flag it loudly rather than silently report
        # the (probably ~zero) delta.
        diag_on = _parse_ml_diag(log_on)
        diag_off = _parse_ml_diag(log_off)
        # Optional C++<->Python pick parity. Runs the SAME models.pt
        # the deployed .bin was packed from, scores every yaml problem
        # in Python with the same feature pipeline the deploy
        # runtime uses, and joins by (M,N,K,B) against the
        # [TILEWRIGHT_PICK] lines bench wrote to the ML-on log. A 100%
        # match rate proves the deployed runtime is reading the
        # weights AND the C++ feature extractor matches the Python
        # one byte-for-byte.
        parity: Optional[Dict[str, Any]] = None
        if parity_enabled:
            parity = _compute_parity(
                yaml_path=yp,
                ml_on_log=log_on,
                train_dir=args.train_dir,
                prior_round_dirs=list(args.prior_round_dir or []),
                rocm_libraries_to_deploy=args.rocm_libraries_to_deploy,
                arch=args.arch,
                device=args.torch_device,
                smart_k=int(args.parity_smart_k),
                origami_python_path=args.origami_python_path,
                origami_device=int(args.origami_device),
                quiet=args.quiet,
                workload_filter=workload_filter,
                weight_dtype=args.parity_weight_dtype,
                scale_mode=args.scale_mode,
            )
            # Write the per-yaml parity table as JSON for the operator
            # to grep through after a real run.
            with (args.out_dir / tag / "parity.json").open("w") as f:
                json.dump(parity, f, indent=2)
        summary["per_yaml"][tag] = {
            "yaml": str(yp),
            "n_calls_on": s_on.get("n_calls", 0),
            "n_calls_off": s_off.get("n_calls", 0),
            "compare": cmp_,
            "ml_diag_on": diag_on,
            "ml_diag_off": diag_off,
            "parity": parity,
        }

        if not args.quiet:
            mean_on = s_on.get("sel_us_mean")
            mean_off = s_off.get("sel_us_mean")
            delta = cmp_["delta"]["sel_us_mean"]
            # Loudly warn when the ML recommender clearly didn't run
            # in the ML-on bench (so the operator doesn't trust the
            # delta blindly).
            if not diag_on.get("ml_diag_seen"):
                # HARD FAIL on silent Origami fallback. If the
                # [TILEWRIGHT_DIAG FILE] marker is missing the deployed
                # weights either didn't load (env var wrong / .bin
                # missing) or the deployed origami isn't linked
                # against ml_recommender. Either way the "ML-on"
                # bench is actually plain analytical Origami and
                # every downstream number on this log (selection
                # time delta, parity rate, sel_eff comparison) is
                # meaningless. Stage 7 used to just warn and
                # continue, which silently produced misleading
                # results; we now exit non-zero so CI / the
                # orchestrator can catch it.
                if diag_on.get("no_weights_seen"):
                    ui.err(
                        "ml_diag",
                        f"{tag}  ML-on log shows `[origami::"
                        f"ml_recommender] no weights loaded` -- "
                        f"check TILEWRIGHT_WEIGHTS path. "
                        f"Aborting; refusing to silently fall "
                        f"back to plain Origami.",
                    )
                else:
                    ui.err(
                        "ml_diag",
                        f"{tag}  ML-on log has NO `[TILEWRIGHT_DIAG FILE]` "
                        f"marker; the deployed origami isn't "
                        f"ml_recommender-linked. Aborting; "
                        f"refusing to silently fall back to "
                        f"plain Origami.",
                    )
                return 2
            else:
                ui.ok(
                    "ml_diag",
                    f"{tag}  ML-on weights loaded: "
                    f"n_cells={diag_on.get('n_cells')} "
                    f"n_splits={diag_on.get('n_splits')} "
                    f"arch={diag_on.get('arch')}",
                )
            # Parity rate against models.pt (only when --train-dir
            # was provided).
            if parity is not None:
                n_cpp = parity.get("n_cpp_picks", 0)
                n_py = parity.get("n_py_picks", 0)
                n_cmp = parity.get("n_compared", 0)
                reason = parity.get("reason")
                if n_cmp == 0:
                    # Diagnose the most actionable case clearly: a
                    # successful bench produced ZERO [TILEWRIGHT_PICK] lines
                    # even though Python did compute picks. That
                    # means the deploy bench was built BEFORE
                    # ml_recommender.cpp got the TILEWRIGHT_PICK_LOG
                    # instrumentation -- the operator needs to
                    # rebuild origami + relink hipblaslt-bench.
                    if n_cpp == 0 and n_py > 0 and reason is None:
                        ui.err(
                            "parity",
                            f"{tag}  no `[TILEWRIGHT_PICK]` lines in the "
                            f"ML-on bench log -- the deployed "
                            f"hipblaslt-bench predates this "
                            f"pipeline's TILEWRIGHT_PICK_LOG=1 "
                            f"instrumentation in "
                            f"`origami::ml_recommender::"
                            f"rank_configs`. Rebuild origami "
                            f"(cmake --build "
                            f"shared/origami/build) and relink "
                            f"hipblaslt-bench, then re-run.",
                        )
                    elif reason and "origami_import_failed" in reason:
                        ui.err(
                            "parity",
                            f"{tag}  parity SKIPPED: Python "
                            f"could not import origami. Pass "
                            f"--origami-python-path to a dir "
                            f"that contains a compiled "
                            f"origami/origami.cpython-*.so "
                            f"(detail: {reason})",
                        )
                    else:
                        ui.warn(
                            "parity",
                            f"{tag}  no problems were compared "
                            f"(n_cpp_picks={n_cpp} "
                            f"n_py_picks={n_py} "
                            f"reason={reason or 'unknown'})",
                        )
                else:
                    mr = parity["match_rate"] * 100.0
                    mismatch = n_cmp - parity["n_match"]
                    fn = ui.ok if parity["n_match"] == n_cmp else ui.warn
                    fb = parity.get("n_py_smart_k_fallback", 0) or 0
                    fb_str = f"  smart_K_fallback={fb}/{n_py}" if fb else ""
                    fn(
                        "parity",
                        f"{tag}  match={mr:.1f}% "
                        f"({parity['n_match']}/{n_cmp}) "
                        f"-- C++ pick == Python pick. "
                        f"mismatched={mismatch}{fb_str}; see "
                        f"{args.out_dir / tag / 'parity.json'}",
                    )
                    if fb:
                        ui.info(
                            "parity",
                            f"{tag}  {fb}/{n_py} problems fell "
                            f"back to LDS+is_kernel_feasible "
                            f"only (cell's K-list had no "
                            f"feasibility-compatible kernel for "
                            f"the problem). Indicates the "
                            f"K-selector should diversify NTA/"
                            f"NTB across the cell's GEMM shapes.",
                        )
            if mean_on is None or mean_off is None or delta is None:
                ui.warn(
                    "timing",
                    f"{tag}  parse missed selection-time lines "
                    f"(n_on={len(calls_on)} n_off={len(calls_off)})",
                )
            else:
                ui.ok(
                    "timing",
                    f"{tag}  ml_on={mean_on:8.2f}us  "
                    f"ml_off={mean_off:8.2f}us  "
                    f"delta={delta:+8.2f}us  "
                    f"(n={s_on['n_calls']}/{s_off['n_calls']}"
                    + (", first-call dropped" if args.drop_first_call else "")
                    + ")",
                )

    # ── direct rank_configs() call-time microbench (CPU-only) ─────────
    # Complements the GPU end-to-end selection time above by isolating
    # the cost of the scorer itself. Runs once per stage-7 invocation
    # over the deploy library's real candidate set. Never fatal.
    if getattr(args, "measure_rank_call_time", False):
        wbin = args.ml_weights_path
        if wbin is None:
            # Resolve the per-stem deployed .bin matching this config's
            # dtype/layout from weights/<arch>/<arch>/ (productized layout);
            # falls back to the legacy global path for older checkouts.
            wbin = _resolve_deployed_weights_bin(
                args.rocm_libraries_to_deploy, args.arch, workload_filter
            )
        userargs = args.rank_time_userargs_yaml or _find_userargs_yaml(
            args.rocm_libraries_to_deploy, args.arch, workload_filter
        )
        rank_time = _measure_rank_call_time(
            origami_repo=args.rocm_libraries_to_deploy,
            weights_path=wbin,
            userargs_yaml=userargs,
            arch=args.arch,
            iters=int(args.rank_time_iters),
            warmup=int(args.rank_time_warmup),
            out_dir=args.out_dir,
            quiet=args.quiet,
        )
        summary["rank_configs_call_time"] = rank_time
        if not args.quiet:
            if rank_time.get("skipped"):
                ui.warn(
                    "rank-time",
                    f"call-time microbench skipped: " f"{rank_time['skipped']}",
                )
            else:
                # Emit the microbench's own table verbatim (origami vs
                # ml_rec per-call us), indented under a banner -- the
                # aligned `path / n / min / median / mean / max` layout
                # bench_selection_time.cpp prints.
                ui.ok(
                    "rank-time",
                    "origami::rank_configs vs "
                    "ml_recommender::rank_configs  (CPU wall-clock "
                    "per rank_configs() call):",
                )
                for _ln in (rank_time.get("stdout") or "").splitlines():
                    if _ln.strip():
                        print(f"        {_ln}", flush=True)

    out_json = args.out_dir / "timings_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok("done", f"timings_summary.json -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
