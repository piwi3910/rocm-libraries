# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pipeline_v3 end-to-end orchestrator.

ONE required input: the cell config yaml. Everything else (devices,
build paths, per-stage knobs, training hyperparams, # of active rounds)
is read from there. The run directory is auto-generated from the config
basename + the current date-time so successive runs of the same config
never collide.

Modes:

    initial   ONE round, round_0 only (full grid seed + bench + train).
    active    ONE additional round on top of an existing run. The
              orchestrator auto-discovers every `round_*/` already
              under `--run-dir` and uses them as priors; the new
              round_R is appended where R = (# of discovered rounds).
    full      EVERY round in one shot. round_0 in `initial` mode, then
              round_1 .. round_{N-1} in `active` mode where N is
              `full.total_rounds` from the config. All N rounds live
              under the SAME run dir.

Priors are ALWAYS read from `<run-dir>/round_*/` -- there's no
"point at a different run's rounds" knob. To branch one run off
another, `cp -r runs/old runs/new` first, then point `--run-dir` at
`runs/new`.

Run-dir layout (NEW stage numbering -- see "Stages" below):

    runs/<config_stem>_<YYYYMMDD_HHMMSS>/
        config_used.yaml
        round_0/                          (INITIAL round)
            stage01/ stage01.log          gen shapes
            stage02/ stage02.log          probe
            stage03/ stage03.log          offline tuning (bench)
            stage04/ stage04.log          enrich
            stage05/ stage05.log          TRAIN initial models
            stage06/ stage06.log          (ONLY if --deploy is passed
                                          on the CLI; the auto-deploy
                                          gates -- target_sel_eff /
                                          on_final_round -- need a
                                          held-out signal that doesn't
                                          exist in round 0 because
                                          stage04b is skipped, so they
                                          stay disabled here.)
        round_1/                          (ACTIVE round; only in
                                           --mode active / full)
            stage01/ stage01.log          gen NEW shapes (deduped vs
                                          every prior round)
            stage02/ stage02.log          probe
            stage03/ stage03.log          bench
            stage04/ stage04.log          enrich
            stage04b/ stage04b.log        VALIDATE round_(R-1)'s
                                          model on round_R's NEW
                                          (unknown) shapes via the
                                          DEPLOYED pipeline (Origami
                                          top-K + ML re-rank) ->
                                          decide pass / retrain /
                                          split. Conceptually "stage
                                          4.5" -- the `b` suffix
                                          encodes "between 4 and 5"
                                          without the decimal-in-a-
                                          filename mess.
            stage05/ stage05.log          RETRAIN failing cells with
                                          cross-round data
                                          augmentation; fresh weights
                                          (no warm-start)
            stage06/ stage06.log          (only if deploy triggered)
        round_2/ ...                      same as round_1
        ...

Stages (logical, after the stage renumbering):

  | # | name                                | role |
  |---|-------------------------------------|------|
  | 1 | stage01_generate_OOB_shapes         | gen shapes |
  | 2 | stage02_probe                       | probe iters |
  | 3 | stage03_load_balance_offline_tuning | bench |
  | 4 | stage04_convert_to_enriched_dataset | enrich |
  | 4.5 | stage04b_iterate_active_learning  | VALIDATE +
  |     | (active rounds only)              | decide |
  | 5 | stage05_train                       | train /
  |   |                                     | retrain |
  | 6 | stage06_deploy_to_origami           | deploy |
  | 7 | stage07_optional_validate_selection_time | optional;
  |   |                                     | not run by orchestrator |
  | 8 | stage08_optional_validate_selection_efficiency | optional;
  |   |                                     | not run by orchestrator |

In active rounds, stage 4.5 (validate) runs BEFORE stage 5
(retrain) -- the active-learning loop validates the PRIOR round's
model on this round's new data, decides which cells need work,
THEN retrains. The stage NUMBER (4.5) reflects this conceptual
position: between data collection (4) and training (5). On disk
the directory is `stage04b/` and the python file is
`stages/stage04b_iterate_active_learning.py` -- the `b` suffix
is preferred over `045` because `045` is easily misread as the
unrelated number forty-five.

Resume + partial reruns:

    --from-round R    Skip rounds 0..R-1 entirely (their outputs are
                      already on disk). Round R restarts at --from-stage,
                      every round > R restarts at stage 1.
                      Default: 0.  Only meaningful in --mode full.
    --from-stage N    Within the resumed round, skip stages whose
                      LOGICAL number is < N. Accepts the floats:
                          1, 2, 3, 4, 4.5, 5, 6
                      Default: 1. (stage 4.5 only exists in active
                      rounds; passing 4.5 in round 0 effectively
                      starts at stage 5 because round 0 has no
                      stage04b.)

Examples:

    # Round 0 only:
    python run_orchestrator.py \\
        --config configs/gfx950_bbs_tn.yaml \\
        --mode initial

    # One active round on top of an earlier run (auto-discovers
    # every round_*/ already under --run-dir as priors):
    python run_orchestrator.py \\
        --config configs/gfx950_bbs_tn.yaml \\
        --mode active \\
        --run-dir runs/<earlier_run>

    # Full run, every active round in one shot:
    python run_orchestrator.py \\
        --config configs/gfx950_bbs_tn.yaml \\
        --mode full

    # Resume a crashed run starting at round_2 stage_4:
    python run_orchestrator.py \\
        --config configs/gfx950_bbs_tn.yaml \\
        --mode full \\
        --run-dir runs/<existing_run> \\
        --from-round 2 --from-stage 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent
STAGES = THIS_DIR / "stages"
RUNS = THIS_DIR / "runs"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lib import ui  # noqa: E402


# Module-level timing ledger. main() installs the live instance; the
# helpers below (and _run_round / _run_validate_mode) call
# `_TIMING.stage_begin(...)` and `_TIMING.stage_end(...)` to record
# every stage's elapsed time. Initialised to a no-op so unit tests
# that import this module don't crash before main() runs.
_TIMING: "Any" = None


# ── small helpers ───────────────────────────────────────────────────────────


def _devices_str(devices: List[int]) -> str:
    return ",".join(str(d) for d in devices)


def _autoname_run_dir(config_path: Path) -> Path:
    base = config_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS / f"{base}_{ts}"


def _need(d: Dict[str, Any], path: List[str], where: str) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            ui.err("cfg", f"missing required key `{'.'.join(path)}` " f"in {where}")
            raise SystemExit(2)
        cur = cur[k]
    return cur


def _opt(d: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


_UNEXPANDED_VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def _expand_config_env(node: Any, missing: set) -> Any:
    """Expand `${VAR}` / `$VAR` / leading `~` in every string of a loaded config.

    Host-specific paths (runtime.rocm_libraries_root,
    deploy.rocm_libraries_to_deploy, stage08 dataset paths, ...) are written
    as environment variables so a config is not tied to one machine's
    directory layout. Names left unexpanded are collected in `missing`: the
    caller fails on them rather than letting a literal "${VAR}" through as a
    path that merely does not exist.
    """
    if isinstance(node, dict):
        return {k: _expand_config_env(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_config_env(v, missing) for v in node]
    if isinstance(node, str) and ("$" in node or node.startswith("~")):
        out = os.path.expanduser(os.path.expandvars(node))
        missing.update(_UNEXPANDED_VAR_RE.findall(out))
        return out
    return node


def _run(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    """Tee subprocess stdout to a per-stage log file AND to the
    operator's real terminal. The per-stage log captures the
    subprocess's exact output. The terminal write goes to the
    UNWRAPPED stdout (`_ORIG_STDOUT`) -- NOT `sys.stdout`, which is
    the orchestrator's tee. We deliberately route subprocess output
    away from the orchestrator tee so `orchestrator.log` records
    only `run_orchestrator.py`'s own prints. Stage subprocesses get
    their own dedicated `<stage>.log`; if you need to see what a
    child printed, read THAT file, not orchestrator.log.

    Both streams are LINE BUFFERED: the child runs under `python -u`
    (already), Popen's `bufsize=1` makes the pipe yield one line at
    a time, and `buffering=1` on the log file forces a flush on
    every newline -- so `tail -f stageN.log` lags by at most one
    line."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ui.info("run", " ".join(str(c) for c in cmd))
    t0 = time.time()
    # `_ORIG_STDOUT` is the OS-level stdout captured before main()
    # installed the _Tee wrapper. Writing here puts subprocess
    # output on the terminal but NOT in orchestrator.log.
    term_out = _ORIG_STDOUT if _ORIG_STDOUT is not None else sys.stdout
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                term_out.write(line)
                term_out.flush()
            except Exception:
                pass
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    el = time.time() - t0
    if rc == 0:
        ui.ok("run", f"completed in {ui.fmt_dur(el)}")
    else:
        ui.err("run", f"FAILED (rc={rc}) after {ui.fmt_dur(el)}")
    return rc


# ── orchestrator stdout tee + ANSI stripper ──────────────────────────────────
#
# We mirror EVERY print the orchestrator (and every subprocess it
# spawns via _run()) makes onto a single `<run-dir>/orchestrator.log`
# file. This is the operator's one-stop file for "what happened on
# this run"; per-stage logs under each round still exist for drill-
# down. ANSI colour codes are stripped before they hit the log so
# the file stays grep-friendly (the terminal still sees them).

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# The REAL `sys.stdout` snapshot, captured before main() installs the
# `_Tee` wrapper. Subprocess output (everything `_run()` forwards
# from child stdout) is written here directly so it appears on the
# operator's terminal but DOES NOT land in `orchestrator.log`. The
# orchestrator's own prints go through `sys.stdout`, which IS the
# tee, so they DO land in `orchestrator.log` -- exactly what we want.
_ORIG_STDOUT: Optional[Any] = None


class _Tee:
    """Duplicate writes to two streams. The first is the original
    `sys.stdout`/`sys.stderr` (preserves colour, tty behaviour); the
    second is a plain-text log file (ANSI escapes stripped)."""

    def __init__(self, terminal: Any, logfile: Any) -> None:
        self.terminal = terminal
        self.logfile = logfile

    def write(self, data: str) -> int:
        n = self.terminal.write(data)
        try:
            self.logfile.write(_ANSI_RE.sub("", data))
        except Exception:
            pass
        return n if n is not None else len(data)

    def flush(self) -> None:
        self.terminal.flush()
        try:
            self.logfile.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        # Defer to the real terminal so lib/ui keeps colouring its
        # output. (`_C._tty` was already cached at import time
        # against the original sys.stdout anyway.)
        return bool(getattr(self.terminal, "isatty", lambda: False)())


class _StageTimingTracker:
    """In-memory ledger of per-(round, stage) elapsed seconds. The
    orchestrator prints a final stacked table at the end of main()
    so the operator can spot the dominant cost across the whole
    multi-round run at a glance."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.run_t0 = time.time()
        # Per-round running timer state.
        self._round_t0: Optional[float] = None
        self._round_idx: Optional[int] = None
        # Per-stage running timer state.
        self._stage_t0: Optional[float] = None
        self._stage_name: Optional[str] = None

    def round_begin(self, round_idx: int) -> None:
        self._round_t0 = time.time()
        self._round_idx = round_idx

    def round_end(self, rc: int) -> float:
        if self._round_t0 is None or self._round_idx is None:
            return 0.0
        el = time.time() - self._round_t0
        self.rows.append(
            {
                "round": self._round_idx,
                "stage": "(round total)",
                "rc": rc,
                "elapsed_s": el,
            }
        )
        self._round_t0 = None
        self._round_idx = None
        return el

    def stage_begin(self, name: str) -> None:
        self._stage_t0 = time.time()
        self._stage_name = name

    def stage_end(self, rc: int = 0) -> float:
        if self._stage_t0 is None or self._stage_name is None:
            return 0.0
        el = time.time() - self._stage_t0
        self.rows.append(
            {
                "round": self._round_idx if self._round_idx is not None else -1,
                "stage": self._stage_name,
                "rc": rc,
                "elapsed_s": el,
            }
        )
        ui.info("stage", f"{self._stage_name} completed in {ui.fmt_dur(el)}")
        self._stage_t0 = None
        self._stage_name = None
        return el

    def total_elapsed_s(self) -> float:
        return time.time() - self.run_t0

    # ── thin helpers wrap ui.banner so each stage's begin/end is
    # always paired even if the caller forgets. Mark `rc=0` by
    # default; pass the actual rc when wrapping a `_run()` call.

    def banner(self, name: str, color: str = ui.C.BLUE) -> None:
        self.stage_begin(name)
        ui.banner(name, color)

    def end_ok(self) -> None:
        self.stage_end(0)

    def end_rc(self, rc: int) -> None:
        self.stage_end(rc)

    def print_summary(self) -> None:
        if not self.rows:
            return
        # Idempotency: don't reprint the same table twice if both the
        # full-mode tail AND the auto-validate path call us.
        if getattr(self, "_summary_printed", False):
            return
        self._summary_printed = True
        ui.banner("pipeline_v3  timing summary", ui.C.HEAD)
        # Group rows by round, preserve insertion order.
        from collections import OrderedDict

        by_round: "OrderedDict[Any, List[Dict[str, Any]]]" = OrderedDict()
        for r in self.rows:
            by_round.setdefault(r["round"], []).append(r)
        # Column widths.
        name_w = max(20, *(len(str(r["stage"])) for r in self.rows))
        for rd, rows in by_round.items():
            header = f"round_{rd}" if isinstance(rd, int) and rd >= 0 else "post-rounds"
            ui.info("timing", header)
            for r in rows:
                tag = "ok " if r["rc"] == 0 else f"rc={r['rc']:>3}"
                line = (
                    f"   {str(r['stage']):<{name_w}}  "
                    f"{ui.fmt_dur(r['elapsed_s']):>8}  [{tag}]"
                )
                if r["stage"] == "(round total)":
                    line += "   <-- round wall-clock"
                print(line, flush=True)
        ui.ok("timing", f"WALL-CLOCK TOTAL: " f"{ui.fmt_dur(self.total_elapsed_s())}")


class _FlowListDumper(yaml.SafeDumper):
    """Dumper that emits scalar-only lists (e.g. `devices: [0, 1, 2, ...]`)
    in FLOW style and keeps dicts in BLOCK style."""


def _scalar_list_repr(dumper, data):
    flow = all(isinstance(x, (int, float, bool, str)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_FlowListDumper.add_representer(list, _scalar_list_repr)


# ── derived paths ───────────────────────────────────────────────────────────


def _derive_paths(cfg: Dict[str, Any], cfg_path: Path) -> Tuple[Path, Path, Path, str]:
    """rocm_libraries_root + arch → (bench_binary, dat_dir, match_table,
    origami_python_path). Standard rocm-libraries layout:
        <root>/projects/hipblaslt/build/release/clients/hipblaslt-bench
        <root>/projects/hipblaslt/build/release/Tensile/library/<arch>/
        <root>/projects/hipblaslt/build/release/device-library/MatchTable.yaml
        <root>/shared/origami/python/src
    """
    runtime = _need(cfg, ["runtime"], str(cfg_path))
    arch = str(_need(cfg, ["arch"], str(cfg_path)))
    rocm_root = Path(_need(runtime, ["rocm_libraries_root"], "runtime"))
    if not rocm_root.exists():
        ui.err("cfg", f"runtime.rocm_libraries_root does not exist: " f"{rocm_root}")
        raise SystemExit(2)
    build = rocm_root / "projects/hipblaslt/build/release"
    bench_binary = build / "clients/hipblaslt-bench"
    dat_dir = build / "Tensile/library" / arch
    match_table = build / "device-library/MatchTable.yaml"
    origami_pp = str(rocm_root / "shared/origami/python/src")
    for p, name in (
        (bench_binary, "bench_binary"),
        (dat_dir, "dat_dir"),
        (match_table, "match_table"),
    ):
        if not p.exists():
            hint = ""
            # An ASIC-revision arch (gfx1250v0) builds into its own
            # Tensile-<arch>/ tree, but hipblaslt-bench's revision promotion
            # only looks under Tensile/library/. Falling back to the split
            # tree here would pair this arch's .dat with whatever library the
            # bench actually loads, so stop and say how to link it instead.
            split = build / f"Tensile-{arch}" / "library" / arch
            if name == "dat_dir" and split.exists():
                hint = (
                    f"\n      {arch} is built at {split} but not under "
                    f"Tensile/library/, where hipblaslt-bench's ASIC-revision "
                    f"promotion loads it from. Link it in once:\n"
                    f"        cp -al {split} {p}"
                )
            ui.err(
                "cfg",
                f"derived {name} missing: {p}{hint}\n"
                f"      check runtime.rocm_libraries_root / arch in "
                f"{cfg_path}",
            )
            raise SystemExit(2)
    return bench_binary, dat_dir, match_table, origami_pp


# ── one round ───────────────────────────────────────────────────────────────


def _run_round(
    *,
    round_idx: int,
    round_mode: str,
    prior_round_dirs: List[Path],
    run_dir: Path,
    cfg: Dict[str, Any],
    cfg_path: Path,
    devices: List[int],
    arch: str,
    bench_binary: Path,
    dat_dir: Path,
    match_table: Path,
    origami_pp: str,
    from_stage: float,
    force_deploy: bool,
    is_final_round: bool,
    # cascaded config knobs
    target_per_cell: int,
    shapes_per_cell: int,
    seed_seed: int,
    n_strata: int,
    bench_blocks_per_gpu: int,
    train_epochs: int,
    train_device: str,
    train_eval_frac: float,
    train_smart_k: int,
    train_min_cell_gemms: int,
    train_smart_k_max: Optional[int] = None,
    train_smart_k_min: int = 4,
    train_smart_k_saturation_eps: float = 0.005,
    sel_eff_threshold: str,
    validate_min_cell_gemms: int,
    split_floor: str,
    split_min_improvement: float,
    split_after_attempts: int,
    max_split_depth: int,
    structural_split_min_octaves: float,
    structural_split_max_depth: int,
    # deploy knobs
    deploy_target_sel_eff: Optional[float],
    deploy_on_final: bool,
    deploy_rocm_libraries_to_deploy: Optional[str],
    deploy_apply_patch: bool,
    deploy_build_after_apply: bool,
    deploy_build_jobs: Optional[int],
    deploy_weight_dtype: str = "bf16",
) -> Dict[str, Any]:
    """Returns a dict with at least:
    rc                 0 on success, non-zero on failure
    held_out_sel_eff   this round's HELD-OUT deployed sel_eff,
                       geomean over stage04b/decisions.json's
                       per-cell `sel_eff_new` weighted by n_eval.
                       NaN in round 0 (no stage04b runs there).
    deployed           True if stage06_deploy ran in this round
    target_reached     True if held_out_sel_eff >= deploy_target_sel_eff
    """
    round_dir = run_dir / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)
    stage01_dir = round_dir / "stage01"
    stage02_dir = round_dir / "stage02"
    stage03_dir = round_dir / "stage03"
    # The run's hipblaslt scale mode (`hipblaslt.scaleA`), threaded into
    # every stage that walks .dat libraries. ABSENT KEY -> None -> the
    # stages keep their pre-MX "load every library" behavior. Present ->
    # each stage loads ONLY the libraries that mode selects, which is what
    # keeps the gfx1250 MX (scaleA: 3) and non-MX (scaleA: 1) F8 TN pools
    # from merging under their identical dtype/layout key (see lib/dat.py).
    _scale_mode = _opt(cfg, ["hipblaslt", "scaleA"], None)
    scale_mode_args: List[str] = (
        [] if _scale_mode is None else ["--scale-mode", str(int(_scale_mode))]
    )

    stage04_dir = round_dir / "stage04"
    stage05_dir = round_dir / "stage05"
    # Output dirs follow the NEW stage numbering:
    #   stage04b = active-round VALIDATE (was stage06 in the old layout)
    #   stage05  = TRAIN (initial in round 0; retrain in active rounds)
    #   stage06  = DEPLOY (was stage07 in the old layout)
    # The `b` suffix on `04b` is deliberate: it encodes "this stage
    # sits between 4 and 5, i.e. logical stage 4.5" without putting a
    # decimal in a filename (`stage045` reads as "stage forty-five",
    # which is wrong and confusing).
    stage04b_dir = round_dir / "stage04b"
    stage06_dir = round_dir / "stage06"  # deploy (was stage07)

    # --from-stage gates each stage's skip. The NEW logical numbering:
    #   1   gen shapes        2   probe        3   bench (offline tune)
    #   4   enrich            4.5 validate (active rounds only)
    #   5   train             6   deploy
    # `from_stage` is a float so 4.5 (validate) sits naturally between
    # enrich (4) and train (5).
    skip_gen = from_stage > 1.0
    skip_probe = from_stage > 2.0
    skip_bench = from_stage > 3.0
    skip_enrich = from_stage > 4.0
    skip_validate = from_stage > 4.5  # active-round validate
    skip_train = from_stage > 5.0  # stage05 (train / retrain)
    skip_deploy = from_stage > 6.0  # stage06 (deploy)

    ui.banner(
        f"pipeline_v3  round_{round_idx}  mode={round_mode}  "
        f"from_stage={from_stage}",
        ui.C.HEAD,
    )
    if round_mode == "active":
        ui.info("cfg", f"prior_round_dirs: " f"{[str(p) for p in prior_round_dirs]}")
    _TIMING.round_begin(round_idx)

    def _ret(
        rc: int,
        *,
        held_out_sel_eff: float = float("nan"),
        deployed: bool = False,
        target_reached: bool = False,
    ) -> Dict[str, Any]:
        # Close out the per-round wall-clock entry no matter how
        # this round exits (success, stage failure, return-early).
        el = _TIMING.round_end(rc)
        (ui.ok if rc == 0 else ui.err)(
            "round",
            f"round_{round_idx} {'completed' if rc == 0 else 'FAILED'} "
            f"in {ui.fmt_dur(el)} (rc={rc})  "
            f"[cumulative E2E so far: "
            f"{ui.fmt_dur(_TIMING.total_elapsed_s())}]",
        )
        return {
            "rc": rc,
            "held_out_sel_eff": held_out_sel_eff,
            "deployed": deployed,
            "target_reached": target_reached,
        }

    # stage01 ────────────────────────────────────────────────────────────────
    if not skip_gen:
        _TIMING.banner("Stage 1 -- generate shapes")
        cmd = [
            sys.executable,
            "-u",
            str(STAGES / "stage01_generate_OOB_shapes.py"),
            "--mode",
            round_mode,
            "--out-dir",
            str(stage01_dir),
            "--config-yaml",
            str(cfg_path),
            "--seed",
            str(seed_seed),
            "--n-strata",
            str(n_strata),
        ]
        if round_mode == "initial":
            cmd += ["--target-per-cell", str(target_per_cell)]
        else:
            if not prior_round_dirs:
                ui.err(
                    "err",
                    "active round requires at least one " "prior round dir (got none)",
                )
                return _ret(2)
            for rd in prior_round_dirs:
                cmd += ["--prior-round-dir", str(rd)]
            cmd += [
                "--shapes-per-cell",
                str(shapes_per_cell),
                # Guarantee every (split) leaf gets >= min_cell_gemms
                # shapes so stage05 trains it -- no untrained/orphan
                # leaves (O1 coverage fix).
                "--min-cell-gemms",
                str(train_min_cell_gemms),
            ]
        rc = _run(cmd, round_dir / "stage01.log")
        _TIMING.end_rc(rc)
        if rc != 0:
            return _ret(1)
    else:
        ui.warn("stage", f"Stage 1 skipped (--from-stage > 1)")

    # stage02 ────────────────────────────────────────────────────────────────
    # Adaptive timing makes stage02 REDUNDANT BY CONSTRUCTION. stage02's whole
    # job was to measure each shape's `us` once and back-calculate an iteration
    # count that would make the real bench run for ~`probe.duration_us` per
    # kernel. With `bench.adaptive.enabled`, hipblaslt-bench does that itself,
    # per kernel, using the variance it actually observes -- it warms up, takes
    # timed samples, and stops when the mean's relative standard error is under
    # `noise_threshold`. Running stage02 anyway would cost a full extra pass
    # over every shape to produce `iters`/`cold_iters` values that the
    # renderer then DROPS, because the bench rejects adaptive combined with a
    # fixed iteration count.
    #
    # stage03 already falls back to stage01's shapes.yaml when
    # stage02/shapes_probed.yaml is absent, so skipping is a clean no-op.
    _adaptive_on = bool(
        ((cfg.get("bench") or {}).get("adaptive") or {}).get("enabled", False)
    )
    if _adaptive_on and not skip_probe:
        skip_probe = True
        ui.info(
            "stage",
            "Stage 2 SKIPPED: bench.adaptive.enabled=true -- "
            "hipblaslt-bench derives its own per-kernel timing, "
            "so there is no iteration count left to calibrate",
        )
    if not skip_probe:
        _TIMING.banner("Stage 2 -- probe (measure us, calibrate iters)")
        cmd = [
            sys.executable,
            "-u",
            str(STAGES / "stage02_probe.py"),
            "--in-dir",
            str(stage01_dir),
            "--out-dir",
            str(stage02_dir),
            "--config-yaml",
            str(cfg_path),
            "--bench-binary",
            str(bench_binary),
            "--devices",
            _devices_str(devices),
        ]
        rc = _run(cmd, round_dir / "stage02.log")
        _TIMING.end_rc(rc)
        if rc != 0:
            return _ret(1)
    elif not _adaptive_on:
        ui.warn("stage", "Stage 2 skipped (--from-stage > 2)")

    # stage03 ────────────────────────────────────────────────────────────────
    if not skip_bench:
        _TIMING.banner("Stage 3 -- per-GPU bench launcher")
        bench_in_dir = (
            stage02_dir
            if (stage02_dir / "shapes_probed.yaml").exists()
            else stage01_dir
        )
        cmd = [
            sys.executable,
            "-u",
            str(STAGES / "stage03_load_balance_offline_tuning.py"),
            "--in-dir",
            str(bench_in_dir),
            "--out-dir",
            str(stage03_dir),
            "--bench-binary",
            str(bench_binary),
            "--devices",
            _devices_str(devices),
            "--blocks-per-gpu",
            str(bench_blocks_per_gpu),
            "--stall-timeout-s",
            str(_opt(cfg, ["bench", "stall_timeout_s"], 300.0)),
            "--startup-grace-s",
            str(_opt(cfg, ["bench", "startup_grace_s"], 900.0)),
            # stage03 reads bench.warmup_pass (and any future bench.*
            # knob) from the run config. Without this it silently falls
            # back to its own argparse default, which is ON -- the run
            # then reports "warmup_pass : True (default)" while the
            # config says false.
            "--config-yaml",
            str(cfg_path),
        ]
        rc = _run(cmd, round_dir / "stage03.log")
        _TIMING.end_rc(rc)
        if rc != 0:
            return _ret(1)
    else:
        ui.warn("stage", "Stage 3 skipped (--from-stage > 3)")

    # stage04 ────────────────────────────────────────────────────────────────
    if not skip_enrich:
        _TIMING.banner("Stage 4 -- enrich logs")
        env = os.environ.copy()
        if origami_pp:
            prev = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = origami_pp + (":" + prev if prev else "")
        cmd = [
            sys.executable,
            "-u",
            str(STAGES / "stage04_convert_to_enriched_dataset.py"),
            "--log-dir",
            str(stage03_dir / "logs"),
            "--out-dir",
            str(stage04_dir),
            "--dat-dir",
            str(dat_dir),
            "--match-table",
            str(match_table),
            # stage04's origami arch-only hardware fallback used to be
            # hardcoded to gfx950; pass the run's real arch so a failed
            # device probe can't silently swap in the wrong hardware.
            "--arch",
            arch,
            *scale_mode_args,
            "--progress-every",
            "60",
        ]
        rc = _run(cmd, round_dir / "stage04.log", env=env)
        _TIMING.end_rc(rc)
        if rc != 0:
            return _ret(1)
    else:
        ui.warn("stage", "Stage 4 skipped (--from-stage > 4)")

    # stage05 / stage06 ──────────────────────────────────────────────────────
    def _train_enriched_args() -> List[str]:
        """Every prior round's stage04 dir + this round's stage04 dir.
        Cross-round data augmentation."""
        out: List[str] = []
        for rd in prior_round_dirs:
            ed = rd / "stage04"
            if ed.exists():
                out += ["--enriched-csv-dir", str(ed)]
        out += ["--enriched-csv-dir", str(stage04_dir)]
        return out

    def _prior_round_dir_args() -> List[str]:
        out: List[str] = []
        for rd in prior_round_dirs:
            out += ["--prior-round-dir", str(rd)]
        return out

    def _smart_k_adaptive_args() -> List[str]:
        """Opt-in adaptive-K flags for stage05. Empty when
        `train.smart_k_max` is unset (legacy fixed-K). When set, the
        per-cell signature list is grown to k_max then knee-truncated
        in [k_min, k_max] -- a selection-time win for saturated cells
        and an OOD-coverage win for cells still improving. Parity-safe
        (the runtime reads whatever-length list ships)."""
        if train_smart_k_max is None:
            return []
        return [
            "--smart-k-max",
            str(int(train_smart_k_max)),
            "--smart-k-min",
            str(int(train_smart_k_min)),
            "--smart-k-saturation-eps",
            str(float(train_smart_k_saturation_eps)),
        ]

    def _train_parallel_args() -> List[str]:
        """stage05 per-cell training parallelism (thread-capped process
        pool over train_cell). train.workers>1 trains cells concurrently;
        threads_per_worker=0 auto-sizes to cpu_count//workers. Bit-for-bit
        identical to sequential (validated), just faster."""
        return [
            "--train-workers",
            str(int(_opt(cfg, ["train", "workers"], 1))),
            "--train-threads-per-worker",
            str(int(_opt(cfg, ["train", "threads_per_worker"], 0))),
        ]

    if round_mode == "initial":
        if not skip_train:
            _TIMING.banner("Stage 5 -- train all populated cells")
            cmd = [
                sys.executable,
                "-u",
                str(STAGES / "stage05_train.py"),
                # Same reason as stage04b: without these, stage05 falls back to
                # its --origami-library-dir default, the containing checkout's
                # gfx950 library -- the wrong arch for any other run. If that
                # is not built, compute_origami_sel_eff() and
                # evaluate_deployed_sel_eff() see no configs and return
                # sel_eff=NaN, n_eval=0, so the ORIGAMI comparison column is
                # blank for every cell; if it is, they rank the wrong arch.
                "--origami-library-dir",
                str(dat_dir),
                "--origami-python-path",
                origami_pp,
                *_train_enriched_args(),
                *_prior_round_dir_args(),
                # --arch matters here: get_mi_latency() keys the MI-latency
                # feature table by arch. Omitting it leaves this stage on the
                # gfx950 default while stage07/08 use the real arch, which
                # silently skews training features against validation ones.
                "--arch",
                arch,
                "--output-dir",
                str(stage05_dir),
                "--epochs",
                str(train_epochs),
                "--device",
                train_device,
                "--eval-sample-frac",
                str(train_eval_frac),
                "--smart-k",
                str(train_smart_k),
                *_smart_k_adaptive_args(),
                *_train_parallel_args(),
                *scale_mode_args,
                "--min-cell-gemms",
                str(train_min_cell_gemms),
            ]
            rc = _run(cmd, round_dir / "stage05.log")
            _TIMING.end_rc(rc)
            if rc != 0:
                return _ret(1)
        else:
            ui.warn("stage", "Stage 5 skipped (--from-stage > 5)")

        # ── Round 0 ends here. ────────────────────────────────────────
        # We intentionally do NOT run stage04b in round 0. stage04b's
        # job is to evaluate a PRIOR round's model on NEW unknown
        # shapes; in round 0 there are no prior models and no
        # held-out shapes. A "snapshot" eval on the just-trained
        # model's own training data is misleading -- it returns
        # training-set accuracy and (worse) drives retrain_cells.txt
        # so the next round would retrain cells based on training
        # noise. The first real generalization signal lands in
        # round 1's stage04b (validates round_0's model on round_1's
        # new shapes).
        if not skip_validate:
            ui.info(
                "round0",
                "Stage 4.5 NOT run in round 0 (no prior "
                "model + no held-out shapes yet). Round 0 ends "
                "after stage 5; round 1 begins with stage 1.",
            )

    else:
        # Active round: stage04b validates the most-recent prior model
        # against this round's NEW data, then stage05 retrains only the
        # cells that failed. Training data is the union of every round.
        prior_train = prior_round_dirs[-1] / "stage05"

        if not skip_validate:
            _TIMING.banner("Stage 4.5 -- validate prior model on new data")
            cmd = [
                sys.executable,
                "-u",
                str(STAGES / "stage04b_iterate_active_learning.py"),
                "--models-dir",
                str(prior_train),
                # --arch matters here: get_mi_latency() keys the MI-latency
                # feature table by arch. Omitting it leaves this stage on the
                # gfx950 default while stage07/08 use the real arch, which
                # silently skews training features against validation ones.
                "--arch",
                arch,
                "--enriched-csv-dir",
                str(stage04_dir),
                "--out-dir",
                str(stage04b_dir),
                "--sel-eff-threshold",
                sel_eff_threshold,
                "--split-floor",
                split_floor,
                "--split-min-improvement",
                str(split_min_improvement),
                "--split-after-attempts",
                str(split_after_attempts),
                "--max-split-depth",
                str(max_split_depth),
                "--structural-split-min-octaves",
                str(structural_split_min_octaves),
                "--structural-split-max-depth",
                str(structural_split_max_depth),
                "--round-idx",
                str(round_idx),
                "--min-cell-gemms",
                str(validate_min_cell_gemms),
                "--smart-k",
                str(train_smart_k),
                # The Origami baseline. WITHOUT these two, stage04b falls back
                # to its --origami-library-dir / --origami-python-path
                # defaults, the containing checkout's gfx950 library -- the
                # wrong arch for any other run, and maybe not built. (If it
                # is built, the baseline silently ranks the wrong arch's
                # kernels.) If it is absent the load throws, the baseline is
                # disabled, and evaluate_deployed_sel_eff() gets an EMPTY
                # config dict -- so it can rank nothing, n_eval is 0, and
                # sel_eff comes back NaN for EVERY cell. Downstream of that:
                #   * every cell decided "retrain" (reason sel_eff_nan_on_new_data)
                #   * ZERO cells can ever become split candidates
                #   * held_out_sel_eff / origami_held_out_sel_eff = NaN,
                #     n_cells = 0, so the report omits both held-out fields
                # These are the same values stage04 already gets.
                "--origami-library-dir",
                str(dat_dir),
                "--origami-python-path",
                origami_pp,
                *scale_mode_args,
                "--device",
                train_device,
            ]
            # On the final round there is no next round to train a split's
            # children, so disable splitting to avoid untrained (orphaned)
            # leaves in the deployed split tree.
            if is_final_round:
                cmd += ["--final-round"]
            # Pass every prior round's stage04b/decisions.json (in age
            # order, oldest first) so stage04b can count cumulative
            # retrain attempts (gates --split-after-attempts).
            for pd in prior_round_dirs:
                pdj = pd / "stage04b" / "decisions.json"
                if pdj.exists():
                    cmd += ["--prior-decisions", str(pdj)]
            # Pass the round_(R-2) models.pt -- i.e. the model state
            # right BEFORE the round_(R-1) retraining that produced
            # `prior_train`. Stage04b evaluates that bundle on THIS
            # round's GEMMs to get a same-dataset, model-only delta
            # for the "did augmentation help?" SPLIT trigger. Only
            # makes sense from round R>=2 (we need two prior rounds
            # so a "model before this retrain" actually exists).
            if len(prior_round_dirs) >= 2:
                pre_models = prior_round_dirs[-2] / "stage05" / "models.pt"
                if pre_models.exists():
                    cmd += ["--prior-models", str(pre_models)]
            rc = _run(cmd, round_dir / "stage04b.log")
            _TIMING.end_rc(rc)
            if rc != 0:
                return _ret(1)
        else:
            ui.warn("stage", "Stage 4.5 skipped (--from-stage > 4.5)")

        retrain_list = ""
        retrain_path = stage04b_dir / "retrain_cells.txt"
        if retrain_path.exists():
            retrain_list = retrain_path.read_text().strip()

        if not skip_train and retrain_list:
            n_cells = len(retrain_list.split(","))
            _TIMING.banner(
                f"Stage 5 -- retrain {n_cells} failing cells "
                f"(cross-round augmentation; sub-cell aware)"
            )
            cmd = [
                sys.executable,
                "-u",
                str(STAGES / "stage05_train.py"),
                # Same reason as stage04b: without these, stage05 falls back to
                # its --origami-library-dir default, the containing checkout's
                # gfx950 library -- the wrong arch for any other run. If that
                # is not built, compute_origami_sel_eff() and
                # evaluate_deployed_sel_eff() see no configs and return
                # sel_eff=NaN, n_eval=0, so the ORIGAMI comparison column is
                # blank for every cell; if it is, they rank the wrong arch.
                "--origami-library-dir",
                str(dat_dir),
                "--origami-python-path",
                origami_pp,
                *_train_enriched_args(),
                *_prior_round_dir_args(),
                # --arch matters here: get_mi_latency() keys the MI-latency
                # feature table by arch. Omitting it leaves this stage on the
                # gfx950 default while stage07/08 use the real arch, which
                # silently skews training features against validation ones.
                "--arch",
                arch,
                # CRITICAL: tell stage05 about THIS round's
                # stage04b/splits.json so the cumulative split tree
                # stage05 builds includes the splits stage04b just
                # decided. Without this, round_R/stage05 sees the
                # new child labels as empty (the GEMMs in
                # round_R/stage04 still get assigned to the parent
                # label) and `to_train` collapses to 0. The
                # children of brand-new splits are the cells the
                # retrain list is asking us to retrain, so missing
                # this routing breaks recursive splitting outright.
                "--current-round-dir",
                str(round_dir),
                "--output-dir",
                str(stage05_dir),
                "--epochs",
                str(train_epochs),
                "--device",
                train_device,
                "--eval-sample-frac",
                str(train_eval_frac),
                "--smart-k",
                str(train_smart_k),
                *_smart_k_adaptive_args(),
                *_train_parallel_args(),
                *scale_mode_args,
                "--min-cell-gemms",
                str(train_min_cell_gemms),
                "--only-cells",
                retrain_list,
            ]
            rc = _run(cmd, round_dir / "stage05.log")
            _TIMING.end_rc(rc)
            if rc != 0:
                return _ret(1)
        elif not retrain_list:
            ui.ok("stage", "no cells to retrain (all PASS)")
        else:
            ui.warn("stage", "Stage 5 retrain skipped (--from-stage > 5)")

    # Read THIS round's HELD-OUT deployed sel_eff to drive the deploy
    # gate. The held-out signal lives in stage04b/decisions.json (per-
    # cell `sel_eff_new`, measured via the deployed pipeline: Origami
    # top-K + ML re-rank, on shapes the model has never seen). We
    # weight by `n_eval` so cells with more held-out samples carry
    # more weight in the geomean.
    #
    # In round 0 (initial mode) there is NO stage04b -- there's no
    # prior model to validate and no held-out data. So we cannot
    # compute a deploy-gate signal in round 0; `held_out_sel_eff`
    # stays NaN and the auto-triggers below stay disabled. (--deploy
    # on the CLI still works because that's an explicit operator
    # override; see `force_deploy` below.) The stage05/metrics.json
    # `global_sel_eff` is TRAINING-SET accuracy and is NOT used here
    # -- shipping a model judged by training fit is exactly the
    # overfit-risk the active loop was built to avoid.
    held_out_sel_eff = float("nan")
    decisions_path = stage04b_dir / "decisions.json"
    if round_mode != "initial" and decisions_path.exists():
        try:
            with decisions_path.open() as f:
                d = json.load(f)
            log_sum = 0.0
            w_sum = 0.0
            for c in d.get("per_cell", []):
                se = c.get("sel_eff_new")
                n = c.get("n_eval") or 0
                if se is None or n <= 0 or se <= 0:
                    continue
                log_sum += math.log(float(se)) * float(n)
                w_sum += float(n)
            if w_sum > 0:
                held_out_sel_eff = math.exp(log_sum / w_sum)
        except Exception:
            pass

    target_reached = False
    if (
        deploy_target_sel_eff is not None
        and not math.isnan(held_out_sel_eff)
        and held_out_sel_eff >= float(deploy_target_sel_eff)
    ):
        target_reached = True
        ui.ok(
            "deploy",
            f"target reached: held_out_sel_eff="
            f"{held_out_sel_eff*100:.2f}% "
            f">= target {float(deploy_target_sel_eff)*100:.2f}%",
        )
    elif not math.isnan(held_out_sel_eff):
        ui.info(
            "deploy",
            f"this round held_out_sel_eff="
            f"{held_out_sel_eff*100:.2f}%  "
            f"target={deploy_target_sel_eff}",
        )
    elif round_mode == "initial":
        ui.info(
            "deploy",
            "no deploy gate in round 0 (no held-out "
            "validation -- stage04b runs from round 1 onwards)",
        )

    # stage06 (deploy) ──────────────────────────────────────────────────────
    # Auto-trigger if ANY of:
    #   * --deploy passed on the CLI (explicit operator override; works
    #     in every round, including round 0 -- training-fit deploy is
    #     the operator's call)
    #   * deploy.target_sel_eff was reached on THIS round's held-out
    #     stage04b measurement (active rounds only -- round 0 has no
    #     held-out signal)
    #   * this is the LAST round of full mode AND deploy.on_final_round
    #     (active rounds only -- a 1-round full run is degenerate and
    #     should pass --deploy explicitly)
    if round_mode == "initial":
        # Round 0: ONLY explicit --deploy triggers deploy. The
        # target_reached / on_final_round gates require a held-out
        # signal that simply doesn't exist yet (stage04b is skipped
        # in round 0 by design).
        should_deploy = force_deploy
    else:
        should_deploy = (
            force_deploy or target_reached or (is_final_round and deploy_on_final)
        )
    if skip_deploy:
        ui.warn("stage", "Stage 6 (deploy) skipped (--from-stage > 6)")
        should_deploy = False
    deployed = False
    if should_deploy:
        if not (stage05_dir / "models.pt").exists():
            ui.warn("deploy", "skipped: stage05 produced no models.pt")
        else:
            reason = (
                "forced"
                if force_deploy
                else "target_reached" if target_reached else "final_round"
            )
            _TIMING.banner(f"Stage 6 -- deploy (reason={reason})")
            cmd = [
                sys.executable,
                "-u",
                str(STAGES / "stage06_deploy_to_origami.py"),
                "--train-dir",
                str(stage05_dir),
                "--config-yaml",
                str(cfg_path),
                "--out-dir",
                str(stage06_dir),
                "--weight-dtype",
                deploy_weight_dtype,
            ]
            for rd in prior_round_dirs:
                cmd += ["--prior-round-dir", str(rd)]
            if deploy_rocm_libraries_to_deploy:
                cmd += [
                    "--rocm-libraries-to-deploy",
                    str(deploy_rocm_libraries_to_deploy),
                ]
                if deploy_apply_patch:
                    cmd += ["--apply-patch"]
                    if deploy_build_after_apply:
                        cmd += ["--build-after-apply"]
                        if deploy_build_jobs is not None:
                            cmd += ["--build-jobs", str(deploy_build_jobs)]
            rc = _run(cmd, round_dir / "stage06.log")
            _TIMING.end_rc(rc)
            if rc != 0:
                return _ret(
                    1, held_out_sel_eff=held_out_sel_eff, target_reached=target_reached
                )
            deployed = True
    return _ret(
        0,
        held_out_sel_eff=held_out_sel_eff,
        deployed=deployed,
        target_reached=target_reached,
    )


# ── validate mode (post-deploy quality gates; no training) ───────────────────


def _sample_trained_shapes(
    run_dir: Path, n: int, config: Dict[str, Any], out_path: Path, seed: int = 42
) -> Tuple[int, int]:
    """Randomly sample up to `n` UNIQUE (M,N,K,batch) shapes from the TRAINED
    dataset -- the union of every round_*/stage01/shapes.yaml -- and write a
    hipblaslt-bench yaml at `out_path`. Used to drive stage 7's timing + the
    C++/Python pick-parity check on IN-DISTRIBUTION shapes (the cells the
    deployed model actually serves; guaranteed matching dtype/layout) instead
    of an external workload yaml. Returns (n_sampled, n_unique_available)."""
    import random as _random
    from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml
    from lib.shapes import Shape

    seen: set = set()
    shapes: List[Tuple[int, int, int, int]] = []
    for yml in sorted(run_dir.glob("round_*/stage01/shapes.yaml")):
        try:
            parsed = parse_shapes_yaml(yml)
        except Exception:
            continue
        for s in parsed:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                shapes.append(t)
    _random.Random(seed).shuffle(shapes)
    pick = shapes[: max(0, int(n))]
    write_bench_yaml(
        [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in pick], config, out_path
    )
    return len(pick), len(shapes)


def _run_validate_mode(
    *,
    cfg: Dict[str, Any],
    cfg_path: Path,
    run_dir: Path,
    rounds: List[Path],
    devices: List[int],
    arch: str,
    bench_blocks_per_gpu: int,
    sel_eff_threshold: str,
    validate_min_cell_gemms: int,
    train_device: str,
    deploy_rocm_libraries_to_deploy: Optional[str],
    runtime_rocm_libraries_root: Optional[str],
    from_round: int = 0,
    from_stage: float = 7.0,
) -> int:
    """Run stages 7 and 8 against an existing --run-dir.

    Every config knob this mode needs is INHERITED from sections you
    already wrote for training:
      * devices                       runtime.devices
      * blocks_per_gpu                bench.blocks_per_gpu
      * sel_eff_threshold             validate.sel_eff_threshold
      * min_cell_gemms                validate.min_cell_gemms
      * torch_device                  train.device
      * arch                          arch
      * deploy.rocm_libraries_to_deploy   stage 7's hipblaslt-bench
                                          source AND stage 8's enrich
                                          paths (falls back to
                                          runtime.rocm_libraries_root
                                          when the deploy build
                                          doesn't exist).

    The ONLY new fields the config carries are the inputs:
      * stage07.bench_yamls    [yaml, ...]   stage 7 inputs
      * stage08.datasets       [{name, path}, ...]   stage 8 inputs

    To make the ML recommender's runtime activation auditable, stage 7
    runs its ML-on bench with `TENSILE_USE_TILEWRIGHT=1 TILEWRIGHT_DIAG=1
    TILEWRIGHT_PICK_LOG=1` (and the ML-off baseline with none of them).
    The runtime finds the weights only through a `tilewright_index` beside
    the Tensile library .dat. With TILEWRIGHT_DIAG=1 it emits one stderr
    line `[TILEWRIGHT_DIAG FILE] arch=... n_cells=N ...` when it loads a
    .bin -- stage 7's summary flags whether the marker was seen so the
    operator can immediately tell whether bench timings reflect the
    deployed model or plain Origami.

    Stage 7 needs a hipblaslt-bench LINKED against an ml_recommender-
    aware origami (i.e. the deploy checkout must have been rebuilt
    with stage06_deploy's `build_after_apply: true`). If that
    binary is missing this function logs a clear warning and skips
    stage 7 -- stage 8 doesn't need it (its bench/enrich paths use
    pure Origami; the ML re-rank is done in Python).
    """
    bench_yamls = list(_opt(cfg, ["stage07", "bench_yamls"], []) or [])
    datasets = list(_opt(cfg, ["stage08", "datasets"], []) or [])
    _psn0 = int(_opt(cfg, ["stage07", "parity_sample_trained"], 0) or 0)
    if not bench_yamls and not datasets and _psn0 <= 0:
        ui.err(
            "err",
            "--mode validate requires at least one of "
            "`stage07.parity_sample_trained` (>0), "
            "`stage07.bench_yamls`, or `stage08.datasets` in the "
            "config; all are empty.",
        )
        return 2

    # Pick the round whose `stage05/` provides the trained models.
    # Default: most recent. `--from-round R > 0` selects round_R.
    target_round = rounds[-1]
    if from_round > 0:
        wanted = run_dir / f"round_{from_round}"
        if wanted in rounds:
            target_round = wanted
        else:
            ui.warn(
                "validate",
                f"--from-round {from_round} not found under "
                f"{run_dir}; falling back to latest "
                f"{rounds[-1].name}",
            )
    target_stage05 = target_round / "stage05"
    if not (target_stage05 / "models.pt").exists():
        ui.err(
            "err",
            f"{target_stage05} has no models.pt -- nothing to "
            f"validate. Run training first (or pick a different "
            f"--from-round).",
        )
        return 2

    # --from-stage gating: < 8 means "run stage 7 if configured";
    # >= 8 means "skip stage 7 (timing) and only run stage 8".
    skip_stage07 = from_stage > 7.0
    skip_stage08 = from_stage > 8.0
    # Prior rounds (strictly < target_round) feed the parity check's
    # split tree so leaf-cell routing matches the deployed runtime.
    prior_rounds = [
        r
        for r in rounds
        if r != target_round
        and int(r.name[len("round_") :]) < int(target_round.name[len("round_") :])
    ]

    validate_dir = run_dir / "validate"
    validate_dir.mkdir(parents=True, exist_ok=True)

    # stage07 input: when `stage07.parity_sample_trained` > 0, sample that
    # many random UNIQUE shapes from the TRAINED dataset (union of every
    # round_*/stage01/shapes.yaml) and use them for the timing + C++/Python
    # pick-parity check, instead of the external stage07.bench_yamls. The
    # parity check is most meaningful on shapes the deployed cells actually
    # serve, and trained shapes are guaranteed to match the run's dtype/layout
    # (so the workload filter never drops them as OOD).
    _psn = _opt(cfg, ["stage07", "parity_sample_trained"], 0)
    if _psn and int(_psn) > 0:
        _samp_yaml = validate_dir / f"stage07_trained_{int(_psn)}.yaml"
        try:
            _got, _tot = _sample_trained_shapes(run_dir, int(_psn), cfg, _samp_yaml)
            if _got > 0:
                bench_yamls = [_samp_yaml]
                ui.ok(
                    "cfg",
                    f"stage 7 bench: {_got} random trained shapes "
                    f"(of {_tot} unique, all rounds) -> {_samp_yaml.name}",
                )
            else:
                ui.warn(
                    "cfg",
                    "parity_sample_trained set but no trained "
                    "shapes found; using stage07.bench_yamls",
                )
        except Exception as e:
            ui.warn(
                "cfg",
                f"trained-shape sampling failed ({e!r}); using " f"stage07.bench_yamls",
            )

    ml_weights_path: Optional[Path] = None
    if deploy_rocm_libraries_to_deploy:
        # weights now live in the framework-agnostic tilewright lib. Productized
        # layout is `backends/hipblaslt/weights/`; fall back to the legacy
        # `data/` dir (older checkouts) then the previous origami paths.
        _root = Path(deploy_rocm_libraries_to_deploy)
        _data = _root / "shared/tilewright/backends/hipblaslt/weights"
        _legacy = _root / "shared/tilewright/backends/hipblaslt/data"
        if (
            not (_data / "tilewright_index").exists()
            and not (_data / "tilewright_weights.bin").exists()
            and (
                (_legacy / "tilewright_index").exists()
                or (_legacy / "tilewright_weights.bin").exists()
            )
        ):
            _data = _legacy
        ml_weights_path = _data / "tilewright_weights.bin"
        if not ml_weights_path.exists():
            # Per-library scheme (no global bin): a `tilewright_index` maps
            # library-logic stems -> per-library `*.tilewright.bin` in the same
            # dir. Point ml_weights_path at any per-library .bin there; stage
            # 7/8 cd into that dir so the deployed runtime discovers
            # `tilewright_index` via the cwd root and loads per-library models.
            # stage06 writes the PRODUCTIZED layout two levels deeper --
            # weights/<arch>/<arch>/ -- mirroring the Tensile library tree so
            # the CMake `tilewright_colocate` target can copy it next to the
            # per-arch logic (see `weights_subdir` in stage06). Checking
            # only the flat root reports "no tilewright weights found" right
            # after a deploy that wrote them, and leaves stages 7/8 without
            # --ml-weights-path.
            # The arch dir wins over any other; `weights/gfx1250/gfx1250` and
            # `weights/gfx1250v0/gfx1250v0` both exist and sort the wrong way.
            _idx = _data / "tilewright_index"
            if not _idx.exists():
                _arch_base = str(arch).split(":")[0]
                _nested = sorted(
                    _data.glob("*/*/tilewright_index"),
                    key=lambda p: (p.parent.name != _arch_base, str(p)),
                )
                if _nested:
                    _idx = _nested[0]
                    _data = _idx.parent
            if _idx.exists():
                _bins = sorted(_data.glob("*.tilewright.bin"))
                if _bins:
                    ml_weights_path = _bins[0]
        if not ml_weights_path.exists():
            for _rel in (
                "shared/tilewright/data/hipblaslt/ml_recommender_weights.bin",
                "shared/origami/data/ml_recommender_weights.bin",
            ):
                _fallback = _root / _rel
                if _fallback.exists():
                    ml_weights_path = _fallback
                    break
        if not ml_weights_path.exists():
            ui.warn(
                "validate",
                f"no tilewright weights found in {_data} (neither a global "
                f"tilewright_weights.bin nor a per-library tilewright_index + "
                f"*.tilewright.bin); stage 7 / 8 cannot point the deployed "
                f"runtime at the model and will fall back to plain "
                f"Origami. (Run --mode initial --deploy first to stage "
                f"the weights.)",
            )
            ml_weights_path = None

    ui.banner(f"pipeline_v3 validate  run_dir={run_dir.name}", ui.C.HEAD)
    ui.info(
        "cfg",
        f"target round         : {target_round.name}  "
        f"(stage05 source for stage 7+8)",
    )
    ui.info(
        "cfg",
        f"prior rounds         : " f"{[r.name for r in prior_rounds] or '(none)'}",
    )
    ui.info(
        "cfg",
        f"stage filter         : "
        f"stage7={'OFF' if skip_stage07 else 'ON '}  "
        f"stage8={'OFF' if skip_stage08 else 'ON '}  "
        f"(--from-stage={from_stage})",
    )
    ui.info(
        "cfg", f"bench_yamls          : " f"{[str(p) for p in bench_yamls] or '(none)'}"
    )
    ui.info(
        "cfg",
        f"datasets             : " f"{[d.get('name') for d in datasets] or '(none)'}",
    )
    ui.info(
        "cfg",
        f"ml_weights_path      : "
        f"{ml_weights_path or '(no .bin found; ML path unverified)'}",
    )
    ui.info("cfg", f"sel_eff_threshold    : {sel_eff_threshold}")
    ui.info("cfg", f"min_cell_gemms       : {validate_min_cell_gemms}")
    ui.info("cfg", f"torch_device         : {train_device}")
    ui.info("cfg", f"arch                 : {arch}")

    # ── stage 7 ────────────────────────────────────────────────────────────
    stage07_rc: Optional[int] = None
    if skip_stage07:
        ui.warn("stage7", f"stage 7 skipped (--from-stage={from_stage} > 7)")
    elif bench_yamls:
        if not deploy_rocm_libraries_to_deploy:
            ui.warn(
                "stage7",
                "skipping stage 7: deploy.rocm_libraries_to_deploy "
                "is null -- no hipblaslt-bench available for "
                "selection-time measurement.",
            )
            stage07_rc = None
        else:
            deploy_bench = (
                Path(deploy_rocm_libraries_to_deploy)
                / "projects/hipblaslt/build/release/clients"
                "/hipblaslt-bench"
            )
            if not deploy_bench.exists():
                ui.warn(
                    "stage7",
                    f"skipping stage 7: hipblaslt-bench not found "
                    f"at {deploy_bench}. The deploy checkout needs "
                    f"to be built (cmake + make in shared/origami "
                    f"and projects/hipblaslt) before selection-"
                    f"time validation can run. Stage 8 will still "
                    f"run since it uses runtime.rocm_libraries_"
                    f"root for its bench/enrich paths.",
                )
                stage07_rc = None
            else:
                stage07_out = validate_dir / "stage07"
                _TIMING.banner(
                    "Stage 7 -- validate selection time + " "C++ <-> Python pick parity"
                )
                cmd = [
                    sys.executable,
                    "-u",
                    str(STAGES / "stage07_optional_validate_selection_time.py"),
                    "--rocm-libraries-to-deploy",
                    str(deploy_rocm_libraries_to_deploy),
                    "--out-dir",
                    str(stage07_out),
                    "--devices",
                    str(devices[0]),
                    # Pick parity: read the SAME models.pt the
                    # deploy .bin was packed from, in Python,
                    # and compare picks per problem against the
                    # `[TILEWRIGHT_PICK]` lines hipblaslt-bench writes.
                    "--train-dir",
                    str(target_stage05),
                    "--arch",
                    arch,
                    "--torch-device",
                    train_device,
                    # Keep the Python parity check in lockstep
                    # with the C++ deploy runtime. Deploy ALWAYS
                    # applies smart_K (default 10) after the
                    # LDS + is_kernel_feasible pre-filter, in
                    # input order. The parity helper does the
                    # same when given the same smart_k. We pull
                    # `train.smart_k` from the config so all
                    # three (training, deploy, parity) agree.
                    "--parity-smart-k",
                    str(int(_opt(cfg, ["train", "smart_k"], 10))),
                    # Round the Python student weights to the SAME
                    # precision the deployed .bin stores (stage06
                    # deploy.weight_dtype). With bf16 deploys this
                    # keeps C++<->Python pick parity exact.
                    "--parity-weight-dtype",
                    str(_opt(cfg, ["deploy", "weight_dtype"], "bf16")).lower(),
                ]
                # All prior rounds feed the parity check's split tree
                # (so the Python pick uses the same leaf-cell routing
                # the deploy runtime does).
                for rd in prior_rounds:
                    cmd += ["--prior-round-dir", str(rd)]
                for yp in bench_yamls:
                    cmd += ["--hipblaslt-bench-yaml", str(yp)]
                # Optional explicit UserArgs.yaml for the rank_configs
                # call-time microbench. stage07 auto-discovery only
                # scores BBS/HHS/F8 name tokens, so TF32/xf32 configs
                # (a_type=f32_r) MUST name their Origami UserArgs here
                # or the microbench picks the wrong candidate set.
                rt_ua = _opt(cfg, ["stage07", "rank_time_userargs_yaml"], None)
                if rt_ua:
                    cmd += ["--rank-time-userargs-yaml", str(rt_ua)]
                if ml_weights_path is not None:
                    cmd += ["--ml-weights-path", str(ml_weights_path)]

                # Origami python binding for the parity check. The
                # path must contain a COMPILED `.so` extension
                # (`origami/origami.cpython-*.so`), not just the
                # source dir -- otherwise `import origami` fails.
                # The deploy checkout's `shared/origami/python/src`
                # is the preferred source because that's the package
                # built against ml_recommender.cpp; fall back to the
                # runtime root (the training checkout's binding) when
                # the deploy origami hasn't been built yet. Note:
                # parity computation does NOT need the ml_recommender
                # symbol at the Python level -- our parity helper
                # runs the student in pure torch -- so the
                # runtime-root fallback is functionally fine.
                def _origami_pp_or_none(repo: Path) -> Optional[Path]:
                    pp = repo / "shared/origami/python/src"
                    if list(pp.glob("origami/origami.cpython-*.so")):
                        return pp
                    return None

                origami_pp_for_parity: Optional[Path] = _origami_pp_or_none(
                    Path(deploy_rocm_libraries_to_deploy)
                )
                if origami_pp_for_parity is None and runtime_rocm_libraries_root:
                    origami_pp_for_parity = _origami_pp_or_none(
                        Path(runtime_rocm_libraries_root)
                    )
                if origami_pp_for_parity is not None:
                    cmd += ["--origami-python-path", str(origami_pp_for_parity)]
                else:
                    ui.warn(
                        "stage7",
                        "no built origami python binding "
                        "(`.so`) found under either "
                        "deploy.rocm_libraries_to_deploy or "
                        "runtime.rocm_libraries_root; the "
                        "parity check will skip itself.",
                    )
                # Workload filter from the config's `hipblaslt:` block.
                # Without this filter the parity check runs against
                # OUT-OF-DISTRIBUTION yaml entries (e.g. BBS_NT
                # problems scored by a BBS_TN model), and the
                # `is_kernel_feasible` per-problem NTA/NTB branch
                # flips. The trained smart_K list is in-distribution,
                # so all its sigs get rejected for the OOD problems
                # and the runtime falls back -- producing fallback
                # rates that don't reflect any real training/deploy
                # parity bug. With the filter, parity measures what
                # the model was actually trained to serve.
                hipblaslt_cfg = _opt(cfg, ["hipblaslt"], {}) or {}
                for cli_flag, cfg_key in (
                    ("--filter-a-type", "a_type"),
                    ("--filter-b-type", "b_type"),
                    ("--filter-c-type", "c_type"),
                    ("--filter-d-type", "d_type"),
                    ("--filter-compute-type", "compute_type"),
                    ("--filter-transA", "transA"),
                    ("--filter-transB", "transB"),
                ):
                    v = hipblaslt_cfg.get(cfg_key)
                    if v is not None and str(v) != "":
                        cmd += [cli_flag, str(v)]
                # Scale mode: picks WHICH .dat libraries the Python side
                # of the parity check ranks. On gfx1250 the MX and non-MX
                # F8 TN libraries share a dtype/layout key, so without
                # this the Python pool is a superset of what the C++
                # runtime can pick and parity mismatches are guaranteed.
                # Absent `hipblaslt.scaleA` -> flag omitted -> old behavior.
                _sm = hipblaslt_cfg.get("scaleA")
                if _sm is not None:
                    cmd += ["--scale-mode", str(int(_sm))]
                stage07_rc = _run(cmd, validate_dir / "stage07.log")
                _TIMING.end_rc(stage07_rc)
                if stage07_rc != 0:
                    ui.err(
                        "stage7",
                        f"stage 7 FAILED with rc={stage07_rc}; "
                        f"continuing to stage 8 anyway",
                    )

    # ── stage 8 ────────────────────────────────────────────────────────────
    # Stage 8's hipblaslt-bench + enrich paths fall back to
    # runtime.rocm_libraries_root when the deploy checkout's build is
    # incomplete (no built bench / no installed origami Python
    # binding). The deploy .bin is still consumed -- via
    # TILEWRIGHT_WEIGHTS -- so the model produced by the
    # deployed-pipeline path is the right one.
    stage08_rc: Optional[int] = None
    if skip_stage08:
        ui.warn("stage8", f"stage 8 skipped (--from-stage={from_stage} > 8)")
    elif datasets:
        stage08_rocm = (
            deploy_rocm_libraries_to_deploy
            if deploy_rocm_libraries_to_deploy
            and (
                Path(deploy_rocm_libraries_to_deploy)
                / "projects/hipblaslt/build/release/clients"
                "/hipblaslt-bench"
            ).exists()
            else runtime_rocm_libraries_root
        )
        if not stage08_rocm:
            ui.err(
                "stage8",
                "stage 8 needs a rocm-libraries checkout with a "
                "built hipblaslt-bench (and origami python). "
                "Neither deploy.rocm_libraries_to_deploy nor "
                "runtime.rocm_libraries_root has one.",
            )
            return 2
        stage08_out = validate_dir / "stage08"
        _TIMING.banner("Stage 8 -- validate selection efficiency")
        cmd = [
            sys.executable,
            "-u",
            str(STAGES / "stage08_optional_validate_selection_efficiency.py"),
            "--train-dir",
            str(target_stage05),
            "--rocm-libraries-to-deploy",
            str(stage08_rocm),
            "--out-dir",
            str(stage08_out),
            "--devices",
            ",".join(str(d) for d in devices),
            "--blocks-per-gpu",
            str(bench_blocks_per_gpu),
            "--sel-eff-threshold",
            sel_eff_threshold,
            "--min-cell-gemms",
            str(validate_min_cell_gemms),
            # Stage 8 evaluates against the SAME candidate set
            # the deploy runtime exercises: LDS + is_kernel_feasible
            # filter, then smart_K=train.smart_k cap. Locking
            # all three (training, deploy, validate) to the same
            # K keeps stage 8's sel_eff numbers honest.
            "--smart-k",
            str(int(_opt(cfg, ["train", "smart_k"], 10))),
            "--arch",
            arch,
            "--device",
            train_device,
            # When a stage08 dataset is a raw bench yaml, stage 8 runs a
            # stage02 probe to calibrate per-shape iters / cold_iters /
            # skip_slow_solution_ratio from this config's `probe:`
            # section (instead of a fixed iters) before the full
            # candidate sweep -- same as the training pipeline.
            "--config-yaml",
            str(cfg_path),
        ]
        # Stage 8's per-cell evaluator routes GEMMs via the same
        # cumulative split tree the deploy runtime sees (so a
        # sub-cell GEMM is scored by the sub-cell model, not its
        # parent). All prior rounds contribute their `cells.json`.
        for rd in prior_rounds:
            cmd += ["--prior-round-dir", str(rd)]

        # Origami python extension: prefer the deploy checkout (so
        # we score against the same analytical Origami that bench's
        # ML pipeline calls); fall back to the runtime/training
        # checkout. Both have the same numerics here -- we just
        # need a built `.so`.
        def _origami_pp_or_none(repo: Path) -> Optional[Path]:
            pp = repo / "shared/origami/python/src"
            if list(pp.glob("origami/origami.cpython-*.so")):
                return pp
            return None

        origami_pp_for_eval: Optional[Path] = _origami_pp_or_none(Path(stage08_rocm))
        if origami_pp_for_eval is None and runtime_rocm_libraries_root:
            origami_pp_for_eval = _origami_pp_or_none(Path(runtime_rocm_libraries_root))
        if origami_pp_for_eval is not None:
            cmd += ["--origami-python-path", str(origami_pp_for_eval)]
        for ds in datasets:
            name = ds.get("name") if isinstance(ds, dict) else None
            path = ds.get("path") if isinstance(ds, dict) else None
            if not name or not path:
                ui.err("stage8", f"stage08.datasets entry missing name/path: {ds!r}")
                return 2
            cmd += ["--dataset", f"{name}={path}"]
            # Per-dataset rocm-libraries used to GENERATE this bench.
            # The bench's `--Solution index` numbers and `--Solution
            # name` strings only resolve in the .dat library THAT
            # checkout shipped; pointing stage 4 enrichment at the
            # deploy library instead silently drops most rows as
            # bad_kp. Find the SHA on `hipBLASLt git version` at the
            # top of any bench log in this dataset; the config field
            # `stage08.datasets[i].rocm_libraries_used` carries the
            # path to the corresponding rocm-libraries checkout.
            bench_lib = ds.get("rocm_libraries_used") if isinstance(ds, dict) else None
            if bench_lib:
                cmd += ["--dataset-rocm-libraries", f"{name}={bench_lib}"]
        if ml_weights_path is not None:
            cmd += ["--ml-weights-path", str(ml_weights_path)]
        # Workload filter from the config's `hipblaslt:` block. A
        # production / OOB trace can be multi-dtype; the trained
        # model is single-workload. Pass the config's a/b/c/d
        # dtype + transA/transB so stage 8 only scores GEMMs whose
        # bench-row fields match. Anything not set in the config
        # is omitted -- stage 8 then doesn't filter on that field.
        hipblaslt_cfg = _opt(cfg, ["hipblaslt"], {}) or {}
        for cli_flag, cfg_key in (
            ("--filter-a-type", "a_type"),
            ("--filter-b-type", "b_type"),
            ("--filter-c-type", "c_type"),
            ("--filter-d-type", "d_type"),
            ("--filter-compute-type", "compute_type"),
            ("--filter-transA", "transA"),
            ("--filter-transB", "transB"),
        ):
            v = hipblaslt_cfg.get(cfg_key)
            if v is not None and str(v) != "":
                cmd += [cli_flag, str(v)]
        # Same scale-mode narrowing as stage 7 (and passed on to the
        # stage04 enrichment stage 8 spawns per dataset). Omitted when
        # the config has no `hipblaslt.scaleA`.
        _sm = hipblaslt_cfg.get("scaleA")
        if _sm is not None:
            cmd += ["--scale-mode", str(int(_sm))]
        stage08_rc = _run(cmd, validate_dir / "stage08.log")
        _TIMING.end_rc(stage08_rc)
        if stage08_rc != 0:
            ui.err("stage8", f"stage 8 FAILED with rc={stage08_rc}")

    ui.ok("validate", f"run_dir   : {run_dir}")
    ui.ok("validate", f"validate/ : {validate_dir}")
    if stage07_rc is not None:
        (ui.ok if stage07_rc == 0 else ui.err)(
            "validate", f"stage 7   : rc={stage07_rc}"
        )
    else:
        ui.warn("validate", "stage 7   : skipped")
    if stage08_rc is not None:
        (ui.ok if stage08_rc == 0 else ui.err)(
            "validate", f"stage 8   : rc={stage08_rc}"
        )
    else:
        ui.warn("validate", "stage 8   : skipped")
    # The combined "Done" banner + per-stage timing table + cross-
    # round held_out_sel_eff aggregate gets printed by main()'s
    # common tail, so all modes share one summary block. Just return
    # the rollup rc here.
    return max(0, stage07_rc or 0, stage08_rc or 0)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="pipeline_v3 end-to-end orchestrator (config-driven).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="OOB configuration yaml -- drives EVERYTHING "
        "(devices, build paths, per-stage knobs, "
        "training hyperparams, full-mode round count). "
        "Identified by `config_id` inside the file; "
        "NOT the same thing as a 96-grid cell.",
    )
    ap.add_argument(
        "--mode",
        choices=("initial", "active", "full", "validate"),
        required=True,
        help="initial = round_0 only. "
        "active = ONE additional round on top of an "
        "existing --run-dir (auto-discovers every "
        "round_*/ already there and uses them as "
        "priors). "
        "full = run round_0 + every active round in "
        "one shot, up to `full.total_rounds` from "
        "config. "
        "validate = POST-DEPLOY validation against "
        "an existing --run-dir: runs stage 7 "
        "(selection time, ML-on vs ML-off, on every "
        "`stage07.bench_yamls` entry) and stage 8 "
        "(selection efficiency on every "
        "`stage08.datasets` entry, against the "
        "LATEST round's trained models). No new "
        "training. All knobs are derived from "
        "runtime/bench/validate/train -- the "
        "config's `stage07` / `stage08` sections "
        "carry ONLY the input lists.",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="explicit RUN dir (default: "
        "runs/<config_stem>_<YYYYMMDD_HHMMSS>). The run "
        "dir is the LEVEL ABOVE the round_<i>/ subdirs. "
        "Pass it to RESUME an existing run (auto-"
        "discovers every round_*/ already under it and "
        "uses them as priors). To BRANCH one run off "
        "another, `cp -r runs/old runs/new` first, "
        "then point `--run-dir` at `runs/new`.",
    )
    ap.add_argument(
        "--deploy",
        action="store_true",
        help="force stage06_deploy_to_origami to run after "
        "EVERY round. Without this flag stage06_deploy "
        "still fires automatically in ACTIVE rounds "
        "when (a) this round's stage04b/decisions.json "
        "HELD-OUT geomean sel_eff >= "
        "deploy.target_sel_eff, or (b) this is the "
        "last round of --mode full AND "
        "deploy.on_final_round is true. Round 0 has "
        "no held-out validation (stage04b is skipped) "
        "so the auto-triggers are DISABLED in round "
        "0 -- use --deploy to force a round-0 deploy.",
    )
    ap.add_argument(
        "--from-round",
        type=int,
        default=0,
        metavar="R",
        help="full mode: skip rounds 0..R-1 entirely; "
        "round R starts at --from-stage, all later "
        "rounds start at stage 1. "
        "validate mode: use round_R's stage05 as "
        "the trained-models source for stage 7+8 "
        "(default = the LATEST round on disk). "
        "Default 0.",
    )
    ap.add_argument(
        "--from-stage",
        type=float,
        default=1.0,
        metavar="N",
        choices=(1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0),
        help="skip stages whose LOGICAL number is < N. "
        "Training-path floats: 1, 2, 3, 4, 4.5, 5, 6. "
        "Validate-path floats: 7 (run stages 7+8) or "
        "8 (skip stage 7 -- bench timing -- and run "
        "ONLY stage 8 selection-efficiency). "
        "Default 1.",
    )
    args = ap.parse_args()

    # ── load config ────────────────────────────────────────────────────────
    if not args.config.exists():
        ui.err("cfg", f"config not found: {args.config}")
        return 2
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    _missing_env: set = set()
    cfg = _expand_config_env(cfg, _missing_env)
    if _missing_env:
        ui.err(
            "cfg",
            f"{args.config} references unset environment "
            f"variable(s): {', '.join(sorted(_missing_env))}. Export "
            f"them (scripts/run_gfx1250v0.sh sets the TILEWRIGHT_* ones "
            f"it can derive) and re-run.",
        )
        return 2

    runtime = _need(cfg, ["runtime"], str(args.config))
    devices = _need(runtime, ["devices"], "runtime")
    if not isinstance(devices, list) or not devices:
        ui.err("cfg", "runtime.devices must be a non-empty list")
        return 2
    devices = [int(d) for d in devices]
    arch = str(_need(cfg, ["arch"], str(args.config)))
    bench_binary, dat_dir, match_table, origami_pp = _derive_paths(cfg, args.config)

    # Cascaded knobs.
    target_per_cell = int(_opt(cfg, ["seed", "target_per_cell"], 1600))
    shapes_per_cell = int(_opt(cfg, ["seed", "shapes_per_cell"], 150))
    seed_seed = int(_opt(cfg, ["seed", "seed"], 42))
    n_strata = int(_opt(cfg, ["seed", "n_strata"], 1))
    train_epochs = int(_opt(cfg, ["train", "epochs"], 200))
    train_device = str(_opt(cfg, ["train", "device"], "cpu"))
    # `eval_sample_frac` controls how many GEMMs the per-epoch
    # sel_eff report covers (1.0 = all GEMMs, the configured
    # default). stage05 trains on ALL GEMMs regardless; this is just
    # a reporting subsample. Accept `val_frac` as a deprecated alias.
    _val_frac_cfg = _opt(cfg, ["train", "val_frac"], None)
    train_eval_frac = float(
        _val_frac_cfg
        if _val_frac_cfg is not None
        else _opt(cfg, ["train", "eval_sample_frac"], 1.0)
    )
    train_smart_k = int(_opt(cfg, ["train", "smart_k"], 10))
    train_min_cell_gemms = int(_opt(cfg, ["train", "min_cell_gemms"], 50))
    # Adaptive-K (opt-in). `train.smart_k_max` unset -> legacy fixed
    # --smart-k. When set, stage05 grows each cell's signature list to
    # k_max then truncates at the regret-curve knee in
    # [smart_k_min, smart_k_max]. Parity-safe (no C++ change).
    _smk_max_raw = _opt(cfg, ["train", "smart_k_max"], None)
    train_smart_k_max: Optional[int] = (
        int(_smk_max_raw) if _smk_max_raw not in (None, "", False) else None
    )
    train_smart_k_min = int(_opt(cfg, ["train", "smart_k_min"], 4))
    train_smart_k_saturation_eps = float(
        _opt(cfg, ["train", "smart_k_saturation_eps"], 0.005)
    )
    # PASS / SPLIT thresholds can be a numeric value (always) OR the
    # literal string "origami" (use the cell's measured Origami
    # sel_eff as the per-cell threshold). We pass them through as
    # strings; stage06 resolves per-cell at decision time.
    sel_eff_threshold = str(_opt(cfg, ["validate", "sel_eff_threshold"], "0.95"))
    split_floor = str(_opt(cfg, ["validate", "split_floor"], "0.85"))
    validate_min_cell_gemms = int(_opt(cfg, ["validate", "min_cell_gemms"], 20))
    split_min_improvement = float(
        _opt(cfg, ["validate", "split_min_improvement"], 0.02)
    )
    split_after_attempts = int(_opt(cfg, ["validate", "split_after_attempts"], 2))
    max_split_depth = int(_opt(cfg, ["validate", "max_split_depth"], 6))
    structural_split_min_octaves = float(
        _opt(cfg, ["validate", "structural_split_min_octaves"], 0.0)
    )
    structural_split_max_depth = int(
        _opt(cfg, ["validate", "structural_split_max_depth"], -1)
    )
    bench_blocks_per_gpu = int(_opt(cfg, ["bench", "blocks_per_gpu"], 20))
    total_rounds = int(_opt(cfg, ["full", "total_rounds"], 3))
    deploy_target_sel_eff: Optional[float] = _opt(
        cfg, ["deploy", "target_sel_eff"], 0.95
    )
    deploy_on_final = bool(_opt(cfg, ["deploy", "on_final_round"], True))
    deploy_stop_on_target = bool(
        _opt(cfg, ["deploy", "stop_when_target_reached"], False)
    )
    deploy_rocm_libraries_to_deploy: Optional[str] = _opt(
        cfg, ["deploy", "rocm_libraries_to_deploy"], None
    )
    deploy_apply_patch = bool(_opt(cfg, ["deploy", "apply_patch"], False))
    deploy_build_after_apply = bool(_opt(cfg, ["deploy", "build_after_apply"], False))
    _deploy_build_jobs_raw = _opt(cfg, ["deploy", "build_jobs"], None)
    deploy_build_jobs: Optional[int] = (
        int(_deploy_build_jobs_raw)
        if _deploy_build_jobs_raw not in (None, "", False)
        else None
    )
    # MLP weight precision for the deployed .bin. bf16 (default)
    # ~halves the binary; the C++ runtime promotes bf16->fp32 at load
    # and the stage 7 parity check rounds identically, so picks are
    # unchanged. fp32 = legacy. Threaded into stage06 (writer) AND
    # stage07 (parity) AND stage08 (deployed-quality eval) so all
    # three agree on the precision being measured.
    deploy_weight_dtype = str(_opt(cfg, ["deploy", "weight_dtype"], "bf16")).lower()

    # ── pick / make run dir ────────────────────────────────────────────────
    run_dir = (args.run_dir or _autoname_run_dir(args.config)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── orchestrator stdout/stderr -> <run-dir>/orchestrator.log ───────────
    # We open the log file in append mode so re-running with the same
    # --run-dir (resume / branch / re-validate) keeps the previous
    # invocation's history; each invocation is fenced by the banner
    # line below so grepping for `mode=` finds the per-invocation
    # boundaries. The Tee preserves the terminal stream verbatim
    # (colours + tty behaviour) while writing a plain-text copy to
    # the log.
    global _TIMING, _ORIG_STDOUT
    _TIMING = _StageTimingTracker()
    _orch_log_path = run_dir / "orchestrator.log"
    _orch_log_fh = _orch_log_path.open("a", buffering=1)
    _orch_log_fh.write(
        f"\n{'='*80}\n"
        f"# pipeline_v3 orchestrator invocation  "
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f"# config={args.config}  mode={args.mode}  "
        f"run_dir={run_dir}\n"
        f"# argv={sys.argv}\n"
        f"{'='*80}\n"
    )
    # Snapshot the REAL stdout BEFORE we replace `sys.stdout` with
    # the tee. `_run()` uses this snapshot to forward subprocess
    # output to the terminal WITHOUT echoing it through the tee
    # (which would otherwise dump every stage's stdout into
    # orchestrator.log). orchestrator.log ends up containing ONLY
    # `run_orchestrator.py`'s own prints.
    _ORIG_STDOUT = sys.stdout
    sys.stdout = _Tee(sys.stdout, _orch_log_fh)
    sys.stderr = _Tee(sys.stderr, _orch_log_fh)

    ui.banner(f"pipeline_v3  config={args.config.name}  mode={args.mode}", ui.C.HEAD)
    ui.info("cfg", f"orchestrator log : {_orch_log_path}")
    ui.info("cfg", f"config_id    : {cfg.get('config_id')}")
    ui.info("cfg", f"run_dir      : {run_dir}")
    ui.info("cfg", f"devices      : {devices}")
    ui.info("cfg", f"bench_binary : {bench_binary}")
    ui.info("cfg", f"dat_dir      : {dat_dir}")
    ui.info("cfg", f"match_table  : {match_table}")
    ui.info(
        "cfg",
        f"train_device : {train_device}  epochs={train_epochs}  "
        f"smart_k={train_smart_k}",
    )
    if args.mode == "full":
        ui.info("cfg", f"full.total_rounds : {total_rounds}")

    # Echo a copy of the config used.
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.dump(cfg, f, Dumper=_FlowListDumper, sort_keys=False)

    # ── plan rounds ────────────────────────────────────────────────────────
    # Priors are EVERY round_*/ already present under --run-dir. We
    # auto-discover them so a resumed active-mode invocation never
    # accidentally re-tests shapes from earlier rounds AND so the
    # internal stage CLIs (stage05/06/07 expect --prior-round-dir
    # one-per-prior) get the right list. To branch off another run,
    # `cp -r runs/old runs/new` BEFORE invoking the orchestrator.
    def _discover_rounds_under(rd: Path) -> List[Path]:
        # Sort by INTEGER round number, NOT lexicographic. The
        # default lex sort puts `round_10` BEFORE `round_2`, which
        # means `rounds[-1]` would return `round_9` instead of
        # `round_14` once we have >= 10 rounds -- the deploy and
        # parity checks would then load a stale models.pt and the
        # validate stage would compare C++ (latest .bin) against
        # Python (round_9 bundle), producing artificial parity
        # mismatches. Numeric sort fixes this regardless of how
        # many rounds the config asks for.
        if not rd.exists() or not rd.is_dir():
            return []
        out: List[Path] = []
        for child in rd.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("round_")
                and child.name[len("round_") :].isdigit()
                and (child / "stage01" / "shapes.yaml").exists()
            ):
                out.append(child)
        out.sort(key=lambda p: int(p.name[len("round_") :]))
        return out

    plan: List[Tuple[int, str, List[Path]]] = []
    if args.mode == "initial":
        plan.append((0, "initial", []))
    elif args.mode == "active":
        priors = _discover_rounds_under(run_dir)
        if not priors:
            ui.err(
                "err",
                "active mode requires --run-dir to point at "
                "an existing run that already contains round_*/ "
                "subdirs. (To branch off another run, "
                "`cp -r runs/old runs/new`, then "
                "`--run-dir runs/new`.)",
            )
            return 2
        ui.info(
            "cfg",
            f"auto-discovered priors under {run_dir}: " f"{[p.name for p in priors]}",
        )
        idx = len(priors)
        plan.append((idx, "active", priors))
    elif args.mode == "validate":
        # Validate mode is a separate top-level branch (no training,
        # no deploy, no rounds). It runs post-deploy quality gates
        # against the round picked by --from-round (default latest),
        # honouring --from-stage (7 = run both, 8 = skip stage 7),
        # and then falls through to the common tail so the cross-
        # round held-out aggregation + timing summary still print.
        rounds = _discover_rounds_under(run_dir)
        if not rounds:
            ui.err(
                "err",
                f"--mode validate requires --run-dir to contain "
                f"existing round_*/ subdirs; nothing found under "
                f"{run_dir}",
            )
            return 2
        # --from-stage default in validate is 7 (run both). The
        # global CLI default is 1 (training-path), which would also
        # imply "run both" -- we keep it.
        _vstage = args.from_stage if args.from_stage in (7.0, 8.0) else 7.0
        _run_validate_mode(
            cfg=cfg,
            cfg_path=args.config,
            run_dir=run_dir,
            rounds=rounds,
            devices=devices,
            arch=arch,
            bench_blocks_per_gpu=bench_blocks_per_gpu,
            sel_eff_threshold=sel_eff_threshold,
            validate_min_cell_gemms=validate_min_cell_gemms,
            train_device=train_device,
            deploy_rocm_libraries_to_deploy=deploy_rocm_libraries_to_deploy,
            runtime_rocm_libraries_root=str(
                _opt(cfg, ["runtime", "rocm_libraries_root"], "")
            )
            or None,
            from_round=int(args.from_round),
            from_stage=_vstage,
        )
        # No further rounds to run; the empty `plan` below is the
        # signal that the round loop should be a no-op.
        # Fall through to the common tail.
    else:  # full
        if total_rounds < 1:
            ui.err("cfg", f"full.total_rounds must be >= 1 " f"(got {total_rounds})")
            return 2
        plan.append((0, "initial", []))
        for r in range(1, total_rounds):
            priors = [run_dir / f"round_{i}" for i in range(r)]
            plan.append((r, "active", priors))

    # ── opt-in post-rounds weak-cell hardening (O4) ────────────────────────
    # Default OFF -> existing behavior unchanged. When enabled (full mode),
    # the planned rounds run WITHOUT deploying; after them we run extra
    # ACTIVE rounds focused on the weak tail until every cell's latest
    # held-out sel_eff >= threshold (or max_rounds), boosting smart_k_min so
    # weak cells get a thicker kernel shortlist. Deploy lands on the last
    # hardening round.
    harden_enabled = (
        bool(_opt(cfg, ["hardening", "enabled"], False)) and args.mode == "full"
    )
    harden_threshold = float(_opt(cfg, ["hardening", "sel_eff_threshold"], 0.80))
    harden_max_rounds = int(_opt(cfg, ["hardening", "max_rounds"], 4))
    harden_smart_k_min = int(_opt(cfg, ["hardening", "smart_k_min"], train_smart_k_min))
    if harden_enabled:
        ui.info(
            "cfg",
            f"hardening (O4)     : ENABLED  threshold="
            f"{harden_threshold:.2f}  max_rounds={harden_max_rounds}  "
            f"smart_k_min={harden_smart_k_min}",
        )

    # ── run rounds ─────────────────────────────────────────────────────────
    last_round_idx = plan[-1][0] if plan else -1
    final_result: Dict[str, Any] = {}
    for round_idx, round_mode, priors in plan:
        if round_idx < args.from_round:
            ui.warn(
                "round", f"round_{round_idx} skipped " f"(--from-round > {round_idx})"
            )
            continue
        # Resumed round starts at --from-stage; subsequent rounds at 1.
        round_from_stage = args.from_stage if round_idx == args.from_round else 1
        # When hardening is enabled, planned rounds never deploy -- the
        # final deploy is done by the last hardening round below.
        is_final = (round_idx == last_round_idx) and not harden_enabled
        result = _run_round(
            round_idx=round_idx,
            round_mode=round_mode,
            prior_round_dirs=priors,
            run_dir=run_dir,
            cfg=cfg,
            cfg_path=args.config,
            devices=devices,
            arch=arch,
            bench_binary=bench_binary,
            dat_dir=dat_dir,
            match_table=match_table,
            origami_pp=origami_pp,
            from_stage=round_from_stage,
            force_deploy=args.deploy,
            is_final_round=is_final,
            target_per_cell=target_per_cell,
            shapes_per_cell=shapes_per_cell,
            seed_seed=seed_seed,
            n_strata=n_strata,
            bench_blocks_per_gpu=bench_blocks_per_gpu,
            train_epochs=train_epochs,
            train_device=train_device,
            train_eval_frac=train_eval_frac,
            train_smart_k=train_smart_k,
            train_smart_k_max=train_smart_k_max,
            train_smart_k_min=train_smart_k_min,
            train_smart_k_saturation_eps=train_smart_k_saturation_eps,
            train_min_cell_gemms=train_min_cell_gemms,
            sel_eff_threshold=sel_eff_threshold,
            validate_min_cell_gemms=validate_min_cell_gemms,
            split_floor=split_floor,
            split_min_improvement=split_min_improvement,
            split_after_attempts=split_after_attempts,
            max_split_depth=max_split_depth,
            structural_split_min_octaves=structural_split_min_octaves,
            structural_split_max_depth=structural_split_max_depth,
            deploy_target_sel_eff=(
                None
                if deploy_target_sel_eff in (None, False)
                else float(deploy_target_sel_eff)
            ),
            deploy_on_final=deploy_on_final,
            deploy_rocm_libraries_to_deploy=deploy_rocm_libraries_to_deploy,
            deploy_apply_patch=deploy_apply_patch,
            deploy_build_after_apply=deploy_build_after_apply,
            deploy_weight_dtype=deploy_weight_dtype,
            deploy_build_jobs=deploy_build_jobs,
        )
        final_result = result
        if result["rc"] != 0:
            ui.err("round", f"round_{round_idx} FAILED with rc={result['rc']}")
            # The report is independent of run outcome: completed rounds are
            # still worth reading when a later round fails. Print the timing
            # ledger FIRST -- it only reaches the log via print_summary(), so
            # returning here without it leaves the report's timing table with
            # no [timing] block to parse for any round that did complete.
            _TIMING.print_summary()
            _emit_report(run_dir)
            return result["rc"]
        # Optional early-stop: if the model hit the deploy target and
        # the config sets stop_when_target_reached, skip the
        # remaining rounds entirely.
        if (
            result.get("target_reached")
            and deploy_stop_on_target
            and round_idx != last_round_idx
        ):
            ui.ok(
                "round",
                f"target reached at round_{round_idx}; "
                f"stopping (deploy.stop_when_target_reached=true)",
            )
            break

    # ── opt-in post-rounds weak-cell hardening loop (O4) ───────────────────
    if harden_enabled and last_round_idx >= 0 and final_result.get("rc", 1) == 0:
        hr_idx = last_round_idx + 1
        extra = 0
        while True:
            weak = _weak_cells_below(run_dir, harden_threshold)
            extra += 1
            hr_is_last = (not weak) or (extra > harden_max_rounds)
            ui.banner(
                f"hardening {extra}/{harden_max_rounds}: {len(weak)} cells "
                f"< {harden_threshold:.0%}"
                f"{'  (final -> deploy)' if hr_is_last else ''}",
                ui.C.HEAD,
            )
            if not weak:
                ui.ok(
                    "harden",
                    f"no cells below {harden_threshold:.0%}; "
                    f"deploying current model",
                )
            hr_priors = [run_dir / f"round_{i}" for i in range(hr_idx)]
            result = _run_round(
                round_idx=hr_idx,
                round_mode="active",
                prior_round_dirs=hr_priors,
                run_dir=run_dir,
                cfg=cfg,
                cfg_path=args.config,
                devices=devices,
                arch=arch,
                bench_binary=bench_binary,
                dat_dir=dat_dir,
                match_table=match_table,
                origami_pp=origami_pp,
                from_stage=1,
                force_deploy=args.deploy,
                is_final_round=hr_is_last,
                target_per_cell=target_per_cell,
                shapes_per_cell=shapes_per_cell,
                seed_seed=seed_seed,
                n_strata=n_strata,
                bench_blocks_per_gpu=bench_blocks_per_gpu,
                train_epochs=train_epochs,
                train_device=train_device,
                train_eval_frac=train_eval_frac,
                train_smart_k=train_smart_k,
                train_smart_k_max=train_smart_k_max,
                train_smart_k_min=harden_smart_k_min,
                train_smart_k_saturation_eps=train_smart_k_saturation_eps,
                train_min_cell_gemms=train_min_cell_gemms,
                sel_eff_threshold=sel_eff_threshold,
                validate_min_cell_gemms=validate_min_cell_gemms,
                split_floor=split_floor,
                split_min_improvement=split_min_improvement,
                split_after_attempts=split_after_attempts,
                max_split_depth=max_split_depth,
                structural_split_min_octaves=structural_split_min_octaves,
                structural_split_max_depth=structural_split_max_depth,
                deploy_target_sel_eff=(
                    None
                    if deploy_target_sel_eff in (None, False)
                    else float(deploy_target_sel_eff)
                ),
                deploy_on_final=deploy_on_final,
                deploy_rocm_libraries_to_deploy=deploy_rocm_libraries_to_deploy,
                deploy_apply_patch=deploy_apply_patch,
                deploy_build_after_apply=deploy_build_after_apply,
                deploy_weight_dtype=deploy_weight_dtype,
                deploy_build_jobs=deploy_build_jobs,
            )
            final_result = result
            if result["rc"] != 0:
                ui.err("round", f"hardening round_{hr_idx} FAILED rc={result['rc']}")
                return result["rc"]
            hr_idx += 1
            if hr_is_last:
                break

    # ── cross-round held-out aggregation ───────────────────────────────────
    # `final_result.held_out_sel_eff` is the LAST round's stage04b
    # geomean -- it only covers cells that had new shapes THIS round
    # (typically the retrain set). To answer "how good is the deployed
    # model across the WHOLE grid?" we need the MOST RECENT held-out
    # measurement per cell across all rounds: a cell that PASSED in an
    # early round and never showed up again still counts, at that
    # round's value. Walk every round_R/stage04b/decisions.json, keep the
    # latest entry per cell, and geomean-aggregate weighted by
    # n_eval.
    aggregated = _aggregate_cross_round_held_out_sel_eff(run_dir)
    # Persist the breakdown NOW, BEFORE stage 7/8. It is derived purely from
    # training data (round_*/stage04b) and is independent of the validators, so
    # writing it here means an interrupted/cancelled stage 8 still leaves the
    # report's held-out aggregate + 3D cell map intact. (The end-of-run block
    # below re-writes the identical dict and prints the audit line.)
    try:
        with (run_dir / "held_out_sel_eff_breakdown.json").open("w") as _bf:
            json.dump(aggregated, _bf, indent=2)
    except Exception:
        pass

    # ── auto-trigger stage 7/8 after full/active runs ──────────────────────
    # When the config carries stage07.bench_yamls or stage08.datasets
    # AND we just completed an initial/active/full run that ended
    # cleanly, run the post-deploy validators on the LATEST round.
    # This matches the operator's expectation that "I configured the
    # validators -- they should run". To skip them, omit the lists
    # from the config (or use `--mode validate` separately, in which
    # case _run_validate_mode already ran above and we skip).
    has_validators = (
        bool(_opt(cfg, ["stage07", "bench_yamls"], []) or [])
        or int(_opt(cfg, ["stage07", "parity_sample_trained"], 0) or 0) > 0
        or bool(_opt(cfg, ["stage08", "datasets"], []) or [])
    )
    if args.mode in ("initial", "active", "full") and has_validators:
        completed_rounds = _discover_rounds_under(run_dir)
        if completed_rounds:
            ui.info(
                "validate",
                "auto-running stage 7/8 because config has "
                "`stage07.bench_yamls` and/or `stage08.datasets`. "
                "Set `--mode validate` separately to re-run "
                "without retraining.",
            )
            # Auto-validate ALWAYS uses the latest round and runs
            # BOTH stages -- no operator override at this point.
            # (Use `--mode validate --from-round R --from-stage S`
            # for a targeted re-run.)
            _validate_rc = _run_validate_mode(
                cfg=cfg,
                cfg_path=args.config,
                run_dir=run_dir,
                rounds=completed_rounds,
                devices=devices,
                arch=arch,
                bench_blocks_per_gpu=bench_blocks_per_gpu,
                sel_eff_threshold=sel_eff_threshold,
                validate_min_cell_gemms=validate_min_cell_gemms,
                train_device=train_device,
                deploy_rocm_libraries_to_deploy=deploy_rocm_libraries_to_deploy,
                runtime_rocm_libraries_root=str(
                    _opt(cfg, ["runtime", "rocm_libraries_root"], "")
                )
                or None,
                from_round=0,
                from_stage=7.0,
            )
            if _validate_rc != 0:
                ui.warn(
                    "validate",
                    f"auto-validate exited rc={_validate_rc} "
                    f"(continuing -- training itself succeeded)",
                )

    ui.banner("pipeline_v3 orchestrator  Done", ui.C.GREEN)
    ui.ok("done", f"run_dir : {run_dir}")

    # Per-stage / per-round timing breakdown at the very end so the
    # operator can spot the dominant cost (typically stage03 bench).
    _TIMING.print_summary()

    if final_result and last_round_idx >= 0:
        hse = final_result.get("held_out_sel_eff", float("nan"))
        if not math.isnan(hse):
            ui.ok(
                "done",
                f"last-round held_out_sel_eff : "
                f"{hse*100:.2f}%  "
                f"(stage04b on round_{last_round_idx}'s NEW shapes)",
            )
        if final_result.get("deployed"):
            ui.ok("done", "deploy artifacts emitted (stage06_deploy)")
    if not math.isnan(aggregated["held_out_sel_eff"]):
        ui.ok(
            "done",
            f"final held_out_sel_eff (MODEL) : "
            f"{aggregated['held_out_sel_eff']*100:.2f}%  "
            f"(n_cells={aggregated['n_cells']})",
        )
        # Origami baseline at the SAME per-cell aggregation, so the
        # operator can see model vs analytical at one glance. If
        # Origami didn't measure every cell, we note the coverage
        # so the comparison isn't accidentally apples-to-oranges.
        ori = aggregated["origami_held_out_sel_eff"]
        if not math.isnan(ori):
            n_oc = aggregated["n_cells_with_origami"]
            delta_pp = (aggregated["held_out_sel_eff"] - ori) * 100.0
            sign = "+" if delta_pp >= 0 else ""
            ui.ok(
                "done",
                f"final held_out_sel_eff (ORIGAMI) : "
                f"{ori*100:.2f}%  (n_cells={n_oc})",
            )
            ui.ok(
                "done",
                f"  MODEL - ORIGAMI            : "
                f"{sign}{delta_pp:.2f} pp  "
                f"(positive = model picks better kernels)",
            )
        else:
            ui.warn(
                "done",
                "Origami baseline geomean unavailable (no cell "
                "carried `origami_sel_eff_new`).",
            )
        # Save the breakdown next to orchestrator.log so the operator
        # can audit which round each cell's measurement came from
        # AND see both model + origami numbers per cell.
        with (run_dir / "held_out_sel_eff_breakdown.json").open("w") as f:
            json.dump(aggregated, f, indent=2)
        ui.ok("done", f"breakdown -> {run_dir/'held_out_sel_eff_breakdown.json'}")

    _emit_report(run_dir)
    return 0


def _emit_report(run_dir: Path) -> None:
    """Write the portable HTML report from whatever the run produced.

    Called on EVERY exit path -- success, a failed round, or a failed stage --
    because the report is the artifact an operator actually reads, and a run
    that died in a late round still holds every earlier round's results worth
    seeing. make_report.py fills in what exists and omits what does not.

    Best-effort: report generation must never change the run's exit status.
    """
    try:
        import subprocess as _sp

        _report = run_dir / "report.html"
        _mk = Path(__file__).resolve().parent / "scripts" / "make_report.py"
        _sp.run(
            [sys.executable, str(_mk), str(run_dir), str(_report)],
            check=False,
            timeout=180,
        )
        if _report.exists():
            ui.ok("done", f"report  -> {_report}  (open in any browser)")
        else:
            ui.warn("done", "report generation produced no file")
    except Exception as _e:
        ui.warn("done", f"report generation skipped ({_e!r})")


def _weak_cells_below(run_dir: Path, threshold: float) -> List[str]:
    """Most-recent-per-cell held-out sel_eff (`sel_eff_new`) across every
    round_*/stage04b/decisions.json; return the cells whose latest value is
    below `threshold`. Used by the opt-in post-rounds hardening loop (O4)."""
    latest: Dict[str, float] = {}
    if run_dir.is_dir():
        rs = [
            p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("round_")
        ]
        try:
            rs.sort(key=lambda p: int(p.name[len("round_") :]))
        except ValueError:
            rs.sort()
        for r in rs:
            dp = r / "stage04b" / "decisions.json"
            if not dp.exists():
                continue
            try:
                d = json.loads(dp.read_text())
            except Exception:
                continue
            for c in d.get("per_cell", []):
                se = c.get("sel_eff_new")
                cell = c.get("cell")
                n = c.get("n_eval") or 0
                if cell is not None and se is not None and n > 0:
                    latest[cell] = float(se)
    return sorted([c for c, se in latest.items() if se < threshold])


def _aggregate_cross_round_held_out_sel_eff(run_dir: Path) -> Dict[str, Any]:
    """Walk every `round_*/stage04b/decisions.json` under `run_dir`
    and take the MOST RECENT `sel_eff_new` for each cell. Aggregate
    as a weighted geomean (weights = `n_eval`). ALSO aggregate
    Origami's per-cell `origami_sel_eff_new` from the same row so
    the operator can compare model vs analytical baseline at the
    same per-cell aggregation.

    Why "most recent per cell" instead of "last round's per_cell":
      * Last round's stage04b only re-evaluates the cells that had
        new shapes THIS round -- typically just the retrain set.
        Cells that PASSED several rounds ago wouldn't reappear in
        round_N's decisions.json even though their model is the
        most recent thing we shipped.
      * The MOST RECENT entry per cell, across all rounds, IS the
        held-out measurement for the model the deploy bundle
        carries. Aggregating those is "how well is the deployed
        model doing on unseen shapes, across the whole grid?".
    """
    per_cell_latest: Dict[str, Dict[str, Any]] = {}
    rounds_seen: List[str] = []
    if run_dir.is_dir():
        # Sort by INTEGER round number so per-cell "latest" snapshot
        # correctly preserves the round_14 reading over the round_9
        # reading (lex sort would put `round_14` BEFORE `round_2`,
        # and the assign-once `per_cell_latest` walk would settle
        # on the round_9 value -- wrong).
        _rs = [
            p
            for p in run_dir.iterdir()
            if (
                p.is_dir()
                and p.name.startswith("round_")
                and p.name[len("round_") :].isdigit()
            )
        ]
        _rs.sort(key=lambda p: int(p.name[len("round_") :]))
        for r in _rs:
            decisions_path = r / "stage04b" / "decisions.json"
            if not decisions_path.exists():
                continue
            try:
                with decisions_path.open() as f:
                    d = json.load(f)
            except Exception:
                continue
            rounds_seen.append(r.name)
            for c in d.get("per_cell", []):
                se = c.get("sel_eff_new")
                n = c.get("n_eval") or 0
                if se is None or n <= 0 or float(se) <= 0:
                    continue
                # Later rounds overwrite earlier entries because we
                # walked sorted order; this is exactly "most recent".
                ori_se = c.get("origami_sel_eff_new")
                ori_n = c.get("origami_n_eval") or 0

                def _f(x):
                    return float(x) if isinstance(x, (int, float)) else None

                per_cell_latest[c["cell"]] = {
                    "round": r.name,
                    "sel_eff_new": float(se),
                    "n_eval": int(n),
                    "origami_sel_eff_new": (
                        float(ori_se) if ori_se is not None else None
                    ),
                    "origami_n_eval": int(ori_n),
                    # Held-out latency geomeans (us) over the SAME unseen shapes,
                    # so the report's worst-cells table can show held-out us that
                    # stay consistent with the held-out sel_eff (winner <= pick).
                    "winner_us_geomean": _f(c.get("winner_us_geomean")),
                    "pick_us_geomean": _f(c.get("pick_us_geomean")),
                    "origami_pick_us_geomean": _f(c.get("origami_pick_us_geomean")),
                }

    def _wlog_geomean(pairs: List[Tuple[Optional[float], int]]) -> float:
        log_sum = 0.0
        w_sum = 0.0
        for se, n in pairs:
            if se is None or n <= 0 or float(se) <= 0:
                continue
            log_sum += math.log(float(se)) * float(n)
            w_sum += float(n)
        return math.exp(log_sum / w_sum) if w_sum > 0 else float("nan")

    model_geomean = _wlog_geomean(
        [(e["sel_eff_new"], e["n_eval"]) for e in per_cell_latest.values()]
    )
    # Origami: only count cells where Origami's baseline actually
    # measured something (n_eval > 0). Cells where we couldn't run
    # Origami (no matching configs or rank failed) just don't
    # contribute -- the geomean is still meaningful over the rest.
    origami_geomean = _wlog_geomean(
        [
            (e.get("origami_sel_eff_new"), e.get("origami_n_eval", 0))
            for e in per_cell_latest.values()
        ]
    )
    return {
        "held_out_sel_eff": model_geomean,
        "origami_held_out_sel_eff": origami_geomean,
        "n_cells": len(per_cell_latest),
        "n_cells_with_origami": sum(
            1
            for e in per_cell_latest.values()
            if (
                e.get("origami_sel_eff_new") is not None
                and e.get("origami_n_eval", 0) > 0
            )
        ),
        "n_rounds_aggregated": len(rounds_seen),
        "rounds": rounds_seen,
        "per_cell": per_cell_latest,
    }


if __name__ == "__main__":
    sys.exit(main())
