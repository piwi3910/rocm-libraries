# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage02_probe -- measure per-shape `us`, then calibrate the final
bench yaml's iters / cold_iters / skip_slow_solution_ratio per shape.

Why this stage exists
---------------------
hipblaslt-bench's per-measurement noise standard error scales like
`1 / sqrt(iters)`. Using a constant `iters` for every shape either
wastes time on slow shapes (overkill) or yields noisy winners on fast
shapes (under-measured), which silently demotes the true best kernel
into "tied for second" and tanks the Origami / model sel_eff.

So we calibrate per shape:

  iters_final = max(1, round(duration_us / probe_us))
  cold_iters_final = iters_final
  skip_slow_solution_ratio = ratio_light  if probe_us < fast_us_threshold
                           = ratio_heavy  otherwise

All of `duration_us`, `fast_us_threshold`, `ratio_light`, `ratio_heavy`
live in the cell config's `probe:` section so they're checked in, NOT
CLI flags.

Probe workflow
--------------
  1. Split stage01's shapes.yaml into N TEMP yamls (round-robin),
     one per device in `--devices`. The first line of each temp yaml
     gets (first_line_iters, first_line_cold_iters); every other line
     gets (other_lines_iters, other_lines_cold_iters). Every probe
     line is also stamped with `requested_solution_num: 1` -- the
     probe just needs ONE kernel per shape to measure latency.
  2. Launch hipblaslt-bench in PARALLEL across all `--devices`. Each
     process produces a log with the bench's single-kernel `us`
     reading per shape.
  3. Parse the per-shape probe_us out of each device's log; map it
     back to the shape's index in stage01's shapes.yaml.
  4. Compute the calibrated knobs above per shape.
  5. Write ONE shapes_probed.yaml (lines in the original shape order;
     `requested_solution_num` reset to -1 so stage03's full bench
     tests every kernel) ready for stage03.

Outputs in `--out-dir`:

  shapes_probed.yaml                            the single calibrated
                                                yaml stage03 reads
  logs/probe_NNNN_linesX-Y_gpuD.log             per-device probe logs.
                                                Same naming convention
                                                as stage03.
                                                (Per-device probe yamls
                                                live in temp files,
                                                deleted after the probe.)

CLI:

  python stage02_probe.py \\
      --in-dir       <run>/round_<R>/stage01 \\
      --out-dir      <run>/round_<R>/stage02 \\
      --config-yaml  configs/gfx950_bbs_tn.yaml \\
      --bench-binary <hipblaslt build>/clients/hipblaslt-bench \\
      --devices 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_V3 = THIS_DIR.parent
if str(PIPELINE_V3) not in sys.path:
    sys.path.insert(0, str(PIPELINE_V3))

import yaml  # noqa: E402

from lib import ui  # noqa: E402


# ── per-line bench-yaml knob rewriting ───────────────────────────────────────

_ITERS_RE = re.compile(r"(?<!\w)iters:\s*\d+")
_COLD_RE = re.compile(r"(?<!\w)cold_iters:\s*\d+")
_SKIP_RE = re.compile(r"(?<!\w)skip_slow_solution_ratio:\s*[\d.]+")
_RSN_RE = re.compile(r"(?<!\w)requested_solution_num:\s*-?\d+")


def _stamp_iters_in_line(
    line: str,
    iters: int,
    cold_iters: int,
    skip_ratio: Optional[float] = None,
    rsn: Optional[int] = None,
) -> str:
    line = _COLD_RE.sub(f"cold_iters: {cold_iters}", line, count=1)
    line = _ITERS_RE.sub(f"iters: {iters}", line, count=1)
    if "iters:" not in line:
        line = (
            line.rstrip("\n").rstrip("}")
            + f", iters: {iters}, cold_iters: {cold_iters}}}\n"
        )
        if not line.endswith("\n"):
            line += "\n"
    if skip_ratio is not None:
        # Always 1 decimal: 0.3 / 0.8 etc., never 0.300 / 0.800.
        sr = f"{skip_ratio:.1f}"
        if _SKIP_RE.search(line):
            line = _SKIP_RE.sub(f"skip_slow_solution_ratio: {sr}", line, count=1)
        else:
            line = (
                line.rstrip("\n").rstrip("}") + f", skip_slow_solution_ratio: {sr}}}\n"
            )
            if not line.endswith("\n"):
                line += "\n"
    if rsn is not None:
        if _RSN_RE.search(line):
            line = _RSN_RE.sub(f"requested_solution_num: {rsn}", line, count=1)
        else:
            line = (
                line.rstrip("\n").rstrip("}") + f", requested_solution_num: {rsn}}}\n"
            )
            if not line.endswith("\n"):
                line += "\n"
    return line


# ── probe-pass split (multi-device) ──────────────────────────────────────────


def _read_shape_lines(in_yaml: Path) -> List[str]:
    """Read every `- {...}` line from a bench yaml, preserving order."""
    out: List[str] = []
    with in_yaml.open() as f:
        for line in f:
            if line.strip().startswith("-"):
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def _build_probe_yamls(
    lines: List[str],
    devices: List[int],
    first_iters: int,
    first_cold: int,
    other_iters: int,
    other_cold: int,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Round-robin distribute `lines` across `devices` and return
    (per_device_lines, per_device_orig_indices). NO disk I/O: each
    device's yaml is materialised into a temp file inside its
    `_ProbeRunner.run()` and removed after the probe.

    First line of each device's bucket gets (first_iters, first_cold);
    every subsequent line gets (other_iters, other_cold). Every probe
    line is stamped with `requested_solution_num: 1` -- only one kernel
    per shape is needed to measure latency."""
    n_dev = len(devices)
    buckets: List[List[Tuple[int, str]]] = [[] for _ in range(n_dev)]
    for i, ln in enumerate(lines):
        buckets[i % n_dev].append((i, ln))
    per_dev_lines: List[List[str]] = []
    per_dev_indices: List[List[int]] = []
    for bucket in buckets:
        ord_idx = [i for i, _ in bucket]
        stamped: List[str] = []
        for j, (_, ln) in enumerate(bucket):
            if j == 0:
                stamped.append(_stamp_iters_in_line(ln, first_iters, first_cold, rsn=1))
            else:
                stamped.append(_stamp_iters_in_line(ln, other_iters, other_cold, rsn=1))
        per_dev_lines.append(stamped)
        per_dev_indices.append(ord_idx)
    return per_dev_lines, per_dev_indices


# ── per-shape probe_us parsing ───────────────────────────────────────────────

# hipblaslt-bench CSV-style output: each shape produces a row (or
# multiple rows when rsn=-1) of the form
#    transA,transB,...,hipblaslt-Gflops,hipblaslt-GB/s,us
#    T,N,...,123.4,5.6,17.3
# We extract `us` (the last numeric token after the last comma).
# When a shape's bench tested multiple kernels (rsn=-1), there are
# several rows. We use the SMALLEST `us` (= the winner) as the
# shape's probe_us. When rsn=1, there is only one row -> that is the
# probe_us.
_HEADER_RE = re.compile(r"hipblaslt-Gflops,hipblaslt-GB/s,us")
_LAST_NUM_RE = re.compile(r",\s*([\d.]+)\s*$")


def _parse_per_shape_winner_us(log_path: Path) -> List[Optional[float]]:
    """Return a list with ONE probe_us per shape, in the order the
    bench encountered them.

    The bench delimits shapes via a `Winner:` line (or by re-emitting
    the header). We use the simpler rule: between successive header
    lines, take the MIN `us` across all data rows -> that's the
    winner's us for that shape."""
    per_shape: List[Optional[float]] = []
    cur_min: Optional[float] = None
    in_block = False
    if not log_path.exists():
        return per_shape
    # `us` is located by NAME from the header row, not by taking the last
    # numeric token. Under adaptive timing the bench appends
    # `batch,samples,hot_iters,mean_us,min_us,cv,rel_iqr,status` after `us`,
    # so the row ends in a WORD (`converged`) and the old last-number regex
    # matched nothing -- every shape would have silently come back unmeasured.
    us_idx: Optional[int] = None
    with log_path.open() as f:
        for line in f:
            if _HEADER_RE.search(line):
                # End previous block (if any), start a new one.
                if in_block:
                    per_shape.append(cur_min)
                cur_min = None
                in_block = True
                hdr = line.split("]:", 1)[-1] if "]:" in line else line
                cols = [c.strip() for c in hdr.strip().split(",")]
                us_idx = cols.index("us") if "us" in cols else None
                continue
            if not in_block:
                continue
            us = None
            if us_idx is not None:
                parts = [c.strip() for c in line.strip().split(",")]
                if len(parts) > us_idx:
                    try:
                        us = float(parts[us_idx])
                    except ValueError:
                        us = None
            if us is None:
                # Pre-adaptive layout (or a header we could not read): fall
                # back to the historical last-numeric-token behaviour.
                m = _LAST_NUM_RE.search(line.rstrip())
                if not m:
                    continue
                try:
                    us = float(m.group(1))
                except ValueError:
                    continue
            if cur_min is None or us < cur_min:
                cur_min = us
    if in_block:
        per_shape.append(cur_min)
    return per_shape


# ── parallel bench launcher (private; mirrors stage03's _DeviceRunner) ──────


class _ProbeRunner(threading.Thread):
    """Probe one device: write `lines` to a temp yaml, run
    hipblaslt-bench on it with `--device <D>`, tee stdout to
    `log_path`, then delete the temp yaml."""

    def __init__(
        self, device: int, lines: List[str], bench_binary: Path, log_path: Path
    ) -> None:
        super().__init__(name=f"probe-d{device}", daemon=True)
        self.device = device
        self.lines = lines
        self.bench_binary = bench_binary
        self.log_path = log_path
        self.rc: Optional[int] = None
        self.wall_s: float = 0.0

    def run(self) -> None:
        # Use hipblaslt-bench's --device flag (NOT env-var masking)
        # so all processes share the same global device numbering.
        t0 = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_yaml: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                tmp_yaml = Path(f.name)
                for ln in self.lines:
                    f.write(ln)
            from lib.bench_env import bench_env_with  # noqa: E402

            # buffering=1 + flush() so a `tail -f` of the per-device
            # probe log shows progress as the bench runs, not at exit.
            with self.log_path.open("w", buffering=1) as logf:
                proc = subprocess.Popen(
                    [
                        str(self.bench_binary),
                        "--device",
                        str(self.device),
                        "--yaml",
                        str(tmp_yaml),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=bench_env_with(),
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                self.rc = proc.wait()
        except Exception:
            self.rc = -1
        finally:
            if tmp_yaml is not None:
                try:
                    tmp_yaml.unlink()
                except OSError:
                    pass
        self.wall_s = time.time() - t0


# ── calibrate per-shape knobs ───────────────────────────────────────────────


def _calibrate(probe_us: Optional[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a probe `us` into the final bench knobs we stamp on
    that shape's line.

    `duration_us` ALONE drives iters -- no min/max cap in config. We
    only clamp at iters >= 1 (the bench tool's hard floor; iters=0
    would crash hipblaslt-bench). cold_iters always equals iters."""
    duration_us = float(cfg["duration_us"])
    # skip_ratio_mode: "bucket" (default, legacy 2-tier) | "sigmoid"
    # (continuous logistic ramp in log-latency, bounded to
    # [skip_ratio_min, skip_ratio_max]). Default keeps existing runs bit-
    # identical; sigmoid is opt-in.
    mode = str(cfg.get("skip_ratio_mode", "bucket")).lower()
    # Legacy 2-tier knobs. OPTIONAL -- only consulted in bucket mode;
    # sigmoid mode ignores them entirely (it uses skip_ratio_* instead),
    # so sigmoid configs need not declare them.
    fast_threshold = float(cfg.get("fast_us_threshold", 30.0))
    ratio_light = float(cfg.get("ratio_light", 0.30))
    ratio_heavy = float(cfg.get("ratio_heavy", 0.80))
    if probe_us is None or probe_us <= 0:
        # Bench failed for this shape; fall back to the active mode's
        # lightest prune so it still runs (a few hundred iters at the
        # bench's own pace).
        fail_ratio = (
            float(cfg.get("skip_ratio_min", 0.05)) if mode == "sigmoid" else ratio_light
        )
        return {
            "iters": 100,
            "cold_iters": 100,
            "skip_slow_solution_ratio": fail_ratio,
            "probe_us": None,
            "bucket": "fail",
        }
    iters = max(1, int(round(duration_us / probe_us)))
    if mode == "sigmoid":
        # Continuous, latency-aware skip ratio. Slower GEMMs (clearer
        # winner separation, costlier to bench every candidate solution)
        # prune harder; fast/near-tied GEMMs prune little to protect the
        # winner label. `center` is the 50% pivot latency, `k` the
        # steepness in natural-log-latency units.
        import math

        lo = float(cfg.get("skip_ratio_min", 0.05))
        hi = float(cfg.get("skip_ratio_max", 0.95))
        center = float(cfg.get("skip_ratio_center_us", fast_threshold))
        k = float(cfg.get("skip_ratio_k", 1.5))
        x = math.log(max(probe_us, 1e-9) / max(center, 1e-9))
        ratio = round(lo + (hi - lo) / (1.0 + math.exp(-k * x)), 4)
        bucket = "sig"
    else:
        bucket = "fast" if probe_us < fast_threshold else "slow"
        ratio = ratio_light if bucket == "fast" else ratio_heavy
    return {
        "iters": iters,
        "cold_iters": iters,
        "skip_slow_solution_ratio": ratio,
        "probe_us": probe_us,
        "bucket": bucket,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage02 -- probe per-shape us then calibrate final "
        "iters / cold_iters / skip_slow_solution_ratio."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="dir from stage01 (must contain shapes.yaml).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="dir to write shapes_probed.yaml + probe.json + " "probe_split/.",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        required=True,
        help="OOB configuration yaml -- `probe:` section "
        "drives iter counts, target duration, ratios.",
    )
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated GPU device IDs for the parallel " "probe pass.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f)
    pcfg = cfg.get("probe", {})
    required_keys = [
        "first_line_iters",
        "first_line_cold_iters",
        "other_lines_iters",
        "other_lines_cold_iters",
        "duration_us",
    ]
    # The 2-tier bucket knobs are only required when that mode is active;
    # sigmoid configs omit them (they use skip_ratio_* instead).
    if str(pcfg.get("skip_ratio_mode", "bucket")).lower() == "bucket":
        required_keys += ["fast_us_threshold", "ratio_light", "ratio_heavy"]
    missing = [k for k in required_keys if k not in pcfg]
    if missing:
        ui.err("cfg", f"cell config probe: missing {missing}")
        return 1

    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    in_yaml = args.in_dir / "shapes.yaml"
    if not in_yaml.exists():
        ui.err("err", f"missing shapes.yaml at {in_yaml}")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"missing hipblaslt-bench: {args.bench_binary}")
        return 1

    if not args.quiet:
        ui.banner("stage02 probe -> calibrate iters")
        ui.info("cfg", f"in_yaml      : {in_yaml}")
        ui.info("cfg", f"devices      : {devices}")
        ui.info("cfg", f"duration_us  : {pcfg['duration_us']}")
        ui.info(
            "cfg",
            f"probe iters  : first={pcfg['first_line_iters']}/"
            f"{pcfg['first_line_cold_iters']}  "
            f"other={pcfg['other_lines_iters']}/"
            f"{pcfg['other_lines_cold_iters']}",
        )
        # skip-ratio knobs differ by mode; the bucket knobs are absent in
        # sigmoid configs, so only print the ones the active mode uses.
        if str(pcfg.get("skip_ratio_mode", "bucket")).lower() == "sigmoid":
            ui.info(
                "cfg",
                f"skip ratio   : sigmoid "
                f"min={pcfg.get('skip_ratio_min', 0.05)} "
                f"max={pcfg.get('skip_ratio_max', 0.95)} "
                f"center={pcfg.get('skip_ratio_center_us', 30)}us "
                f"k={pcfg.get('skip_ratio_k', 1.5)}",
            )
        else:
            ui.info(
                "cfg",
                f"fast cutoff  : {pcfg['fast_us_threshold']}us  "
                f"light={pcfg['ratio_light']}  heavy={pcfg['ratio_heavy']}",
            )
        ui.info(
            "cfg",
            f"iters floor=1; cold_iters == iters; no upper cap "
            "-- duration_us drives the count",
        )

    # ── Step 1: build per-device probe yamls in memory ─────────────────────
    if not args.quiet:
        ui.banner("Step 1/4  Build per-device probe yamls (in-memory)", ui.C.BLUE)
    lines = _read_shape_lines(in_yaml)
    n_total = len(lines)
    if n_total == 0:
        ui.err("err", "no shape lines in shapes.yaml")
        return 1
    per_dev_lines, per_dev_indices = _build_probe_yamls(
        lines,
        devices,
        int(pcfg["first_line_iters"]),
        int(pcfg["first_line_cold_iters"]),
        int(pcfg["other_lines_iters"]),
        int(pcfg["other_lines_cold_iters"]),
    )
    if not args.quiet:
        for dev, idx in zip(devices, per_dev_indices):
            ui.ok(
                "split",
                f"  device {dev}: {len(idx):>6,} shapes "
                f"(line range {idx[0]+1}..{idx[-1]+1})",
            )

    # ── Step 2: launch probe in parallel ────────────────────────────────────
    if not args.quiet:
        ui.banner("Step 2/4  Run probe on all devices in parallel", ui.C.BLUE)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runners: List[_ProbeRunner] = []
    # Probe log naming mirrors stage03's block log naming so the format
    # is uniform across the pipeline:
    #   probe_NNNN_linesX-Y_gpuD.log
    # where N is the bucket index (0-based), X..Y is the 1-based source-
    # line range (NOTE: probe assignment is round-robin -> the line range
    # is the smallest contiguous range that covers this device's lines).
    for d_idx, (dev, idx_list) in enumerate(zip(devices, per_dev_indices)):
        start_line = idx_list[0] + 1
        end_line = idx_list[-1] + 1
        log = logs_dir / (
            f"probe_{d_idx:04d}_" f"lines{start_line}-{end_line}_" f"gpu{dev}.log"
        )
        runners.append(_ProbeRunner(dev, per_dev_lines[d_idx], args.bench_binary, log))
    t0 = time.time()
    for r in runners:
        r.start()
    last_print = 0.0
    while any(r.is_alive() for r in runners):
        time.sleep(1.0)
        if not args.quiet and time.time() - last_print >= 10.0:
            done = sum(1 for r in runners if not r.is_alive())
            alive = [r.device for r in runners if r.is_alive()]
            ui.grey(
                "probe",
                f"devices_done={done}/{len(runners)}  "
                f"alive={alive}  elapsed={ui.fmt_dur(time.time()-t0)}",
            )
            last_print = time.time()
    for r in runners:
        r.join()
    n_ok = sum(1 for r in runners if r.rc == 0)
    n_fail_dev = sum(1 for r in runners if r.rc != 0)
    for r in runners:
        if r.rc == 0:
            if not args.quiet:
                ui.ok(
                    "probe",
                    f"  device {r.device}: rc=0  " f"wall={ui.fmt_dur(r.wall_s)}",
                )
        else:
            ui.err(
                "probe",
                f"  device {r.device}: rc={r.rc}  "
                f"log={r.log_path}  (shapes for this device will get "
                f"default knobs)",
            )
    if n_ok == 0:
        # NOT fatal by itself. hipblaslt-bench processes the yaml in order and
        # streams a result row per shape, so a device that segfaults on shape
        # N still leaves N-1 valid measurements in its log. Discarding those
        # because the process exited non-zero throws away most of a run: even
        # when every device crashes part-way through, their logs together can
        # still hold many good rows. Parse what landed and let the un-probed
        # remainder take default knobs; only give up in Step 3 if NOTHING was
        # measured anywhere.
        ui.warn(
            "probe",
            "every probe device exited non-zero; salvaging the "
            "partial measurements each one wrote before it died",
        )
    if n_fail_dev > 0:
        ui.warn(
            "probe",
            f"{n_fail_dev}/{len(runners)} devices failed; "
            f"keeping the {n_ok} healthy devices' calibrations and "
            f"falling back to default knobs for the rest",
        )

    # ── Step 3: parse per-shape probe_us (skip failed devices) ──────────────
    if not args.quiet:
        ui.banner("Step 3/4  Parse per-shape probe_us", ui.C.BLUE)
    per_shape_us: List[Optional[float]] = [None] * n_total
    for r, idx_list in zip(runners, per_dev_indices):
        # Deliberately NOT skipping rc != 0: see the salvage note above. The
        # length check below already handles a short (truncated) parse.
        us_list = _parse_per_shape_winner_us(r.log_path)
        if len(us_list) < len(idx_list):
            ui.warn(
                "parse",
                f"  device {r.device}: parsed {len(us_list)}/"
                f"{len(idx_list)} shapes (rest -> default knobs)",
            )
        for shape_idx, us in zip(idx_list, us_list):
            per_shape_us[shape_idx] = us
    n_have = sum(1 for u in per_shape_us if u is not None)
    if n_have == 0:
        ui.err(
            "err",
            "no shape on any device produced a probe measurement; "
            "cannot produce shapes_probed.yaml (check the probe logs -- "
            "a total failure here usually means the bench cannot run this "
            "config's dtype/layout at all, not that a few shapes are bad)",
        )
        return 2
    if not args.quiet:
        ui.ok("parse", f"probe_us populated for {n_have:,}/{n_total:,} shapes")

    # ── Step 4: calibrate + write shapes_probed.yaml ────────────────────────
    if not args.quiet:
        ui.banner("Step 4/4  Calibrate + write shapes_probed.yaml", ui.C.BLUE)
    n_fast = n_slow = n_sig = n_fail = 0
    out_yaml = args.out_dir / "shapes_probed.yaml"
    with out_yaml.open("w") as out:
        for i, (line, us) in enumerate(zip(lines, per_shape_us)):
            knobs = _calibrate(us, pcfg)
            # Restore rsn=-1 explicitly so stage03's full bench tests
            # every kernel (the probe forced rsn=1 for speed; that
            # value is on the *probe* yamls only and never reaches the
            # original `line` we're stamping now).
            new_line = _stamp_iters_in_line(
                line,
                knobs["iters"],
                knobs["cold_iters"],
                skip_ratio=knobs["skip_slow_solution_ratio"],
                rsn=-1,
            )
            out.write(new_line)
            # Buckets are mode-dependent: bucket mode emits fast/slow,
            # sigmoid mode emits "sig" for every SUCCESSFULLY calibrated
            # shape, and only a genuinely failed probe emits "fail".
            # Counting "not fast and not slow" as failures made a healthy
            # sigmoid run report `fail(default)=<every shape>`, which reads
            # like a total probe wipeout when in fact nothing went wrong.
            b = knobs["bucket"]
            if b == "fast":
                n_fast += 1
            elif b == "slow":
                n_slow += 1
            elif b == "sig":
                n_sig += 1
            else:
                n_fail += 1
    if not args.quiet:
        if n_sig:
            ui.ok(
                "calib", f"sigmoid-calibrated={n_sig:,}  " f"fail(default)={n_fail:,}"
            )
        else:
            ui.ok(
                "calib",
                f"fast={n_fast:,}  slow={n_slow:,}  " f"fail(default)={n_fail:,}",
            )
        ui.ok("write", f"shapes_probed.yaml: {n_total:,} lines -> {out_yaml}")
    # (No probe.json -- per-shape probe_us + calibrated knobs are
    #  already visible inside shapes_probed.yaml on each line. Per-
    #  device probe logs live under <out-dir>/logs/.)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.info(
            "next", "stage03_load_balance_offline_tuning.py over " "shapes_probed.yaml."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
