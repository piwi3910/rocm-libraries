# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-shape bench probe: compute `iters`, `cold_iters`,
`skip_slow_solution_ratio` for each shape so the full bench pass has
predictable measurement noise.

Why this exists
---------------
hipblaslt-bench's `us` measurement has standard error proportional to
`1 / sqrt(iters)`. The default iters=50 is too noisy on small-K shapes
where many kernels are within a few percent of each other -- noise
swaps the "empirical winner" between adjacent ranks and Origami's
top-1 pick looks much worse than it actually is.

pipeline_v3's stage02 probe calibrates `iters` per shape to hit a
target wall-time-per-kernel from the config so every round has the
same measurement-quality floor.

Probe algorithm (per shape)
---------------------------
1. Run hipblaslt-bench ONCE on the shape with a quick configuration
   (iters=20, cold_iters=20, requested_solution_num=1). This gives a
   single representative `us` reading we call `probe_us`.
2. Decide:
     iters / cold_iters = `--probe-target-iters` (default 283) for
                          `probe_us < probe_us_threshold_us`
                          (default 30us -- "fast" shape; needs lots of
                           iters because the per-iter noise floor is
                           a bigger fraction)
                        = `--probe-target-iters-large` (default 60)
                          otherwise ("slow" shape; per-iter noise floor
                          is small relative to runtime, fewer iters
                          OK).
     skip_slow_solution_ratio = `--ratio-light` (default 0.3) for fast
                              shapes
                              = `--ratio-heavy` (default 0.8) for slow
                              shapes
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bench_yaml import DEFAULT_LINE_KNOBS, parse_knobs, render_bench_line
from .shapes import Shape


@dataclass
class ProbeConfig:
    bench_binary: Path
    device: int = 0
    probe_iters: int = 20
    probe_cold_iters: int = 20
    target_iters_fast: int = 283  # for probe_us < probe_us_threshold
    target_iters_slow: int = 60  # for probe_us >= probe_us_threshold
    probe_us_threshold: float = 30.0  # us; cutoff between "fast" / "slow"
    ratio_light: float = 0.3
    ratio_heavy: float = 0.8


# ── parsing the bench output for the probe's us reading ──────────────────────

_US_LINE_RE = re.compile(r"hipblaslt-Gflops,hipblaslt-GB/s,us")
_US_DATA_RE = re.compile(r",\s*([\d.]+)\s*$")


def _probe_us_from_output(output: str) -> Optional[float]:
    """Find the LAST `us` value in a hipblaslt-bench stdout block (one
    `us` per kernel; we want the one from the requested solution).
    Returns None if no row matched."""
    last_us: Optional[float] = None
    in_data = False
    for line in output.splitlines():
        if _US_LINE_RE.search(line):
            in_data = True
            continue
        if in_data:
            m = _US_DATA_RE.search(line)
            if m:
                try:
                    last_us = float(m.group(1))
                except ValueError:
                    pass
    return last_us


# ── per-shape probe ──────────────────────────────────────────────────────────


def probe_one_shape(
    shape: Shape, config: Dict[str, Any], cfg: ProbeConfig
) -> Optional[float]:
    """Run hipblaslt-bench with rsn=1 + tiny iters and return the
    measured `us`. Returns None on bench failure (caller falls back
    to the default knob set)."""
    knobs = dict(DEFAULT_LINE_KNOBS)
    knobs["iters"] = cfg.probe_iters
    knobs["cold_iters"] = cfg.probe_cold_iters
    knobs["requested_solution_num"] = 1  # one kernel only
    line = render_bench_line(shape, config, knobs)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        tmp = Path(f.name)
        f.write(line + "\n")
    from .bench_env import bench_env_with

    env = bench_env_with({"HIP_VISIBLE_DEVICES": str(cfg.device)})
    try:
        proc = subprocess.run(
            [str(cfg.bench_binary), "--yaml", str(tmp)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        return None
    return _probe_us_from_output(proc.stdout + "\n" + proc.stderr)


def decide_knobs(probe_us: Optional[float], cfg: ProbeConfig) -> Dict[str, Any]:
    """Translate a probe `us` reading into the per-shape bench knobs we
    stamp on the final bench yaml line."""
    if probe_us is None or probe_us < cfg.probe_us_threshold:
        return {
            "iters": cfg.target_iters_fast,
            "cold_iters": cfg.target_iters_fast,
            "skip_slow_solution_ratio": cfg.ratio_light,
        }
    return {
        "iters": cfg.target_iters_slow,
        "cold_iters": cfg.target_iters_slow,
        "skip_slow_solution_ratio": cfg.ratio_heavy,
    }


def probe_shapes(
    shapes: Iterable[Shape],
    config: Dict[str, Any],
    cfg: ProbeConfig,
    on_progress: Optional[Any] = None,
) -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    """Probe every shape and return the per-shape knob dict.

    `on_progress(i, total, shape, probe_us, knobs)` is invoked after
    every probe so the caller can render a progress line. Stage02 wires
    this to a colourful UI; standalone callers can omit it.
    """
    out: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    # Materialise once so the iteration below works whether the caller
    # passed a list, a tuple, or a one-shot generator.
    shapes = list(shapes)
    total = len(shapes)
    for i, s in enumerate(shapes, 1):
        probe_us = probe_one_shape(s, config, cfg)
        k = decide_knobs(probe_us, cfg)
        out[(s.m, s.n, s.k, s.batch)] = k
        if on_progress is not None:
            on_progress(i, total, s, probe_us, k)
    return out
