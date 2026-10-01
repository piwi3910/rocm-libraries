#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Collect every hipblaslt-bench failure in a stage03 log directory.

For each failure emits the GEMM shape and the kernel that was in flight when
the GPU faulted, so the crash can be reproduced from the shape + solution
index alone. stage03's own `skipped_gemms.txt` records only the handful of
shapes it retried; the aperture violations that kill a whole block never make
it there, and the block's remaining shapes are lost with no record of what was
running.

Usage: collect_bench_failures.py <stage03/logs dir> [more dirs...]
"""
from __future__ import annotations

import csv
import os
import re
import sys

HEADER_RE = re.compile(r"^\[(\d+)\]:transA")
SOL_IDX_RE = re.compile(r"^\s*--Solution index:\s+(\d+)")
SOL_NAME_RE = re.compile(r"^\s*--Solution name:\s+(\S+)")
SKIP_RE = re.compile(r"^Skip solution:\s+(\d+).*?solution index = (\d+)")
WINNER_RE = re.compile(r"^Winner:")

# Anything that means the GPU or the process died, as opposed to a slow kernel.
FATAL_RE = re.compile(
    r"(HSA_STATUS_ERROR[A-Z_]*"
    r"|Memory access fault"
    r"|hang analysis"
    r"|GPU core dump"
    r"|Segmentation fault"
    r"|terminate called"
    r"|rocblaslt error)",
    re.I,
)

# stage03's own crash bookkeeping. These carry the shape explicitly and are
# where most losses show up once a device has wedged: the bench never emits a
# kernel line at all, it just produces no output until the stall watchdog
# kills it.
CRASH_SHAPE_RE = re.compile(
    r"\*\*\* SKIPPED CRASHED GEMM at block-offset (\d+) \(yaml line (\d+)\):\s*"
    r"M=(\d+)\s+N=(\d+)\s+K=(\d+)\s*\(rc=(-?\d+)\)"
)
STALL_RE = re.compile(r"\*\*\* KILLED \[(\w+)\]: no output for (\d+)s")
ABORT_RE = re.compile(r"\*\*\* ABORT: (\d+) consecutive sub-runs produced no output")


def parse_log(path):
    """Yield one dict per fatal event in a single block log."""
    gpu = ""
    m_gpu = re.search(r"_gpu(\d+)\.log$", path)
    if m_gpu:
        gpu = m_gpu.group(1)
    block = os.path.basename(path).split("_lines")[0]

    shape = {}
    sol_idx = sol_name = ""
    n_winners = 0
    events = []

    with open(path, errors="replace") as fh:
        lines = fh.readlines()

    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")

        if HEADER_RE.match(line):
            shape = {}  # data row follows on the next line
            sol_idx = sol_name = ""
            continue
        if WINNER_RE.match(line):
            n_winners += 1
            continue

        # The CSV data row directly under a [rank]: header carries the shape.
        if not shape and line.startswith("    T,") or line.startswith("    N,"):
            f = line.strip().split(",")
            if len(f) > 6:
                try:
                    shape = {
                        "transA": f[0],
                        "transB": f[1],
                        "batch": int(f[3]),
                        "m": int(f[4]),
                        "n": int(f[5]),
                        "k": int(f[6]),
                    }
                except ValueError:
                    pass
            continue

        mi = SOL_IDX_RE.match(line)
        if mi:
            sol_idx = mi.group(1)
            continue
        mn = SOL_NAME_RE.match(line)
        if mn:
            sol_name = mn.group(1)
            continue
        ms = SKIP_RE.match(line)
        if ms:
            sol_idx = ms.group(2)
            sol_name = "(skipped)"
            continue

        mc = CRASH_SHAPE_RE.search(line)
        if mc:
            events.append(
                {
                    "block": block,
                    "gpu": gpu,
                    "line_no": i,
                    "m": int(mc.group(3)),
                    "n": int(mc.group(4)),
                    "k": int(mc.group(5)),
                    "batch": "",
                    "transA": "T",
                    "transB": "N",
                    "sol_idx": "",
                    "sol_name": "",
                    "error": f"crash_skipped(rc={mc.group(6)})",
                    "detail": f"yaml line {mc.group(2)}, block-offset {mc.group(1)}",
                }
            )
            continue
        ma = ABORT_RE.search(line)
        if ma:
            events.append(
                {
                    "block": block,
                    "gpu": gpu,
                    "line_no": i,
                    "m": "",
                    "n": "",
                    "k": "",
                    "batch": "",
                    "transA": "",
                    "transB": "",
                    "sol_idx": "",
                    "sol_name": "",
                    "error": "block_abandoned",
                    "detail": f"{ma.group(1)} consecutive sub-runs produced no "
                    f"output (device wedged); rest of block lost",
                }
            )
            continue

        if FATAL_RE.search(line):
            # Collapse the multi-line HSA dump into one event per shape.
            if (
                events
                and events[-1]["line_no"] > i - 25
                and events[-1]["m"] == shape.get("m", "")
                and events[-1]["sol_idx"] == sol_idx
            ):
                events[-1]["detail"] += " | " + line.strip()[:120]
                continue
            events.append(
                {
                    "block": block,
                    "gpu": gpu,
                    "line_no": i,
                    "m": shape.get("m", ""),
                    "n": shape.get("n", ""),
                    "k": shape.get("k", ""),
                    "batch": shape.get("batch", ""),
                    "transA": shape.get("transA", ""),
                    "transB": shape.get("transB", ""),
                    "sol_idx": sol_idx,
                    "sol_name": sol_name,
                    "error": FATAL_RE.search(line).group(1),
                    "detail": line.strip()[:160],
                }
            )

    truncated = not any(WINNER_RE.match(l) for l in lines[-40:]) and n_winners > 0
    return events, n_winners, truncated


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    all_events = []
    block_rows = []
    for d in argv[1:]:
        for name in sorted(os.listdir(d)):
            if not name.startswith("block_") or not name.endswith(".log"):
                continue
            p = os.path.join(d, name)
            ev, nw, trunc = parse_log(p)
            all_events += ev
            block_rows.append(
                (
                    os.path.basename(os.path.dirname(os.path.dirname(d))) or d,
                    name,
                    nw,
                    len(ev),
                    trunc,
                )
            )

    out = "bench_failures.csv"
    cols = [
        "block",
        "gpu",
        "m",
        "n",
        "k",
        "batch",
        "transA",
        "transB",
        "sol_idx",
        "sol_name",
        "error",
        "detail",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in all_events:
            w.writerow(e)

    print(f"fatal events : {len(all_events)}   -> {out}")
    print(
        f"blocks parsed: {len(block_rows)}   "
        f"truncated (died mid-block): "
        f"{sum(1 for r in block_rows if r[4])}"
    )

    if all_events:
        print("\n-- failures (shape / kernel / error) " + "-" * 40)
        for e in all_events:
            print(
                f"  {e['block']} gpu{e['gpu']}  "
                f"{e['transA']}{e['transB']} m={e['m']} n={e['n']} "
                f"k={e['k']} b={e['batch']}  sol={e['sol_idx']}  "
                f"{e['error']}"
            )
            if e["sol_name"] and e["sol_name"] != "(skipped)":
                print(f"        kernel: {e['sol_name'][:130]}")

        from collections import Counter

        print("\n-- by solution index " + "-" * 56)
        for idx, c in Counter(e["sol_idx"] for e in all_events).most_common():
            print(f"  sol {idx or '(unknown)'}: {c}")
        print("\n-- by error " + "-" * 64)
        for err, c in Counter(e["error"] for e in all_events).most_common():
            print(f"  {err}: {c}")

    print("\n-- blocks that died mid-stream " + "-" * 46)
    for run, name, nw, nev, trunc in block_rows:
        if trunc or nev:
            print(
                f"  {name}  winners={nw}  fatal_events={nev}  "
                f"{'TRUNCATED' if trunc else ''}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
