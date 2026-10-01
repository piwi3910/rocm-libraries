# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Terminal UI helpers used uniformly across pipeline_v3 stages.

Every stage prints with:
  * coloured `[tag]` log lines (ok / info / warn / err)
  * banners between major phases
  * human-readable duration / integer formatting

Colours are auto-disabled when stdout is not a TTY (e.g. piped into a
file) so log files stay plain-text.
"""
from __future__ import annotations

import sys
import time


class _C:
    _tty = sys.stdout.isatty()
    HEAD = "\033[95m" if _tty else ""
    BLUE = "\033[94m" if _tty else ""
    GREEN = "\033[92m" if _tty else ""
    YELLOW = "\033[93m" if _tty else ""
    RED = "\033[91m" if _tty else ""
    GREY = "\033[90m" if _tty else ""
    ENDC = "\033[0m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""


C = _C  # public alias


def banner(title: str, color: str = _C.HEAD) -> None:
    """80-char-wide divider with bolded title in the middle row."""
    print(f"\n{color}{'='*80}{_C.ENDC}", flush=True)
    print(f"{color}{_C.BOLD}  {title}{_C.ENDC}", flush=True)
    print(f"{color}{'='*80}{_C.ENDC}", flush=True)


def ok(tag: str, msg: str) -> None:
    print(f"{_C.GREEN}[{tag}]{_C.ENDC} {msg}", flush=True)


def info(tag: str, msg: str) -> None:
    print(f"{_C.BLUE}[{tag}]{_C.ENDC} {msg}", flush=True)


def warn(tag: str, msg: str) -> None:
    print(f"{_C.YELLOW}[{tag}]{_C.ENDC} {msg}", flush=True)


def err(tag: str, msg: str) -> None:
    print(f"{_C.RED}[{tag}]{_C.ENDC} {msg}", flush=True)


def grey(tag: str, msg: str) -> None:
    print(f"{_C.GREY}[{tag}]{_C.ENDC} {msg}", flush=True)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_dur(s: float) -> str:
    """Format seconds as '1h23m', '23m45s', '45s' depending on magnitude."""
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def pct(num: float, den: float) -> str:
    if den <= 0:
        return " --.-%"
    return f"{100.0 * num / den:5.1f}%"


class Timer:
    """`with Timer() as t: ...; t.elapsed_s, t.elapsed_str`."""

    def __init__(self) -> None:
        self.t0 = 0.0
        self.elapsed_s = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        return self

    def __exit__(self, *_a) -> None:
        self.elapsed_s = time.time() - self.t0

    @property
    def elapsed_str(self) -> str:
        return fmt_dur(self.elapsed_s)
