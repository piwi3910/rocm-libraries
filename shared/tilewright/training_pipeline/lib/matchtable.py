# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt MatchTable.yaml parser.

Each entry maps `sol_idx_global -> (yaml_path, sol_idx_local)`.
MatchTable.yaml has two line shapes Tensile emits:

    14299: [/abs/path/to/yaml, 207]              (inline form)

    14299: [/abs/path/to/yaml,                    (split form)
      207]

We accept both. No full yaml load is performed -- the file can be very
large (50k+ entries on a full hipblaslt build).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple


_LINE_RE = re.compile(r"^(\d+)\s*:\s*\[\s*([^,]+?)\s*,\s*(\d+)\s*\]\s*$")
_OPEN_RE = re.compile(r"^(\d+)\s*:\s*\[\s*([^,]+?)\s*,\s*$")
_CLOSE_RE = re.compile(r"^\s*(\d+)\s*\]\s*$")


def parse_match_table(path: Path) -> Dict[int, Tuple[str, int]]:
    """Returns dict { sol_idx_global -> (yaml_path, sol_idx_local) }."""
    table: Dict[int, Tuple[str, int]] = {}
    pending_key: Optional[int] = None
    pending_path: Optional[str] = None
    with Path(path).open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("---") or not line.strip() or line.strip() == "...":
                continue
            m1 = _LINE_RE.match(line)
            if m1:
                table[int(m1.group(1))] = (m1.group(2), int(m1.group(3)))
                continue
            m2 = _OPEN_RE.match(line)
            if m2:
                pending_key = int(m2.group(1))
                pending_path = m2.group(2)
                continue
            m3 = _CLOSE_RE.match(line)
            if m3 and pending_key is not None and pending_path is not None:
                table[pending_key] = (pending_path, int(m3.group(1)))
                pending_key = None
                pending_path = None
    return table
