# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small filesystem helpers used by multiple stages.

Centralized here so we get consistent atomic writes + JSON shapes
across the pipeline (and no individual stage has to reinvent
`json.dump(..., indent=2)` over and over).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def atomic_write_text(path: Path, contents: str) -> None:
    """Write `contents` to `path` atomically (write to a temp file in
    the same directory, fsync, then rename). Cheap insurance against
    half-written files when a long-running stage is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Binary counterpart of `atomic_write_text` (temp file in the same
    dir, fsync, rename). Used for the deploy `.bin` so a SIGTERM / OOM
    mid-write can never leave a torn weights blob that the C++ runtime
    would load (or silently reject)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: Path, obj: Any, *, indent: Optional[int] = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent) + "\n")


def read_json(path: Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def chunked_csv_paths(folder: Path, prefix: str = "chunk_") -> list:
    """Return the list of chunk_*.csv files in `folder`, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [
            folder / p
            for p in os.listdir(folder)
            if p.startswith(prefix) and p.endswith(".csv")
        ]
    )
