# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture-name canonicalisation.

hipBLASLt distinguishes an ASIC *revision* from a compiler *target*:

  * The compiler target is what LLVM builds for, and what origami's
    `architecture_t` enumerates -- e.g. `gfx1250`. Every gfx1250 logic
    yaml says `ArchitectureName: gfx1250` regardless of revision.
  * The revision is a separate *tuning* tree selected at runtime from the
    probed `asicRevision` -- e.g. `gfx1250v0`. It shows up as a directory
    name (`Tensile/library/gfx1250v0/`,
    `Logic/asm_full/gfx1250v0/gfx1250v0/`) and, for tilewright, as the
    weights path `weights/<arch>/<arch>/`.

This pipeline's `arch` config key must be the REVISION (`gfx1250v0`),
because that is what selects the right kernels and the right deploy
path. But anything that indexes a *hardware model* -- origami's
`architecture_t`, our `_MI_LATENCY_TABLE` -- is keyed by the compiler
target, since both revisions execute the same ISA.

`canonical_arch` is the one place that bridges the two. Without it a
`gfx1250v0` run silently falls back to gfx950 MI latencies and loses
origami's hardware constants, with only a warning to show for it.
"""
from __future__ import annotations

import re

# `gfx1250v0` -> `gfx1250`. Deliberately anchored and narrow: only a
# trailing `v<digits>` is stripped, so real target names that merely end
# in a digit (`gfx950`, `gfx1201`) are untouched.
_REVISION_SUFFIX_RE = re.compile(r"^(gfx[0-9a-f]+?)v\d+$")


def canonical_arch(arch: str) -> str:
    """Return the compiler target for an arch string that may name an
    ASIC revision. `gfx1250v0` -> `gfx1250`; anything else unchanged."""
    if not arch:
        return arch
    m = _REVISION_SUFFIX_RE.match(str(arch).strip())
    return m.group(1) if m else str(arch).strip()


def is_revisioned(arch: str) -> bool:
    """True when `arch` names a specific ASIC revision (`gfx1250v0`)
    rather than a bare compiler target (`gfx1250`)."""
    return canonical_arch(arch) != str(arch or "").strip()
