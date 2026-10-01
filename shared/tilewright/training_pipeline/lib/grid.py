# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grid tier definitions -- the 96-cell categorical partition of shape
space pipeline_v3 trains one model per.

Tiers:
    M, N : Tiny (<=32)  Small (<=128)  Mid (<=512)  Large (>512)
    K    : TinyK (<=32) MidK (<=512)   LargeK (>512)
    B    : Bnone (B==1) Bany (B>1)

Cell label format:  '<Mtier>|<Ntier>|<Ktier>|<Btier>'
"""
from __future__ import annotations

from typing import List


def m_tier(M: int) -> str:
    if M <= 32:
        return "Tiny"
    if M <= 128:
        return "Small"
    if M <= 512:
        return "Mid"
    return "Large"


def n_tier(N: int) -> str:
    if N <= 32:
        return "Tiny"
    if N <= 128:
        return "Small"
    if N <= 512:
        return "Mid"
    return "Large"


def k_tier(K: int) -> str:
    if K <= 32:
        return "TinyK"
    if K <= 512:
        return "MidK"
    return "LargeK"


def b_tier(B: int) -> str:
    return "Bnone" if B == 1 else "Bany"


def cell_key(M: int, N: int, K: int, B: int) -> str:
    return f"{m_tier(M)}|{n_tier(N)}|{k_tier(K)}|{b_tier(B)}"


M_TIERS = ("Tiny", "Small", "Mid", "Large")
K_TIERS_ALL = ("TinyK", "MidK", "LargeK")
B_TIERS = ("Bnone", "Bany")


def all_cell_labels() -> List[str]:
    return [
        f"{m}|{n}|{k}|{b}"
        for m in M_TIERS
        for n in M_TIERS
        for k in K_TIERS_ALL
        for b in B_TIERS
    ]


# Per-tier numeric ranges used by synthesis (log-uniform sampling).
#
# IMPORTANT: the *tier classifier* (`m_tier`/`k_tier` above) is
# unbounded-above -- anything >512 is "Large"/"LargeK". These ranges
# are ONLY the SAMPLING bounds. They MUST cover the real workload
# operating range, or the trained model is forced to extrapolate on
# real GEMMs it never saw. Real-world workload traces reach "Large"
# M/N and K values far above the prior 8192/16384 ceilings, which left
# the model blind to the top decades of the Large/LargeK cells. The
# per-matrix-bytes + FLOPs feasibility caps in lib/shapes.py keep the
# (now huge) corner of the box benchable: the realistic shapes here
# are tall-skinny / short-fat (one big axis, others small), and the
# joint all-axes-huge cube is filtered out.
TIER_RANGES_MN = {
    "Tiny": (1, 32),
    "Small": (33, 128),
    "Mid": (129, 512),
    "Large": (513, 524288),  # was 8192; 2**19, sized to cover the
    # M/N range of the OOB workload envelope
}
TIER_RANGES_K = {
    "TinyK": (1, 32),
    "MidK": (33, 512),
    "LargeK": (513, 131072),  # was 16384; 2**17, sized to cover the
    # K range of the OOB workload envelope
}
# Batched values, log-spaced over the range REAL workloads + the tuning
# library actually use. Evidence: the gfx942 GridBased BBS library tunes B
# up to 4096, and real-world workload traces use large B too -- paired
# with small dims (big-B => small-dims). The old 2..16 list made large B
# (>=512) a far out-of-distribution input the MLP cannot extrapolate
# (log2_b is smooth, but batch_k_iters_per_tile is LINEAR in B), so the
# batched cells were effectively unservable. synthesize_cell_shapes now
# couples B to the sampled dims (picks from the feasible subset that keeps
# the shape benchable), so large B only attaches to small dims -- exactly
# the real envelope -- and the per_matrix_bytes_feasible caps (working set
# <=16 GiB, max dim <=8192) remain the hard backstop.
BANY_VALUES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# Macro-tile boundaries we bias toward in MT-aligned synthesis.
MT_VALUES = (16, 32, 64, 128, 256)


def why_unpopulated(cell: str) -> str:
    """Return a human-readable reason why a cell would be empty in a
    typical OOB run -- used by stages that report coverage gaps."""
    mt, nt, kt, bt = cell.split("|")
    if bt == "Bany" and ("Large" in (mt, nt) or kt == "LargeK"):
        return (
            "is_rotating512_safe filter: B>1 with "
            "max(M,N)>256 OR K>512 is excluded for bench OOM safety"
        )
    return "no shape sampled (not in this round's pool)"
