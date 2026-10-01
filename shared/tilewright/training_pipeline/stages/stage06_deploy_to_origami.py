# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage06_deploy_to_origami -- pack the trained per-cell models into
the GRID-aware ML recommender weights binary and drop it into a
target rocm-libraries checkout.

(Renumbered from stage07 to stage06 in an earlier stage renaming;
the OLD validate-and-decide stage06 became stage04b -- i.e. it
moved to "stage 4.5" because it logically sits BETWEEN the round's
data-collection block and stage05's retrain.)

ASSUMPTIONS about the target checkout (`--rocm-libraries-to-deploy`):

  * Kernels are ALREADY built (the per-arch `.dat` files exist under
    `projects/hipblaslt/build/release/Tensile/library/<arch>/`).
    stage06 does NOT rebuild kernels.

  * The grid-aware C++ runtime is ALREADY checked in -- specifically
    `shared/origami/src/origami/ml_recommender.cpp` and
    `shared/origami/include/origami/ml_recommender.hpp` are the
    grid-aware files (the cluster-based versions on the prior branch
    have been REPLACED with the grid-aware drop-in; stage06 does NOT
    touch C++ source on the deploy side).

  * The hipBLASLt dispatch hook is ALREADY in place:
    `projects/hipblaslt/tensilelite/include/Tensile/PredictionLibrary
    .hpp` already calls `origami::rank_configs(...)` with
    `prediction_mode = ml_recommender` whenever `Debug::Instance()
    .useMLRecommender()` returns true (driven by
    `TENSILE_USE_TILEWRIGHT=1`). stage06 does NOT patch this file.

stage06's job is just:

      (1) Pack each trained cell's MLP weights + the cumulative split
      tree into a single `tilewright_weights.bin` (v6 format --
      magic "MLREC_v1"; see the spec in
      <repo>/shared/tilewright/src/tilewright/model.cpp).

  (2) Optionally drop the .bin into
      `<repo>/shared/tilewright/backends/hipblaslt/weights/` (the productized
      install path the tilewright loader's `ensure_weights()` discovers; a
      transitional copy is also mirrored into the legacy `data/` dir).

  (3) Optionally rebuild the parts of the deploy repo that
      consume the new weights:
        * `cmake --build <repo>/shared/origami/build --parallel`
          -> relinks libOrigami (which embeds the new ml_recommender
             .cpp) AND the origami Python binding `.so`.
        * `make -C <repo>/projects/hipblaslt/build/release/clients
          -j <jobs>`
          -> relinks hipblaslt-bench against the new libOrigami.
        * `cp shared/origami/build/python/src/origami/origami.cpython
          -*.so <venv_site_packages>/origami/`
          -> so stage08_validate_sel_eff's offline-eval Python path
             picks up the new library too. The destination is
             auto-detected from sys.

If `--rocm-libraries-to-deploy` is omitted, stage06 emits the .bin +
manifest under `<out-dir>/staged_for_deploy/` only and prints a
preview -- the operator can copy by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

import yaml  # noqa: E402

from lib import features as fs  # noqa: E402
from lib import grid as _grid  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402


# Schema version for the .bin + manifest. Bumping this number tells
# the runtime "I refuse to load older blobs"; the C++ side hard-checks
# the same magic ("MLREC_v1") and the feature-names hash.
WEIGHTS_BIN_MAGIC = b"MLREC_v1"
WEIGHTS_BIN_VERSION = 1
WEIGHTS_BIN_TRAILER = b"MLRECEND"
SCHEMA_VERSION = "pipeline-v3-deploy-2026-05-29-bf16-weights"

# Weight storage precision. v6 adds a 1-byte `weight_dtype` tag right
# after the endian marker:
#   0 = fp32  -- the matmul `.weight` matrices stored as float32
#               (same on-disk layout v5 used).
#   1 = bf16  -- the `.weight` matrices stored as bfloat16 (top 16
#               bits of the float32, round-to-nearest-even). Biases,
#               whitening means/stds, and temperature stay float32.
#
# bf16 halves the dominant byte cost (the weight matrices dominate
# the per-cell payload) for ~no accuracy cost: the C++ runtime
# promotes bf16 -> float32 at load (bits << 16) and computes in
# float32, and the Python parity path rounds the same way, so
# C++<->Python pick parity stays exact. Defaults to bf16.
#   2 = int8  -- the `.weight` matrices stored as per-tensor symmetric
#               int8: each matrix is prefixed by a float32 `scale`
#               (= max|w|/127) followed by its int8 values; the C++
#               runtime dequantizes `q * scale` -> float32 at load.
#               Biases / whitening / temperature stay float32. A
#               held-out sweep found picks IDENTICAL to bf16 (the
#               trained weights need few mantissa bits), so int8 halves
#               the dominant byte cost again (bf16 2B -> int8 1B/weight,
#               + one float32 scale per matrix).
#   3 = int4  -- per-tensor symmetric int4: float32 `scale` (= max|w|/7)
#               then ceil(n/2) bytes, two signed 4-bit values per byte
#               (low nibble = even index, high nibble = odd index; each
#               stored as q+8 in [1,15] so 0 is unused/clean). The C++
#               runtime unpacks + dequantizes `q*scale`. ~2x smaller than
#               int8, at a small sel_eff cost (use only when size is
#               the priority).
WEIGHT_DTYPE_FP32 = 0
WEIGHT_DTYPE_BF16 = 1
WEIGHT_DTYPE_INT8 = 2
WEIGHT_DTYPE_INT4 = 3
_WEIGHT_DTYPE_BY_NAME = {
    "fp32": WEIGHT_DTYPE_FP32,
    "bf16": WEIGHT_DTYPE_BF16,
    "int8": WEIGHT_DTYPE_INT8,
    "int4": WEIGHT_DTYPE_INT4,
}


# Order in which we PACK the 14 MLP tensors per cell. MUST match
# `score_one()` in shared/origami/src/origami/ml_recommender.cpp.
_WEIGHT_ORDER: Tuple[str, ...] = (
    "q_proj.0.weight",
    "q_proj.0.bias",
    "q_proj.2.weight",
    "q_proj.2.bias",
    "q_proj.4.weight",
    "q_proj.4.bias",
    "i_proj.0.weight",
    "i_proj.0.bias",
    "i_proj.2.weight",
    "i_proj.2.bias",
    "inter_mlp.0.weight",
    "inter_mlp.0.bias",
    "inter_mlp.2.weight",
    "inter_mlp.2.bias",
)

# Repo-relative paths the deploy targets. One-line update if rocm-
# libraries ever moves them.
_REPO_WEIGHTS_DIR = "shared/tilewright/backends/hipblaslt/weights"
_REPO_DATA_FILE = "tilewright_weights.bin"  # legacy global fallback name
_REPO_MANIFEST = "ml_recommender.manifest.json"
_REPO_SUMMARY = "ml_recommender.summary.txt"
_REPO_INSTALL_MD = "shared/origami/ML_RECOMMENDER_INSTALL.md"
# Per-library index consumed by the tilewright runtime: each line maps a Tensile
# library-logic stem (== the basename the runtime sees as ctx->filename, minus
# extension) to its tilewright weights .bin. One model serves every logic file of
# the same dtype/layout, so multiple stems may map to the same .bin.
_REPO_INDEX_FILE = "tilewright_index"


def _match_library_stems(repo, arch, hbl):
    """Return the deploy repo's Tensile library .dat stems (basename minus
    '.dat') whose (a_type, d_type, compute_type, transA, transB) match this
    config's hipblaslt dtype/layout -- i.e. the logic files the deployed model
    serves. These stems are the tilewright_index keys (== runtime ctx->filename
    basenames). Empty if the library dir is absent.

    The dtype/layout 5-tuple ALONE is not enough on gfx1250: the MX and
    non-MX F8 TN libraries decode identically, so a config trained on the
    MXFP8 pool would also claim the non-MX stem and the deployed .bin
    would be routed to a library whose kernels the model never saw. When
    the config pins `hipblaslt.scaleA`, we additionally require the
    library's own scale mode (`lib.dat.parse_scale_mode`) to match.
    Configs with no `scaleA` key keep the old, unfiltered behavior."""
    from lib.dat import parse_dtype_layout, parse_scale_mode  # noqa: E402

    want = (
        str(hbl.get("a_type")),
        str(hbl.get("d_type")),
        str(hbl.get("compute_type")),
        str(hbl.get("transA")),
        str(hbl.get("transB")),
    )
    want_scale = int(hbl["scaleA"]) if hbl.get("scaleA") is not None else None
    lib_root = Path(repo) / "projects/hipblaslt/build/release/Tensile/library"
    lib_dir = lib_root / arch
    if not lib_dir.is_dir():
        # An ASIC revision (gfx1250v0) and its bare compiler target (gfx1250)
        # ship the SAME .dat stems -- the filenames end in the target token for
        # both -- so a deploy checkout built for only one of them still yields
        # the correct tilewright_index keys. Without this fallback, deploying a
        # gfx1250v0 model into a repo whose build tree has only `gfx1250/`
        # silently matches NO stems and the .bin is written under the legacy
        # global name, which the runtime's per-library index never looks up:
        # the model ships but is never loaded.
        from lib.arch import canonical_arch  # noqa: E402

        alt = lib_root / canonical_arch(arch)
        if alt.is_dir() and alt != lib_dir:
            lib_dir = alt
    stems = []
    if lib_dir.is_dir():
        # Tensile logic ships either plain `.dat` or compressed `.dat.zlib`
        # (the productized build emits compressed; the gfx1250v0 tree ships
        # ONLY the latter). Match both and strip the full suffix so the stem
        # == the runtime ctx->filename basename (the tilewright_index key).
        # Check `.dat.zlib` first so a `.dat` glob can't leave a dangling
        # `.zlib` on the stem.
        seen = set()
        for suffix in (".dat.zlib", ".dat"):
            for p in sorted(lib_dir.glob(f"*Contraction_l_*{suffix}")):
                key = parse_dtype_layout(p.name)
                if key is None or tuple(str(x) for x in key) != want:
                    continue
                # gfx1250: MX and non-MX F8 TN decode to the same 5-tuple,
                # so a pinned scaleA must also match the library's own mode.
                if want_scale is not None and parse_scale_mode(p.name) != want_scale:
                    continue
                stem = p.name[: -len(suffix)]
                if stem not in seen:
                    seen.add(stem)
                    stems.append(stem)
    return stems


def _collect_model_sol_idxs(train_dir, prior_round_dirs):
    """Distinct global solution indices the deployed model was trained on,
    read from the enriched stage04 CSVs (the FINAL round is enough -- every
    chunk of a given config's library references the same liblogic yaml).
    Used only to resolve the .bin filename; empty on miss."""
    import csv as _csv

    cand_dirs = []
    if train_dir is not None:
        cand_dirs.append(Path(train_dir).parent / "stage04")
    cand_dirs += [Path(p) / "stage04" for p in (prior_round_dirs or [])]
    for d in cand_dirs:
        if not d.is_dir():
            continue
        idxs = set()
        for cf in sorted(d.glob("*.csv")):
            try:
                with cf.open() as fh:
                    for row in _csv.DictReader(fh):
                        v = row.get("sol_idx_global")
                        if v:
                            idxs.add(int(v))
            except Exception:
                continue
        if idxs:
            return idxs
    return set()


def _liblogic_stem_from_matchtable(repo, sol_idxs, want_layout=None):
    """Map global solution indices -> the dominant Tensile library-logic yaml
    basename (sans .yaml) via the deploy repo's MatchTable.yaml. This is the
    'actual liblogic name the model runs on'. Returns (stem, n_distinct_yaml)
    or None if the table is absent / no index resolves. Parsed with a line
    regex (the table is large -- one 2-line flow entry per solution index;
    full yaml.load is slow).

    `want_layout` = (a_type, d_type, compute_type, transA, transB) of THIS
    config; when given, ONLY liblogic yamls whose own dtype/layout matches are
    counted. A kernel's global index can appear in several liblogic yamls
    (kernels are shared across transposes), so without this filter a T/T model
    can resolve to the T/N or N/T liblogic name -- whose .bin filename then
    COLLIDES with that other config's deploy and silently overwrites it. When
    nothing matches the requested layout we return None so the caller falls
    back to the unambiguous runtime stem (lib_stems[0])."""
    mt = Path(repo) / "projects/hipblaslt/build/release/device-library/MatchTable.yaml"
    if not mt.exists() or not sol_idxs:
        return None
    import re as _re, os as _os, collections as _c

    want_idxs = {int(i) for i in sol_idxs}
    rx = _re.compile(r"^(\d+):\s*\[(.+\.yaml)\s*,?\s*$")
    found = {}
    with mt.open() as f:
        for line in f:
            m = rx.match(line)
            if m and int(m.group(1)) in want_idxs:
                found[int(m.group(1))] = _os.path.basename(m.group(2))
    if not found:
        return None
    vals = list(found.values())
    if want_layout is not None:
        # Keep only liblogic yamls whose TRANSPOSE matches this config. A kernel
        # global index can map to liblogic yamls of SEVERAL transposes (kernels
        # are shared), so picking the wrong one collides with that config's .bin
        # and overwrites it (the T/T-named-as-T/N bug). The liblogic yaml name
        # embeds the layout as `Cijk_<Alayout>_<Blayout>_` (e.g. Cijk_Alik_Bjlk
        # = T/T); map it via DAT_LAYOUT_MAP and compare to (transA, transB).
        from lib.dat import DAT_LAYOUT_MAP  # noqa: E402

        want_t = (str(want_layout[3]), str(want_layout[4]))

        def _yaml_transpose(name):
            m = re.search(r"Cijk_(A\w+?_B\w+?)_", name)
            lay = DAT_LAYOUT_MAP.get(m.group(1)) if m else None
            return (str(lay[0]), str(lay[1])) if lay else None

        matched = [v for v in vals if _yaml_transpose(v) == want_t]
        if not matched:
            return None
        vals = matched
    cnt = _c.Counter(vals)
    stem = cnt.most_common(1)[0][0]
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    return stem, len(cnt)


# UserArgs liblogic dtype token <- (a_type, d_type, compute_type). The Origami
# UserArgs yaml is named `<arch>_Cijk_<layout>_<dtype_token>_..._UserArgs`; the
# token is the human-readable dtype tag (NOT the TensileLibrary `_BB_BB_` form).
_USERARGS_DTYPE_TOKEN = {
    ("bf16_r", "bf16_r", "c_f32_r"): "BBS",
    ("f32_r", "f32_r", "c_xf32_r"): "S_MX",
    ("f8_r", "bf16_r", "c_f32_r"): "F8BS",
    ("f16_r", "f16_r", "c_f32_r"): "HHS",
}


def _origami_userargs_stem(repo, arch, hbl):
    """Deterministic .bin basename: the Origami UserArgs liblogic yaml for THIS
    config's (dtype, transpose). Globs the deploy repo's Origami logic dir and
    matches `<layout_token>_<dtype_token>`; returns the yaml basename (sans
    .yaml) or None.

    Preferred over MatchTable sol-idx resolution, which is fragile: a kernel's
    global index can appear in several transposes' liblogic, so that path can
    name a T/T model after the T/N or N/T yaml -> filename COLLISION that
    overwrites the other config's .bin. This lookup is collision-free by
    construction (exactly one yaml per dtype+transpose)."""
    from lib.dat import DAT_LAYOUT_MAP  # noqa: E402

    rev = {v: k for k, v in DAT_LAYOUT_MAP.items()}
    layout = rev.get((str(hbl.get("transA")), str(hbl.get("transB"))))
    dt = _USERARGS_DTYPE_TOKEN.get(
        (str(hbl.get("a_type")), str(hbl.get("d_type")), str(hbl.get("compute_type")))
    )
    if not layout or not dt:
        return None
    od = (
        Path(repo)
        / "projects/hipblaslt/library/src/amd_detail/rocblaslt"
        / "src/Tensile/Logic/asm_full"
        / arch
        / arch
        / "Origami"
    )
    if not od.is_dir():
        return None
    cands = sorted(od.glob(f"{arch}_Cijk_{layout}_{dt}_*UserArgs.yaml"))
    if len(cands) == 1:
        return cands[0].name[: -len(".yaml")]
    return None


_SEG_RE = re.compile(r"^([MNKB])(<=|>)(-?\d+)$")


def _splits_from_cell_labels(labels) -> Dict[str, "sc.SplitRule"]:
    """Reconstruct the SplitRule tree from the DEPLOYED cell labels alone, so
    the routing table is self-contained in the .bin and never depends on
    prior-round decisions.json being passed at deploy time.

    Each label is `base#seg1#seg2...` with seg = `<axis><op><threshold>`
    (op in {<=, >}). Every prefix boundary implies the split rule for that
    internal node: parent splits on `axis` at `threshold` into
    `parent#axis<=threshold` (lo) and `parent#axis>threshold` (hi).

    Background: with empty/mismatched prior_round_dirs at deploy time,
    `load_cumulative_split_tree() == {}` -> n_splits=0 in the .bin, so the
    runtime cannot descend the base grid to any leaf and collapses to a
    single kernel. The deployed cells ARE the source of truth for what the
    runtime must be able to reach, so we derive the rules from them."""
    rules: Dict[str, "sc.SplitRule"] = {}
    for lbl in labels:
        if "#" not in lbl:
            continue
        base = sc.base_cell(lbl)
        parent = base
        for seg in lbl[len(base) :].split("#")[1:]:
            m = _SEG_RE.match(seg)
            if not m:
                continue
            axis, _op, thr = m.group(1), m.group(2), int(m.group(3))
            lo = f"{parent}#{axis}<={thr}"
            hi = f"{parent}#{axis}>{thr}"
            ex = rules.get(parent)
            if ex is None:
                rules[parent] = sc.SplitRule(
                    cell=parent, axis=axis, threshold=thr, lo_label=lo, hi_label=hi
                )
            elif ex.axis != axis or int(ex.threshold) != thr:
                # Each node splits on exactly one axis/threshold; disagreement
                # means corrupt labels.
                raise ValueError(
                    f"inconsistent split for {parent}: "
                    f"{ex.axis}{ex.threshold} vs {axis}{thr}"
                )
            parent = f"{parent}#{seg}"
    return rules


def _merge_index(index_path, stems, bin_filename):
    """Read the existing tilewright_index (stem<TAB>weights per line), set/replace
    the entries for `stems` -> `bin_filename` (preserving every OTHER config's
    entries), and rewrite atomically."""
    from lib import fs as _fsutil  # noqa: E402

    entries = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                entries[parts[0]] = parts[1]
    for st in stems:
        entries[st] = bin_filename
    # NOTE: this file ships in the public rocm-libraries repo -- keep the
    # header neutral (no internal tool/pipeline names, no perf numbers).
    body = [
        "# tilewright per-library weights index: <logic_stem>\\t<weights.bin>",
        "# auto-generated by the training pipeline's deploy stage",
    ]
    for st in sorted(entries):
        body.append(f"{st}\t{entries[st]}")
    _fsutil.atomic_write_bytes(index_path, ("\n".join(body) + "\n").encode())


# Source paths the BUILD system already knows about.
_REPO_ORIGAMI_BUILD = "shared/origami/build"
_REPO_HIPBLASLT_CLIENTS = "projects/hipblaslt/build/release/clients"
_REPO_PYBINDING_GLOB = "shared/origami/build/python/src/origami/origami.cpython-*.so"


# ── binary writer ──────────────────────────────────────────────────


def _tensor_to_f32(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32))


def _temperature_scalar(state_dict: Dict[str, Any]) -> float:
    t = state_dict.get("temperature")
    if t is None:
        return 1.0
    a = _tensor_to_f32(t).reshape(-1)
    return float(a[0]) if a.size > 0 else 1.0


def _tensor_to_bf16_bytes(t: Any) -> bytes:
    """Pack a tensor as bfloat16 (uint16 = top 16 bits of the float32,
    round-to-nearest-even). MUST use torch so the rounding matches the
    Python parity path (`torch.Tensor.to(torch.bfloat16)`) AND the C++
    read path (`uint16 << 16` reinterpret). numpy has no bfloat16, so
    we route through torch here even though the rest of the writer is
    numpy-based."""
    import torch  # local import; stage06 already requires torch

    a32 = _tensor_to_f32(t)  # contiguous float32
    bf16 = torch.from_numpy(a32).to(torch.bfloat16)
    u16 = bf16.view(torch.uint16).numpy()
    return np.ascontiguousarray(u16, dtype=np.uint16).tobytes(order="C")


def _tensor_to_int8_bytes(t: Any) -> bytes:
    """Pack a `.weight` matrix as per-tensor symmetric int8. Layout:
    <float32 scale><int8 values, row-major>, where scale = max|w|/127 and
    q = round(w/scale) clamped to [-127, 127]. The C++ read path
    dequantizes `q * scale` -> float32; the Python parity path applies the
    SAME quant so C++<->Python picks stay identical. Verified pick-for-pick
    equal to bf16 on a held-out OOD set (the trained weights only need a few
    mantissa bits)."""
    a32 = _tensor_to_f32(t).reshape(-1)
    amax = float(np.abs(a32).max()) if a32.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.round(a32 / scale), -127, 127).astype(np.int8)
    return struct.pack("<f", scale) + q.tobytes(order="C")


def _tensor_to_int4_bytes(t: Any) -> bytes:
    """Pack a `.weight` matrix as per-tensor symmetric int4. Layout:
    <float32 scale><ceil(n/2) bytes>, scale = max|w|/7, q = round(w/scale)
    clamped to [-7,7], stored as (q+8) nibble in [1,15]; even index -> low
    nibble, odd index -> high nibble. C++ unpacks `(nibble-8)*scale`."""
    a32 = _tensor_to_f32(t).reshape(-1)
    amax = float(np.abs(a32).max()) if a32.size else 0.0
    scale = (amax / 7.0) if amax > 0 else 1.0
    q = np.clip(np.round(a32 / scale), -7, 7).astype(np.int32) + 8  # [1,15]
    n = q.size
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    packed[: (n + 1) // 2] = q[0::2].astype(np.uint8)  # low nibble
    if n > 1:
        hi = q[1::2].astype(np.uint8)
        packed[: hi.size] |= hi << 4  # high nibble
    return struct.pack("<f", scale) + packed.tobytes(order="C")


def _pack_weights(
    state_dict: Dict[str, Any], weight_dtype: int = WEIGHT_DTYPE_FP32
) -> bytes:
    """Concatenate the 14 MLP tensors in `_WEIGHT_ORDER`, raw row-major.

    `.weight` matrices use `weight_dtype` (fp32 or bf16); `.bias`
    vectors are ALWAYS float32 (they're tiny -- one per output neuron
    -- and the additive bias term is precision-sensitive, so there's
    no size win and a real accuracy cost to quantizing them).

    Caller has already written the per-cell header (dims + temperature
    + whitening means/stds + smart_K list)."""
    chunks: List[bytes] = []
    for name in _WEIGHT_ORDER:
        if name not in state_dict:
            raise KeyError(f"missing tensor '{name}' in cell state_dict")
        if weight_dtype == WEIGHT_DTYPE_BF16 and name.endswith(".weight"):
            chunks.append(_tensor_to_bf16_bytes(state_dict[name]))
        elif weight_dtype == WEIGHT_DTYPE_INT8 and name.endswith(".weight"):
            chunks.append(_tensor_to_int8_bytes(state_dict[name]))
        elif weight_dtype == WEIGHT_DTYPE_INT4 and name.endswith(".weight"):
            chunks.append(_tensor_to_int4_bytes(state_dict[name]))
        else:
            chunks.append(_tensor_to_f32(state_dict[name]).tobytes(order="C"))
    return b"".join(chunks)


def _pack_floats(values: List[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes(order="C")


def _write_lenstr_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 0xFFFF:
        raise ValueError(f"label too long ({len(b)} bytes): {s!r}")
    return struct.pack("<H", len(b)) + b


def _write_lenstr_u32(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _emit_v6_bin(
    *,
    feature_names_hash: str,
    arch: str,
    q_dim: int,
    i_dim: int,
    x_dim: int,
    splits: List[sc.SplitRule],
    cells: List[Dict[str, Any]],
    weight_dtype: int = WEIGHT_DTYPE_BF16,
) -> bytes:
    """`cells` are dicts with keys
        label, embed_dim, hidden_dim, inter_hidden, temperature,
        q_mean, q_std, i_mean, i_std, x_mean, x_std, state_dict.

    `weight_dtype` selects fp32 vs bf16 for the `.weight` matrices
    (see WEIGHT_DTYPE_* constants); a 1-byte tag records it so the
    C++ reader knows how to decode the per-cell weights.
    """
    out: List[bytes] = []
    # Header.
    out.append(WEIGHTS_BIN_MAGIC)
    out.append(struct.pack("<I", WEIGHTS_BIN_VERSION))
    out.append(struct.pack("<I", 0x01020304))  # LE marker
    # v6: weight-dtype tag (1 byte) immediately after the endian
    # marker, before the variable-length feature hash.
    out.append(struct.pack("<B", int(weight_dtype)))
    out.append(_write_lenstr_u32(feature_names_hash))
    out.append(_write_lenstr_u32(arch))
    out.append(struct.pack("<III", q_dim, i_dim, x_dim))
    out.append(struct.pack("<II", len(cells), len(splits)))

    # Splits.
    for r in splits:
        out.append(_write_lenstr_u16(r.cell))
        if r.axis not in ("M", "N", "K"):
            raise ValueError(f"unsupported split axis: {r.axis!r}")
        out.append(struct.pack("<cc", r.axis.encode("ascii"), b"\x00"))
        out.append(struct.pack("<i", int(r.threshold)))
        out.append(_write_lenstr_u16(r.lo_label))
        out.append(_write_lenstr_u16(r.hi_label))

    # Cells.
    for c in cells:
        out.append(_write_lenstr_u16(c["label"]))
        out.append(
            struct.pack(
                "<III",
                int(c["embed_dim"]),
                int(c["hidden_dim"]),
                int(c["inter_hidden"]),
            )
        )
        out.append(struct.pack("<f", float(c["temperature"])))
        out.append(_pack_floats(list(c["q_mean"])))
        out.append(_pack_floats(list(c["q_std"])))
        out.append(_pack_floats(list(c["i_mean"])))
        out.append(_pack_floats(list(c["i_std"])))
        out.append(_pack_floats(list(c["x_mean"])))
        out.append(_pack_floats(list(c["x_std"])))
        # smart_K kernel-signature list (v5).
        sk_sigs = c.get("smart_k_signatures") or []
        out.append(struct.pack("<I", len(sk_sigs)))
        for sig in sk_sigs:
            t = tuple(int(v) for v in sig)
            if len(t) != 8:
                raise ValueError(f"smart_k signature must be 8-tuple, got {t!r}")
            out.append(struct.pack("<8i", *t))
        out.append(_pack_weights(c["state_dict"], weight_dtype))

    out.append(WEIGHTS_BIN_TRAILER)
    return b"".join(out)


# ── manifest + summary ─────────────────────────────────────────────


def _build_manifest(
    *,
    feature_names: Dict[str, List[str]],
    feature_names_hash: str,
    feature_version: str,
    arch: str,
    config_id: str,
    splits: List[sc.SplitRule],
    cells: List[Dict[str, Any]],
    weights_bin_size: int,
    weight_dtype: str = "bf16",
) -> Dict[str, Any]:
    cell_briefs: List[Dict[str, Any]] = []
    for c in cells:
        d: Dict[str, Any] = {
            "label": c["label"],
            "embed_dim": int(c["embed_dim"]),
            "hidden_dim": int(c["hidden_dim"]),
            "inter_hidden": int(c["inter_hidden"]),
            "best_sel_eff": c.get("best_sel_eff"),
            "best_epoch": c.get("best_epoch"),
        }
        if sc.is_subcell(c["label"]):
            d["base_cell"] = sc.base_cell(c["label"])
            d["split_depth"] = sc.split_depth(c["label"])
        cell_briefs.append(d)
    return {
        "schema_version": SCHEMA_VERSION,
        "weights_bin_magic": WEIGHTS_BIN_MAGIC.decode("ascii"),
        "weights_bin_version": WEIGHTS_BIN_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_id": config_id,
        "arch": arch,
        "feature_version": feature_version,
        "feature_names_hash": feature_names_hash,
        "feature_names": feature_names,
        "feature_dims": {
            "q": len(feature_names["query"]),
            "i": len(feature_names["item"]),
            "x": len(feature_names["interaction"]),
            "total": len(feature_names["query"])
            + len(feature_names["item"])
            + len(feature_names["interaction"]),
        },
        "grid": {
            "tier_ranges_mn": {k: list(v) for k, v in _grid.TIER_RANGES_MN.items()},
            "tier_ranges_k": {k: list(v) for k, v in _grid.TIER_RANGES_K.items()},
            "b_tiers": list(_grid.B_TIERS),
            "bany_values": list(_grid.BANY_VALUES),
        },
        "splits": [r.to_json() for r in splits],
        "cells": cell_briefs,
        "weights_bin_size": weights_bin_size,
        "weight_dtype": weight_dtype,
        "weight_param_order": list(_WEIGHT_ORDER),
    }


def _format_summary(manifest: Dict[str, Any], tree: Dict[str, sc.SplitRule]) -> str:
    n_base = sum(1 for c in manifest["cells"] if "#" not in c["label"])
    n_sub = sum(1 for c in manifest["cells"] if "#" in c["label"])
    lines = [
        "pipeline_v3 (grid) deploy summary",
        f"  config_id            : {manifest['config_id']}",
        f"  arch                 : {manifest['arch']}",
        f"  feature_version      : {manifest['feature_version']}",
        f"  feature_names_hash   : {manifest['feature_names_hash']}",
        f"  feature dims Q/I/X   : "
        f"{manifest['feature_dims']['q']} / "
        f"{manifest['feature_dims']['i']} / "
        f"{manifest['feature_dims']['x']}",
        f"  cells total          : {len(manifest['cells'])} "
        f"({n_base} base + {n_sub} sub-cells)",
        f"  weights .bin size    : {manifest['weights_bin_size']:,} bytes",
        f"  weights .bin magic   : {manifest['weights_bin_magic']} "
        f"(v{manifest['weights_bin_version']})",
        "  split tree           :",
    ]
    if tree:
        for line in sc.format_split_tree(tree).splitlines():
            lines.append(f"    {line}")
    else:
        lines.append("    (no splits decided yet)")
    return "\n".join(lines) + "\n"


def _install_md(manifest: Dict[str, Any]) -> str:
    return f"""# ML Recommender install (grid-aware, schema {SCHEMA_VERSION})

Auto-generated by
`shared/tilewright/training_pipeline/stages/stage06_deploy_to_origami.py`
on {manifest['generated_at']}.

## What was written to this rocm-libraries checkout

```
{_REPO_WEIGHTS_DIR}/{_REPO_DATA_FILE}     ({manifest['weights_bin_size']:,} bytes)
```

ONLY the weights `.bin` is written into this repo. The manifest
(`{_REPO_MANIFEST}`) and human-readable summary (`{_REPO_SUMMARY}`)
contain CONFIDENTIAL per-cell performance numbers and are kept ONLY
in the pipeline-side run output (`<out-dir>/staged_for_deploy/`), never
in this checkout.

The grid-aware C++ runtime is committed to this repo in the standalone,
framework-agnostic `tilewright` library at `shared/tilewright/src/tilewright/model.cpp`
(public API in `shared/tilewright/include/tilewright/`). origami consumes it through
the thin adapter `shared/origami/src/origami/ml_recommender.cpp`. It replaces
the prior cluster-based runtime in this branch. stage06 does NOT touch C++
source; the source is shipped via git, the model parameters via this .bin.

The dispatch hook in
`projects/hipblaslt/tensilelite/include/Tensile/PredictionLibrary.hpp`
already routes to `origami::rank_configs(...)` whenever
`Debug::Instance().useMLRecommender()` returns true (env var
`TENSILE_USE_TILEWRIGHT=1`). origami's own dispatch then forwards to
`origami::ml_recommender::rank_configs` (the new grid path).

## How to enable / disable at runtime

```
export TENSILE_PREDICTION_LIB=1       # always: route through the
                                       # prediction library
export TENSILE_USE_TILEWRIGHT=1       # ON  -- score with the model
unset  TENSILE_USE_TILEWRIGHT         # OFF -- plain Origami
```

Selection time prints with `TENSILE_DB=0x10000`.

## Configuration

- {len(manifest['cells'])} cells trained (
  {sum(1 for c in manifest['cells'] if '#' not in c['label'])} base +
  {sum(1 for c in manifest['cells'] if '#' in c['label'])} sub-cells
)
- {len(manifest['splits'])} split rules
- Feature catalog version : `{manifest['feature_version']}`
- Feature names hash      : `{manifest['feature_names_hash']}`
- Q / I / X dims          : {manifest['feature_dims']['q']} /
  {manifest['feature_dims']['i']} / {manifest['feature_dims']['x']}

## Validating after install

```
python shared/tilewright/training_pipeline/stages/stage07_optional_validate_selection_time.py ...
python shared/tilewright/training_pipeline/stages/stage08_optional_validate_selection_efficiency.py ...
```

## Re-deploying

stage06 is idempotent against the .bin (and only the .bin):
re-running with `--apply-patch` simply OVERWRITES
`{_REPO_WEIGHTS_DIR}/{_REPO_DATA_FILE}` and (if `--build-after-apply`)
relinks libOrigami + hipblaslt-bench + the Python binding.
"""


# ── build helpers ──────────────────────────────────────────────────


def _tee_subprocess(
    cmd: List[str], log_path: Path, cwd: Optional[Path] = None, quiet: bool = False
) -> int:
    """Run `cmd`, tee output to log_path AND stdout (unless quiet).
    The log file is opened LINE-BUFFERED so `tail -f` lags by at most
    one line; we also `flush()` explicitly to cover writes that may
    arrive without a trailing newline (e.g. ANSI progress carriage
    returns from cmake)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not quiet:
        ui.info(
            "run",
            " ".join(str(c) for c in cmd)
            + (f"   (cwd={cwd})" if cwd is not None else ""),
        )
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=str(cwd) if cwd is not None else None,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    el = time.time() - t0
    if rc == 0:
        if not quiet:
            ui.ok("run", f"completed in {ui.fmt_dur(el)}")
    else:
        ui.err("run", f"FAILED (rc={rc}) after {ui.fmt_dur(el)}; " f"see {log_path}")
    return rc


def _host_has_avx512bf16() -> bool:
    """True if THIS host's CPU exposes AVX512-BF16, so the tilewright scorer's
    bf16/AVX512 fast path can actually run. We build AND deploy on the same
    host, so detecting the build host here is safe (unlike a shipped binary,
    where baking in AVX512 would SIGILL on a non-AVX512 deploy host -- which is
    exactly why tilewright keeps TILEWRIGHT_ENABLE_AVX512 OFF by default)."""
    try:
        with open("/proc/cpuinfo") as f:
            return "avx512_bf16" in f.read()
    except Exception:
        return False


def _enable_tilewright_avx512(
    hbl_build_root: Path, log: Path, jobs: Optional[int], quiet: bool
) -> None:
    """When the build host supports AVX512-BF16, turn on the tilewright
    bf16/AVX512 scorer paths in the embedded (hipBLASLt) tilewright build and
    rebuild the lib, so the relink below picks up the faster scorer instead
    of the portable scalar fp32 fallback. No-op if unsupported or already on.
    Best-effort: never fails the deploy."""
    if not _host_has_avx512bf16():
        return
    cache = hbl_build_root / "CMakeCache.txt"
    try:
        if "TILEWRIGHT_ENABLE_AVX512:BOOL=ON" in cache.read_text():
            return  # already enabled
    except Exception:
        pass
    if not quiet:
        ui.info(
            "build",
            "host has AVX512-BF16 -> enabling tilewright "
            "bf16/AVX512 scorer (TILEWRIGHT_ENABLE_AVX512=ON) for this "
            "self-built runtime",
        )
    # Reconfigure the hipBLASLt build cache (tilewright's option() honors the
    # -D cache value), then rebuild just the tilewright lib; the bench/lib
    # relink in (b) propagates it.
    _tee_subprocess(
        ["cmake", str(hbl_build_root), "-DTILEWRIGHT_ENABLE_AVX512=ON"],
        log,
        quiet=quiet,
    )
    bld = [
        "cmake",
        "--build",
        str(hbl_build_root),
        "--target",
        "tilewright",
        "--parallel",
    ]
    if jobs is not None:
        bld.append(str(jobs))
    _tee_subprocess(bld, log, quiet=quiet)


def _build_after_apply(
    repo_root: Path, out_dir: Path, jobs: Optional[int], quiet: bool
) -> int:
    """Drop-in build: assumes kernels are already built. We only:
    (a) cmake --build shared/origami/build  -- relinks libOrigami
        (which now embeds the new ml_recommender.cpp) plus the
        origami Python binding `.so`.
    (b) make -C projects/hipblaslt/build/release/clients   -- relinks
        hipblaslt-bench against the new libOrigami.
    (c) cp the Python binding into the venv's site-packages so
        stage08_validate_sel_eff's offline-eval Python path uses it.
    """
    origami_build = repo_root / _REPO_ORIGAMI_BUILD
    clients_dir = repo_root / _REPO_HIPBLASLT_CLIENTS
    log = out_dir / "build.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # (a) origami (libOrigami + Python binding).
    if not origami_build.exists():
        ui.err(
            "build",
            f"missing {origami_build}; configure with "
            f"`cmake -S shared/origami -B shared/origami/build "
            f"-DORIGAMI_ENABLE_PYTHON=ON` before retrying.",
        )
        return 1
    cmake_cmd = ["cmake", "--build", str(origami_build), "--parallel"]
    if jobs is not None:
        cmake_cmd.append(str(jobs))
    rc = _tee_subprocess(cmake_cmd, log, quiet=quiet)
    if rc != 0:
        return rc
    # The default `all` target builds libOrigami but NOT the Python module
    # (`_pyorigami`), so build it explicitly -- stage08's offline-eval Python
    # path needs it. Best-effort: older origami checkouts may lack the target,
    # so a failure here doesn't fail the deploy (the warning in (c) covers it).
    py_cmd = [
        "cmake",
        "--build",
        str(origami_build),
        "--target",
        "_pyorigami",
        "--parallel",
    ]
    if jobs is not None:
        py_cmd.append(str(jobs))
    _tee_subprocess(py_cmd, log, quiet=quiet)

    # (a2) Enable the AVX512/bf16 scorer in the embedded tilewright when this
    # host supports it (rebuilds libtilewright; the relink below propagates it
    # into libhipblaslt.so + hipblaslt-bench).
    _enable_tilewright_avx512(clients_dir.parent, log, jobs, quiet)

    # (b) hipblaslt-bench relink (kernels already built; just `make`).
    if not clients_dir.exists():
        ui.err(
            "build",
            f"missing {clients_dir}; kernels must be built "
            f"first via `./install.sh -c` (one-time).",
        )
        return 1
    make_cmd = ["make"]
    if jobs is not None:
        make_cmd += ["-j", str(jobs)]
    else:
        # Use a sane default: half the CPUs (the build is small).
        try:
            ncpu = os.cpu_count() or 4
            make_cmd += ["-j", str(max(1, ncpu // 2))]
        except Exception:
            pass
    rc = _tee_subprocess(make_cmd, log, cwd=clients_dir, quiet=quiet)
    if rc != 0:
        return rc

    # (b2) Co-locate the just-deployed weights with the runtime's Tensile
    # library. The weights ship at weights/<arch>/<arch>/ but the loader reads
    # them from Tensile/library/<arch>/ (next to the logic); the hipBLASLt
    # `tilewright_colocate` custom target does that copy. `make -C clients`
    # above does NOT run it (it's a top-level target), so build it explicitly
    # -- otherwise a freshly-deployed .bin never reaches the runtime. Best-
    # effort: absent on older checkouts, so a failure here doesn't fail deploy.
    hbl_build_root = clients_dir.parent
    _tee_subprocess(
        ["cmake", "--build", str(hbl_build_root), "--target", "tilewright_colocate"],
        log,
        quiet=quiet,
    )

    # (c) Python binding -> venv. The `_pyorigami` target emits the module into
    # the SOURCE tree (shared/origami/python/src/origami/), not the build dir,
    # so look there first (then fall back to the build dir for older layouts).
    so_globs = list(
        (repo_root / "shared/origami/python/src/origami").glob("origami.cpython-*.so")
    ) or list(origami_build.glob("python/src/origami/origami.cpython-*.so"))
    if not so_globs:
        ui.warn(
            "build",
            f"no Python binding `.so` found under "
            f"{repo_root}/shared/origami/python/src/origami/ -- did the "
            f"cmake build skip ORIGAMI_ENABLE_PYTHON? "
            f"stage08_validate_sel_eff will use whatever origami is "
            f"currently importable.",
        )
    else:
        so_path = so_globs[0]
        # Find the import site-packages dir. `sysconfig.get_paths()["purelib"]`
        # works inside a venv too.
        purelib = Path(sysconfig.get_paths()["purelib"])
        origami_pkg = purelib / "origami"
        if origami_pkg.exists():
            dest = origami_pkg / so_path.name
            shutil.copy2(so_path, dest)
            if not quiet:
                ui.ok("install", f"copied {so_path.name} -> {dest}")
        else:
            if not quiet:
                ui.warn(
                    "install",
                    f"venv site-packages has no `origami/` dir at "
                    f"{origami_pkg}; install the package first or "
                    f"copy the .so manually. ({so_path})",
                )
    return 0


# ── stage entrypoint ───────────────────────────────────────────────


def deploy(
    *,
    train_dir: Path,
    out_dir: Path,
    config_yaml: Path,
    prior_round_dirs: Optional[List[Path]] = None,
    rocm_libraries_to_deploy: Optional[Path] = None,
    apply_patch: bool = False,
    build_after_apply: bool = False,
    build_jobs: Optional[int] = None,
    weight_dtype: str = "bf16",
    quiet: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_pt = train_dir / "models.pt"
    cells_json_path = train_dir / "cells.json"
    if not models_pt.exists() or not cells_json_path.exists():
        ui.err("err", f"missing models.pt or cells.json in {train_dir}")
        return 1
    if not _TORCH_OK:
        ui.err(
            "err",
            "torch is required to load models.pt; install via " "requirements.txt",
        )
        return 1

    with config_yaml.open() as f:
        cfg = yaml.safe_load(f)
    config_id = str(cfg.get("config_id", config_yaml.stem))
    arch = str(cfg.get("arch", "gfx950"))
    # Productized layout: per-library weights live co-located with the per-arch
    # Tensile library at weights/<arch>/<arch>/ (see the hipBLASLt CMake
    # `tilewright_colocate` target, which mirrors the library-logic layout and
    # ONLY copies from that nested dir into the runtime's Tensile/library/<arch>/
    # where the loader discovers them next to the logic). Writing into the flat
    # weights/ root is silently ignored by the runtime. `_arch_base` strips any
    # `:xnack+` feature suffix exactly like the CMake `REGEX REPLACE ":.*"`.
    _arch_base = arch.split(":")[0]
    weights_subdir = f"{_REPO_WEIGHTS_DIR}/{_arch_base}/{_arch_base}"

    # Per-library deploy: name the .bin after the Tensile library-logic file(s)
    # this config serves, and index each matching logic stem -> the .bin so
    # multiple dtype/layout models coexist (the runtime loads per-library via
    # the index). Requires the deploy repo to glob its library .dat stems;
    # staging-only falls back to the legacy single tilewright_weights.bin.
    hbl = cfg.get("hipblaslt", {}) or {}
    lib_stems = []
    bin_filename = _REPO_DATA_FILE
    if rocm_libraries_to_deploy is not None:
        # tilewright_index KEYS are the runtime ctx->filename stems (the compiled
        # library .dat basenames) -- these MUST match what the deployed runtime
        # sees, so they stay as-is.
        lib_stems = _match_library_stems(rocm_libraries_to_deploy, arch, hbl)
        if lib_stems:
            # .bin FILENAME follows the actual Tensile library-logic yaml the
            # model's kernels map to (resolved data-driven via MatchTable:
            # every kernel of a config's library resolves to one liblogic
            # yaml). Falls back to the runtime stem if the table/indices can't
            # be resolved, so deploy never breaks.
            bin_filename = lib_stems[0] + ".tilewright.bin"
            # Name the .bin after the canonical Origami UserArgs liblogic yaml
            # for this config's (dtype, transpose) -- deterministic and
            # collision-free (one yaml per dtype+transpose). Falls back to the
            # runtime stem if the yaml can't be located.
            _ua = _origami_userargs_stem(rocm_libraries_to_deploy, arch, hbl)
            if _ua is not None:
                bin_filename = _ua + ".tilewright.bin"
                if not quiet:
                    ui.info(
                        "deploy",
                        f"liblogic (Origami UserArgs): {_ua}; " f".bin={bin_filename}",
                    )
            elif not quiet:
                ui.warn(
                    "deploy",
                    "could not resolve Origami UserArgs yaml for "
                    "this dtype/layout; naming .bin after the runtime stem.",
                )
        elif not quiet:
            ui.warn(
                "deploy",
                "no matching library .dat stems found for this "
                "dtype/layout; falling back to the global "
                f"{_REPO_DATA_FILE} (no per-library index entry).",
            )

    if not quiet:
        ui.banner(f"stage06  deploy  config_id={config_id}  arch={arch}", ui.C.HEAD)
        ui.info("cfg", f"train_dir          : {train_dir}")
        ui.info("cfg", f"out_dir            : {out_dir}")
        if rocm_libraries_to_deploy is not None:
            ui.info("cfg", f"rocm_libraries     : {rocm_libraries_to_deploy}")
            ui.info("cfg", f"apply_patch        : {apply_patch}")
            if apply_patch:
                jb = "" if build_jobs is None else f"  (-j {build_jobs})"
                ui.info("cfg", f"build_after_apply  : " f"{build_after_apply}{jb}")
        if prior_round_dirs:
            ui.info(
                "cfg", f"prior_round_dirs   : " f"{[p.name for p in prior_round_dirs]}"
            )

    bundle = torch.load(str(models_pt), map_location="cpu", weights_only=False)
    with cells_json_path.open() as f:
        cells_json = json.load(f)

    q_names = list(bundle.get("q_names", fs.query_feature_names()))
    i_names = list(bundle.get("i_names", fs.item_feature_names()))
    x_names = list(bundle.get("x_names", fs.interaction_feature_names()))
    feature_version = str(
        bundle.get("feature_version")
        or cells_json.get("feature_version", fs.FEATURES_VERSION)
    )
    feature_names_hash = str(
        cells_json.get("feature_names_hash")
        or bundle.get("feature_names_hash")
        or fs.feature_names_hash()
    )

    # Sanity: dims must match the C++ side's compile-time expectation.
    if len(q_names) != fs.feature_count() - len(i_names) - len(x_names):
        # Best-effort soft check: the trained model's q/i/x split came
        # from the SAME features.py the C++ codegen was built against,
        # so a mismatch here means catalog drift.
        ui.warn(
            "dims",
            "feature dim mismatch vs current features.py "
            f"(q={len(q_names)} i={len(i_names)} x={len(x_names)} "
            f"-- features.feature_count()={fs.feature_count()})",
        )
    if feature_names_hash != fs.feature_names_hash():
        ui.warn(
            "hash",
            f"feature_names_hash trained={feature_names_hash} "
            f"current={fs.feature_names_hash()} -- catalog drift; "
            f"the runtime will REFUSE to load this .bin",
        )

    # Splits: reconstruct the routing tree from the DEPLOYED cell labels so
    # the .bin is self-contained (a deploy with empty/mismatched
    # prior_round_dirs would otherwise ship n_splits=0 -> the runtime can't
    # reach any leaf and collapses to a single kernel). The deployed cells are
    # authoritative for what the runtime must route to.
    _deployed_labels = list(bundle["models"].keys())
    tree = _splits_from_cell_labels(_deployed_labels)
    splits = [tree[k] for k in sorted(tree.keys())]
    if not quiet:
        _prior = sc.load_cumulative_split_tree(prior_round_dirs or [])
        ui.info(
            "splits",
            f"reconstructed {len(splits)} split rules from "
            f"{len(_deployed_labels)} deployed cells "
            f"(prior-decisions tree had {len(_prior)})",
        )

    # Assemble per-cell records. We iterate `models` (the bundle's
    # full dict) -- NOT `cells_json["cells"]` -- because cells.json
    # lists only the cells trained IN this round; every cell stage05
    # carried forward from a prior round is in models.pt but absent
    # from cells.json. Without iterating models directly, those
    # carry-forwarded cells would silently drop from the deploy
    # bundle (e.g. a late round might pack only the few cells it
    # retrained while many more leaves are alive).
    #
    # cells.json is still useful for OPTIONAL metadata
    # (`best_sel_eff`, `best_epoch`) on the cells trained this round;
    # we look those up by label and leave them None for carried-
    # forward cells where we don't have a fresh measurement.
    cells: List[Dict[str, Any]] = []
    models = bundle["models"]
    meta_by_label: Dict[str, Dict[str, Any]] = {
        e["label"]: e for e in (cells_json.get("cells") or [])
    }

    # Coverage check (O1): every leaf of the deployed split tree MUST resolve
    # to a trained cell -- otherwise a GEMM routed there has no matching model
    # (orphaned region) and the runtime silently falls back to analytical
    # Origami. Walk every leaf; flag any whose lineage contains no trained
    # cell. (The real fix is upstream -- ensure synthesis trains every leaf --
    # but this surfaces residual holes at deploy instead of shipping silently.)
    def _resolves_to_model(leaf: str) -> bool:
        cur = leaf
        while True:
            if cur in models:
                return True
            j = cur.rfind("#")
            if j < 0:
                return False
            cur = cur[:j]

    _bases = {k.split("#", 1)[0] for k in list(models) + list(tree)}
    _holes: List[str] = []
    for _b in sorted(_bases):
        try:
            _leaves = sc.leaves_under(_b, tree)
        except Exception:
            _leaves = [_b]
        for _lf in _leaves:
            if not _resolves_to_model(_lf):
                _holes.append(_lf)
    if _holes:
        ui.warn(
            "coverage",
            f"{len(_holes)} split-tree leaf region(s) have NO trained "
            f"cell in their lineage -- GEMMs there have no matching model "
            f"and fall back to analytical Origami at runtime. "
            f"e.g. {_holes[:4]}. Upstream fix: ensure synthesis trains "
            f"every leaf (O1).",
        )
    elif not quiet:
        ui.ok(
            "coverage",
            "every deployed split-tree leaf resolves to a "
            "trained cell (no orphaned regions)",
        )
    for label in sorted(models.keys()):
        m = models[label]
        sd = m["state_dict"]
        norms = m.get("feature_norms", {})
        meta = meta_by_label.get(label, {})
        # Per-cell smart_K signature list. Prefer the model entry
        # (the bundle written by stage 5; survives carry-forward
        # across rounds), fall back to cells.json metadata (in case
        # an older bundle was missing it for some cells).
        sk_sigs_raw = (
            m.get("smart_k_signatures") or meta.get("smart_k_signatures") or []
        )
        sk_sigs = [tuple(int(v) for v in s) for s in sk_sigs_raw]
        cell = {
            "label": label,
            "embed_dim": int(m["embed_dim"]),
            "hidden_dim": int(m["hidden_dim"]),
            "inter_hidden": int(m["inter_hidden"]),
            "temperature": _temperature_scalar(sd),
            "q_mean": list(norms.get("q_mean", [0.0] * len(q_names))),
            "q_std": list(norms.get("q_std", [1.0] * len(q_names))),
            "i_mean": list(norms.get("i_mean", [0.0] * len(i_names))),
            "i_std": list(norms.get("i_std", [1.0] * len(i_names))),
            "x_mean": list(norms.get("x_mean", [0.0] * len(x_names))),
            "x_std": list(norms.get("x_std", [1.0] * len(x_names))),
            "state_dict": sd,
            "smart_k_signatures": sk_sigs,
            # best_sel_eff / best_epoch come from cells.json (THIS
            # round's freshly-trained cells). Carry-forwarded cells
            # don't have a fresh measurement here; the model entry's
            # `best_epoch` is preserved from the round that trained
            # it, so we fall back to that.
            "best_sel_eff": meta.get("best_sel_eff"),
            "best_epoch": meta.get("best_epoch", m.get("best_epoch")),
        }
        cells.append(cell)

    if not cells:
        ui.err(
            "pack",
            "no trainable cells found in models.pt -- "
            "refusing to emit an empty weights binary",
        )
        return 1

    wdt = _WEIGHT_DTYPE_BY_NAME.get((weight_dtype or "bf16").lower())
    if wdt is None:
        ui.err(
            "pack",
            f"unknown weight_dtype {weight_dtype!r}; "
            f"expected one of {sorted(_WEIGHT_DTYPE_BY_NAME)}",
        )
        return 1
    bin_bytes = _emit_v6_bin(
        feature_names_hash=feature_names_hash,
        arch=arch,
        q_dim=len(q_names),
        i_dim=len(i_names),
        x_dim=len(x_names),
        splits=splits,
        cells=cells,
        weight_dtype=wdt,
    )

    manifest = _build_manifest(
        feature_names={"query": q_names, "item": i_names, "interaction": x_names},
        feature_names_hash=feature_names_hash,
        feature_version=feature_version,
        arch=arch,
        config_id=config_id,
        splits=splits,
        cells=cells,
        weights_bin_size=len(bin_bytes),
        weight_dtype=(weight_dtype or "bf16").lower(),
    )
    summary = _format_summary(manifest, tree)
    install_md = _install_md(manifest)

    if not quiet:
        ui.ok(
            "pack",
            f"v6 weights .bin ({weight_dtype}) = "
            f"{len(bin_bytes):,} bytes; "
            f"{len(cells)} cells; {len(splits)} splits",
        )

    # ── stage everything under <out-dir>/staged_for_deploy/ ────────
    staging_root = out_dir / "staged_for_deploy"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    data_dir = staging_root / weights_subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    from lib import fs as _fsutil

    _fsutil.atomic_write_bytes(data_dir / bin_filename, bin_bytes)
    _fsutil.write_json(data_dir / _REPO_MANIFEST, manifest)
    (data_dir / _REPO_SUMMARY).write_text(summary)
    inst_path = staging_root / _REPO_INSTALL_MD
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    inst_path.write_text(install_md)

    if not quiet:
        ui.ok("stage", f"staged bundle: {staging_root}")
        ui.ok(
            "stage",
            f"  .bin       : "
            f"{weights_subdir}/{bin_filename} "
            f"({len(bin_bytes):,} bytes)",
        )
        if lib_stems:
            ui.ok(
                "stage",
                f"  index keys : {len(lib_stems)} logic stem(s) " f"-> {bin_filename}",
            )
        ui.ok("stage", f"  manifest   : {weights_subdir}/{_REPO_MANIFEST}")
        ui.ok("stage", f"  install.md : {_REPO_INSTALL_MD}")

    if rocm_libraries_to_deploy is None:
        if not quiet:
            ui.banner(
                "Done (no rocm_libraries_to_deploy given; " "staging only)", ui.C.GREEN
            )
        return 0

    if not apply_patch:
        if not quiet:
            ui.warn(
                "apply",
                "apply_patch=false -- bundle was staged but NOT "
                "copied into the deploy repo. Rerun with "
                "--apply-patch to commit.",
            )
            ui.banner("Done (staged only; nothing written to repo)", ui.C.GREEN)
        return 0

    # ── copy onto the repo ───────────────────────────────────────
    # CONFIDENTIALITY: ONLY the weights `.bin` is written into the
    # rocm-libraries checkout. The manifest JSON + summary TXT carry
    # CONFIDENTIAL per-cell performance numbers, so they stay ONLY in
    # the pipeline-side run output under
    # `<out-dir>/staged_for_deploy/` (written above) and are NEVER
    # copied into the (potentially shared / published) deploy repo.
    repo = rocm_libraries_to_deploy
    target_data_dir = repo / weights_subdir
    target_data_dir.mkdir(parents=True, exist_ok=True)
    # Atomic copy of the .bin into the LIVE repo: stage to a temp file
    # then os.replace, so a concurrent reader (hipblaslt-bench
    # mmap'ing the weights) can never observe a torn .bin -- which the
    # C++ loader would silently reject and fall back to plain Origami.
    _dst = target_data_dir / bin_filename
    _tmp = _dst.with_suffix(_dst.suffix + ".tmp")
    shutil.copy2(data_dir / bin_filename, _tmp)
    os.replace(_tmp, _dst)
    # Merge the per-library index (preserving every OTHER dtype/layout's
    # entries) so this model coexists with previously-deployed ones.
    if lib_stems:
        _merge_index(target_data_dir / _REPO_INDEX_FILE, lib_stems, bin_filename)
    # CONFIDENTIALITY: the install .md describes the internal training/deploy
    # pipeline (tool + stage names) and is NOT written into the (public)
    # deployed repo. The runtime documents itself via shared/tilewright/CLAUDE.md.
    # The .md is kept in the pipeline-side staging output only (above).
    if not quiet:
        ui.ok("write", f"{target_data_dir / bin_filename}")
        if lib_stems:
            ui.ok(
                "write",
                f"{target_data_dir / _REPO_INDEX_FILE}  "
                f"(+{len(lib_stems)} entry/entries)",
            )
        ui.info(
            "write",
            f"install.md, manifest + summary kept in run output only "
            f"(confidential / internal): {inst_path}, "
            f"{data_dir / _REPO_MANIFEST}, {data_dir / _REPO_SUMMARY}",
        )

    if build_after_apply:
        rc = _build_after_apply(repo, out_dir, build_jobs, quiet)
        if rc != 0:
            return rc

    if not quiet:
        end_msg = "weights + manifest written into rocm-libraries"
        if build_after_apply:
            end_msg += " + origami/hipblaslt-bench relinked"
        ui.banner(f"Done ({end_msg})", ui.C.GREEN)
    return 0


# ── argparse main ──────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage06: deploy per-cell models into a rocm-"
        "libraries / hipBLASLt checkout."
    )
    ap.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="dir with models.pt + cells.json from stage05.",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        required=True,
        help="OOB configuration yaml (config_id + arch).",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="staging dir for the deploy bundle."
    )
    ap.add_argument(
        "--prior-round-dir",
        "--prior-run-dir",
        type=Path,
        action="append",
        default=None,
        dest="prior_round_dir",
        help="repeatable; each MUST point at a " "`<run>/round_<i>/` ROUND directory.",
    )
    ap.add_argument(
        "--rocm-libraries-to-deploy",
        type=Path,
        default=None,
        help="rocm-libraries CHECKOUT to deploy into. When "
        "omitted, stage06 only emits the staging bundle.",
    )
    ap.add_argument(
        "--apply-patch",
        action="store_true",
        help="actually drop the .bin/manifest into the "
        "target checkout (default: stage only).",
    )
    ap.add_argument(
        "--build-after-apply",
        action="store_true",
        help="after dropping the .bin, relink the origami "
        "library AND hipblaslt-bench AND copy the "
        "origami Python binding into the venv site-"
        "packages. Assumes kernels are already built. "
        "Requires --apply-patch; otherwise a no-op.",
    )
    ap.add_argument(
        "--build-jobs",
        type=int,
        default=None,
        help="parallel jobs for the build commands "
        "(default: half the CPUs for `make`, "
        "unbounded for `cmake --build`).",
    )
    ap.add_argument(
        "--weight-dtype",
        choices=("fp32", "bf16", "int8", "int4"),
        default="bf16",
        help="precision for the MLP `.weight` matrices in "
        "the deployed .bin. bf16 (default) ~halves the "
        "binary with no pick-parity change (C++ "
        "promotes bf16->fp32 at load; the parity check "
        "rounds identically). fp32 = legacy full "
        "precision.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return deploy(
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        config_yaml=args.config_yaml,
        prior_round_dirs=args.prior_round_dir,
        rocm_libraries_to_deploy=args.rocm_libraries_to_deploy,
        apply_patch=args.apply_patch,
        build_after_apply=args.build_after_apply,
        build_jobs=args.build_jobs,
        weight_dtype=args.weight_dtype,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
