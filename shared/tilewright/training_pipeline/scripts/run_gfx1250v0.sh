#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Launch one gfx1250v0 tilewright training run with the environment the
# pipeline needs. Usage:  scripts/run_gfx1250v0.sh <config.yaml> [extra orchestrator args...]
#
# Environment:
#   TILEWRIGHT_BENCH_ROOT   rocm-libraries checkout (or a pinned snapshot of
#                           one) whose built hipblaslt-bench + Tensile library
#                           the run benchmarks. Default: the checkout this
#                           pipeline lives in. PIN_DIR is accepted as an alias.
#   TILEWRIGHT_DEPLOY_ROOT  checkout that receives the trained weights
#                           (stage06). Default: the checkout this pipeline
#                           lives in.
#   TILEWRIGHT_ORIGAMI_BUILD  out-of-tree origami build providing a liborigami
#                           matching the python binding.
#                           Default: training_pipeline/.origami_build_gfx1250
#   TILEWRIGHT_EXTRA_LD_LIBRARY_PATH  appended to LD_LIBRARY_PATH; see below.
#   TILEWRIGHT_DATASETS     directory holding stage08 dataset yamls, for
#                           configs that list any.
# The configs reference the TILEWRIGHT_* paths as ${VAR}; this script exports
# them so the orchestrator's expansion and this script's library setup agree.
#
# Why this wrapper exists -- three environment facts that silently produce a
# WRONG run if you get them wrong, and that nothing else in the pipeline sets:
#
#  1. HIPBLASLT_TENSILE_LIBPATH must stay UNSET.
#     tensile_host.cpp only promotes to the ASIC-revision subtree
#     (`<libdir>/gfx1250v0/`) in the branch where that env var is absent.
#     Setting it pins the path verbatim -> you silently benchmark the v1
#     library on v0 silicon. We therefore actively unset it.
#
#  2. LD_LIBRARY_PATH must point at the bench root's library dir first.
#     LD_LIBRARY_PATH beats the binary's DT_RUNPATH, so an inherited value
#     naming another checkout makes hipblaslt-bench load THAT checkout's
#     libhipblaslt -- and it still succeeds, so the mistake is invisible.
#     Same hazard for liborigami: /opt/rocm/lib can ship an older one whose
#     ABI is missing symbols the freshly built python binding needs.
#
#  3. PYTHONPATH needs origami's python/src. The orchestrator prepends it
#     for stage04 itself, but stage05/07/08 and any manual invocation need
#     it too, and lib/features.py cannot compute a single feature without
#     `origami.get_hardware_for_device`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# training_pipeline/ lives at <rocm-libraries>/shared/tilewright/training_pipeline.
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
CFG="${1:?usage: run_gfx1250v0.sh <config.yaml> [orchestrator args...]}"; shift || true

export TILEWRIGHT_BENCH_ROOT="${TILEWRIGHT_BENCH_ROOT:-${PIN_DIR:-$REPO_ROOT}}"
export TILEWRIGHT_DEPLOY_ROOT="${TILEWRIGHT_DEPLOY_ROOT:-$REPO_ROOT}"
PIN="$TILEWRIGHT_BENCH_ROOT"
ORIGAMI_BUILD="${TILEWRIGHT_ORIGAMI_BUILD:-$HERE/.origami_build_gfx1250}"
ORIGAMI_PY="$PIN/shared/origami/python/src"

[ -d "$PIN" ] || { echo "FATAL: bench root missing: $PIN" >&2; exit 2; }

unset HIPBLASLT_TENSILE_LIBPATH || true
# The bench root's RUNPATH names libtensilelite-host.so.1 in its SOURCE build
# tree, which the library dir does not cover -- so a rebuild during a run would
# swap the Tensile host out from under a pinned snapshot. Put the root's own
# copy ahead of it.
LD_LIBRARY_PATH="$PIN/projects/hipblaslt/build/release/library:$PIN/projects/hipblaslt/build/release/tensilelite:$ORIGAMI_BUILD:/opt/rocm/lib"
# stage07 (parity) runs the tilewright-enabled bench from the deploy checkout.
# If that was built against a different ROCm than the bench root it can need
# runtime libs (e.g. libomp.so, libamd_smi.so) that are not on this path;
# without them it exits 127 -- the dynamic loader, not a missing command -- and
# parity is skipped while the rest of the run reports success. Point
# TILEWRIGHT_EXTRA_LD_LIBRARY_PATH at the directories that carry them.
# APPENDED, never prepended: the bench root's libhipblaslt/liborigami must win.
if [ -n "${TILEWRIGHT_EXTRA_LD_LIBRARY_PATH:-}" ]; then
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TILEWRIGHT_EXTRA_LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH
export PYTHONPATH="$ORIGAMI_PY${PYTHONPATH:+:$PYTHONPATH}"

# Keep each training worker from grabbing every core (stage05 spawns a pool).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------------------------------------------------------------------------
# ORIGAMI-ONLY CANDIDATE SET
#
# TENSILE_PREDICTION_LIB=1 alone is NOT enough. lib/bench_env.py sets it for
# every bench, but ExactLogicLibrary only skips EqualityMatching/RangeMatching
# rows under it, so GridBased and Equality kernels still reach the candidate
# set. TENSILE_PREDICTION_LIB_STRICT=1 drops GridBased; the kernel-name
# exclusion drops the rest.
#
# All gfx1250v0 Origami kernels are "_HA_S_SAV_UserArgs_"; the Equality and
# GridBased libraries use "_HA_S_UserArgs_" (no SAV), so that substring is an
# exact separator for both MXFP8 (F8BS) and bf16 (BBS).
# ---------------------------------------------------------------------------
export TENSILE_PREDICTION_LIB_STRICT="${TENSILE_PREDICTION_LIB_STRICT:-1}"
export HIPBLASLT_EXCLUDE_KERNEL_SUBSTR="${HIPBLASLT_EXCLUDE_KERNEL_SUBSTR:-_HA_S_UserArgs_}"
# TemporalHint short-circuit: evaluate one of {non-TH, THA1, THB1}, not all three.
export ORIGAMI_DEDUP_EQUIVALENT_CONFIGS="${ORIGAMI_DEDUP_EQUIVALENT_CONFIGS:-1}"
echo "origami-only    : STRICT=$TENSILE_PREDICTION_LIB_STRICT dedup=$ORIGAMI_DEDUP_EQUIVALENT_CONFIGS"
echo "kernel exclusion: $HIPBLASLT_EXCLUDE_KERNEL_SUBSTR"

# Optional: drop kernels by macro tile from the Origami (Prediction) candidate
# set, as a comma-separated "MxNxK" list. Only honoured by a hipblaslt-bench
# built with the exclusion patch; unset or empty means "exclude nothing", which
# is the upstream behaviour.
if [ -n "${ORIGAMI_EXCLUDE_MACRO_TILES:-}" ]; then
  export ORIGAMI_EXCLUDE_MACRO_TILES
  echo "macro-tile exclusion: $ORIGAMI_EXCLUDE_MACRO_TILES"
fi

# Optional: drop kernels by name substring, applied where hipBLASLt enumerates
# supported solutions -- so it covers every library family (Origami/Prediction,
# Equality, GridBased), unlike the origami-level knob above which only sees the
# Prediction pool. Only honoured by a hipblaslt-bench carrying the exclusion
# patch.
if [ -n "${HIPBLASLT_EXCLUDE_KERNEL_SUBSTR:-}" ]; then
  export HIPBLASLT_EXCLUDE_KERNEL_SUBSTR
  echo "kernel exclusion    : $HIPBLASLT_EXCLUDE_KERNEL_SUBSTR"
fi

_rev="$(head -1 "$PIN/PINNED_COMMIT" 2>/dev/null || git -C "$PIN" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=== gfx1250v0 run ==="
echo "config      : $CFG"
echo "bench root  : $PIN  ($_rev)"
echo "deploy root : $TILEWRIGHT_DEPLOY_ROOT"
echo "bench       : $PIN/projects/hipblaslt/build/release/clients/hipblaslt-bench"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PYTHONPATH=$PYTHONPATH"
echo "HIPBLASLT_TENSILE_LIBPATH=<unset, so v0 promotion happens>"
python3 -c "import origami; h=origami.get_hardware_for_device(0); print(f'origami probe OK: N_CU={h.N_CU} XCD={h.NUM_XCD} clk={h.compute_clock_ghz}')"
echo "====================="

cd "$HERE"
exec python3 run_orchestrator.py --config "$CFG" "$@"
