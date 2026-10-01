#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Measure selection time of origami::rank_configs and
# origami::ml_recommender::rank_configs against a real Tensile
# Origami UserArgs.yaml.
#
# Override via env vars:
#   YAML         = Tensile UserArgs.yaml to source kernel params from
#   WEIGHTS_BIN  = tilewright_weights.bin to load
#   ORIGAMI_REPO = rocm-libraries checkout with a built liborigami.a
#   SHAPES       = ";"-separated "M,N,K[,B]" shapes (default: 8 tier-spanning)
#   ITERS, WARMUP, ARCH, ROCM_PATH
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives at <rocm-libraries>/shared/tilewright/training_pipeline/scripts/experiments.
REPO_ROOT="$(cd "$HERE/../../../../.." && pwd)"

ORIGAMI_REPO="${ORIGAMI_REPO:-$REPO_ROOT}"
YAML="${YAML:-$ORIGAMI_REPO/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Origami/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml}"
ORIGAMI_INC="$ORIGAMI_REPO/shared/origami/include"
# CMake `generate_export_header` emits origami/origami_export.h into the BUILD
# tree (not the source include dir), so add the built origami include dir too --
# co-located with the liborigami.a we link below.
ORIGAMI_GEN_INC="$ORIGAMI_REPO/projects/hipblaslt/build/release/origami/include"
ORIGAMI_LIB="$ORIGAMI_REPO/projects/hipblaslt/build/release/origami/liborigami.a"
# tilewright engine (standalone) + its header-only hipBLASLt backend adapter.
TILEWRIGHT_INC="$ORIGAMI_REPO/shared/tilewright/include"
TILEWRIGHT_BACKEND_INC="$ORIGAMI_REPO/shared/tilewright/backends/hipblaslt/include"
TILEWRIGHT_LIB="$ORIGAMI_REPO/projects/hipblaslt/build/release/tilewright/libtilewright.a"
WEIGHTS_BIN="${WEIGHTS_BIN:-$ORIGAMI_REPO/shared/tilewright/backends/hipblaslt/weights/gfx950/gfx950/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.bin}"

ARCH="${ARCH:-gfx950}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-50}"
SHAPES="${SHAPES:-}"
SHAPES_FILE="${SHAPES_FILE:-$HERE/bench_default_shapes.txt}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

OUT_DIR="${OUT_DIR:-$HERE/bench_selection_time_out}"
mkdir -p "$OUT_DIR"
BIN="$OUT_DIR/bench_selection_time"
TSV="$OUT_DIR/bench_inputs.tsv"

if [[ -z "${CXX:-}" ]]; then
  if   command -v "$ROCM_PATH/bin/amdclang++" >/dev/null 2>&1; then CXX="$ROCM_PATH/bin/amdclang++"
  elif command -v clang++ >/dev/null 2>&1;                    then CXX=clang++
  elif command -v g++     >/dev/null 2>&1;                    then CXX=g++
  else echo "no C++ compiler found; set CXX=..." >&2; exit 1
  fi
fi

for p in "$YAML" "$ORIGAMI_INC/origami/origami.hpp" \
         "$TILEWRIGHT_BACKEND_INC/tilewright/hipblaslt/adapter.hpp" \
         "$ORIGAMI_LIB" "$TILEWRIGHT_LIB" "$WEIGHTS_BIN"; do
  [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

# 1. Parse the Origami yaml -> TSV (problems + unique kernels).
#    SHAPES (inline) wins over SHAPES_FILE; SHAPES_FILE defaults to the
#    bundled synthetic TN BBS shapes that span every base grid cell.
DUMP_ARGS=(--yaml "$YAML" --out "$TSV")
if [[ -n "$SHAPES" ]]; then
  DUMP_ARGS+=(--shapes "$SHAPES")
elif [[ -n "$SHAPES_FILE" && -e "$SHAPES_FILE" ]]; then
  DUMP_ARGS+=(--shapes-file "$SHAPES_FILE")
fi
python3 "$HERE/bench_selection_time.py" "${DUMP_ARGS[@]}"

# 2. Compile.
echo "compiling with: $CXX" >&2
"$CXX" -O3 -std=c++20 -DNDEBUG \
  -I"$ORIGAMI_INC" -I"$ORIGAMI_GEN_INC" -I"$TILEWRIGHT_INC" -I"$TILEWRIGHT_BACKEND_INC" \
  -isystem "$ROCM_PATH/include" \
  -D__HIP_PLATFORM_AMD__=1 \
  "$HERE/bench_selection_time.cpp" \
  "$TILEWRIGHT_LIB" "$ORIGAMI_LIB" \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
  -lamdhip64 -ldl -lpthread \
  -o "$BIN"

# 3. Run.
exec "$BIN" \
  --weights "$WEIGHTS_BIN" \
  --inputs  "$TSV" \
  --arch    "$ARCH" \
  --iters   "$ITERS" \
  --warmup  "$WARMUP"
