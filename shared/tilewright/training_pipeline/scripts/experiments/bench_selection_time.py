#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Parse a Tensile Origami UserArgs.yaml (the source-of-truth library a
hipBLASLt build ships) and emit a TSV that bench_selection_time.cpp
consumes.

Usage:
  bench_selection_time.py --yaml <path> --out <tsv> [--max-configs 0]
                          [--shapes 'M,N,K,B;M,N,K,B;...']

The TSV is line-oriented:

    PROBLEM_HEADER  a_dtype b_dtype c_dtype d_dtype mi_dtype trans_a trans_b
    NPROBLEMS <n>
    M N K B                                                (×n)
    NCONFIGS <n>
    mt_m mt_n mt_k mi_m mi_n mi_k occ cha chb grvwa grvwb gwvwd \
        chc chd che pgr plr lrvw lpa lpb lbspa lbspb lsu        (×n)

The transpose / dtype info comes from the YAML's top-level ProblemType
block, so every problem in this run inherits it (matching the
single-library assumption -- each Tensile UserArgs.yaml covers one
(dtype, transpose) layout).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


DEFAULT_SHAPES = [
    # M, N, K, B  -- spans tiny / small / mid / large for OOB coverage
    (32, 32, 64, 1),
    (128, 128, 128, 1),
    (512, 512, 512, 1),
    (1024, 1024, 1024, 1),
    (4096, 4096, 4096, 1),
    (8192, 8192, 8192, 1),
    (16384, 1024, 1024, 1),
    (1024, 16384, 1024, 1),
]


def parse_shapes_arg(s: str) -> list[tuple[int, int, int, int]]:
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(x) for x in chunk.split(",")]
        if len(parts) == 3:
            m, n, k = parts
            b = 1
        elif len(parts) == 4:
            m, n, k, b = parts
        else:
            raise SystemExit(f"bad shape spec: {chunk!r} (need M,N,K or M,N,K,B)")
        out.append((m, n, k, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Tensile Origami UserArgs.yaml (one dtype/transpose layout)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="cap candidates (0 = all unique kernels in the yaml)",
    )
    ap.add_argument(
        "--shapes",
        type=str,
        default=None,
        help='";"-separated GEMM shapes "M,N,K[,B]". '
        "Default is an 8-shape tier-spanning set.",
    )
    ap.add_argument(
        "--shapes-file",
        type=Path,
        default=None,
        help="file with one shape per line: 'M N K [B]'. "
        "Overrides --shapes and the built-in default.",
    )
    args = ap.parse_args()

    doc = yaml.safe_load(args.yaml.read_text())
    # Tensile UserArgs.yaml layout:
    #   [ MinimumRequiredVersion, arch, arch, [Device ...], ProblemType,
    #     [solution_dict, solution_dict, ...], ... ]
    # Find the ProblemType (dict with DataType) and the solution list
    # (a list whose first element is a dict with MacroTile0).
    problem_type = None
    solutions: list[dict] = []
    for item in doc:
        if isinstance(item, dict) and "DataType" in item and "TransposeA" in item:
            problem_type = item
        elif (
            isinstance(item, list)
            and item
            and isinstance(item[0], dict)
            and "MacroTile0" in item[0]
        ):
            solutions = item
    if problem_type is None:
        raise SystemExit(f"no ProblemType block found in {args.yaml}")
    if not solutions:
        raise SystemExit(f"no solutions list found in {args.yaml}")

    a_dtype = int(problem_type.get("DataTypeA", problem_type["DataType"]))
    b_dtype = int(problem_type.get("DataTypeB", problem_type["DataType"]))
    c_dtype = int(problem_type.get("DestDataType", problem_type["DataType"]))
    d_dtype = c_dtype
    mi_dtype = int(problem_type.get("DataType", 0))  # MI input type
    # Tensile bool -> origami transpose_t: T=0, N=1.
    trans_a = 0 if bool(problem_type.get("TransposeA", False)) else 1
    trans_b = 0 if bool(problem_type.get("TransposeB", False)) else 1

    seen: set[tuple] = set()
    configs: list[dict] = []
    for s in solutions:
        try:
            row = {
                "mt_m": int(s["MacroTile0"]),
                "mt_n": int(s["MacroTile1"]),
                "mt_k": int(s["DepthU"]),
                "mi_m": int(s["MatrixInstM"]),
                "mi_n": int(s["MatrixInstN"]),
                "mi_k": int(s["MatrixInstK"]),
                "occ": max(int(s.get("CUOccupancy", -1) or -1), 1),
                "cha": int(s.get("NonTemporalA", 0) or 0),
                "chb": int(s.get("NonTemporalB", 0) or 0),
                "grvwa": int(s.get("GlobalReadVectorWidthA", 1) or 1),
                "grvwb": int(s.get("GlobalReadVectorWidthB", 1) or 1),
                "gwvwd": int(s.get("GlobalWriteVectorWidth", 1) or 1),
                "chc": int(s.get("NonTemporalC", 0) or 0),
                "chd": int(s.get("NonTemporalD", 0) or 0),
                "che": int(s.get("NonTemporalE", 0) or 0),
                "pgr": int(s.get("PrefetchGlobalRead", 1) or 1),
                "plr": int(s.get("PrefetchLocalRead", 1) or 1),
                "lrvw": int(s.get("LocalReadVectorWidth", 1) or 1),
                "lpa": int(s.get("LdsPadA", 0) or 0),
                "lpb": int(s.get("LdsPadB", 0) or 0),
                "lbspa": int(s.get("LdsBlockSizePerPadA", 0) or 0),
                "lbspb": int(s.get("LdsBlockSizePerPadB", 0) or 0),
                "lsu": int(s.get("LocalSplitU", 1) or 1),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if row["mt_m"] <= 0 or row["mt_n"] <= 0 or row["mt_k"] <= 0:
            continue
        sig = (
            row["mt_m"],
            row["mt_n"],
            row["mt_k"],
            row["mi_m"],
            row["mi_n"],
            row["mi_k"],
            row["cha"],
            row["chb"],
        )
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(row)
        if args.max_configs and len(configs) >= args.max_configs:
            break

    if args.shapes_file:
        shapes = []
        for line in args.shapes_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                shapes.append((int(parts[0]), int(parts[1]), int(parts[2]), 1))
            elif len(parts) == 4:
                shapes.append(tuple(int(x) for x in parts))
            else:
                raise SystemExit(f"bad line in {args.shapes_file}: {line!r}")
    elif args.shapes:
        shapes = parse_shapes_arg(args.shapes)
    else:
        shapes = DEFAULT_SHAPES

    with args.out.open("w") as f:
        f.write(f"# parsed from {args.yaml.name}\n")
        f.write(
            "# dtype enum follows origami::data_type_t "
            "(Float=0 Half=4 BFloat16=7 Float8=16 BFloat8=17)\n"
        )
        f.write(
            f"PROBLEM_HEADER {a_dtype} {b_dtype} {c_dtype} {d_dtype} "
            f"{mi_dtype} {trans_a} {trans_b}\n"
        )
        f.write(f"NPROBLEMS {len(shapes)}\n")
        f.write("# M N K B\n")
        for m, n, k, b in shapes:
            f.write(f"{m} {n} {k} {b}\n")
        f.write(f"NCONFIGS {len(configs)}\n")
        f.write(
            "# mt_m mt_n mt_k mi_m mi_n mi_k occ cha chb grvwa grvwb gwvwd"
            " chc chd che pgr plr lrvw lpa lpb lbspa lbspb lsu\n"
        )
        for r in configs:
            vals = (
                r["mt_m"],
                r["mt_n"],
                r["mt_k"],
                r["mi_m"],
                r["mi_n"],
                r["mi_k"],
                r["occ"],
                r["cha"],
                r["chb"],
                r["grvwa"],
                r["grvwb"],
                r["gwvwd"],
                r["chc"],
                r["chd"],
                r["che"],
                r["pgr"],
                r["plr"],
                r["lrvw"],
                r["lpa"],
                r["lpb"],
                r["lbspa"],
                r["lbspb"],
                r["lsu"],
            )
            f.write(" ".join(str(v) for v in vals) + "\n")

    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  yaml: {args.yaml}", file=sys.stderr)
    print(
        f"  dtypes: A={a_dtype} B={b_dtype} C={c_dtype} D={d_dtype} MI={mi_dtype}"
        f"  trans: A={'T' if trans_a==0 else 'N'} B={'T' if trans_b==0 else 'N'}",
        file=sys.stderr,
    )
    print(
        f"  problems: {len(shapes)}  configs (unique): {len(configs)}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
