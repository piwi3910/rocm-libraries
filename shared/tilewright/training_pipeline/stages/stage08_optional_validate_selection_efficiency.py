# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage08_optional_validate_selection_efficiency -- run the trained
per-cell models against one or more EXTERNAL datasets and report
per-cell selection efficiency for both the ML recommender and Origami.

Sister of stage07_validate_sel_time (which measures selection TIME):
together they tell the operator whether deploying the ML recommender
is worth the extra wall-clock.

Each "dataset" is either:

  (1) a hipblaslt-bench YAML of shape lines, OR
  (2) a folder of stage03-style block_*.log files (the load
      balancer's output -- i.e. an "offline tuning result based on
      Origami"), OR
  (3) a folder of already-enriched chunk_*.csv files (stage04
      output -- skips both the bench and the enrich steps).

When (1) is provided, stage08 invokes hipblaslt-bench through the
deployed rocm-libraries checkout (`--rocm-libraries-to-deploy`) to
produce (2). Stage04 enrichment then converts (2) into (3). Once we
have (3), the stage04b evaluator scores each GEMM with both Origami
and the ML recommender and writes a unified decisions.json per
dataset.

Multiple datasets are supported. Pass them either as repeated
`--dataset NAME=PATH` flags or as a single YAML file via `--datasets-yaml`:

    # datasets.yaml
    - name: 25k_oob
      path: /path/to/dataset_yaml_or_logs_or_chunks
    - name: real_world_traces
      path: /another/path

Output (per dataset, under <out-dir>/<dataset-name>/):
  bench/                     stage03-style block_*.log files
                             (only when input was a yaml)
  stage04_enriched/          chunk_*.csv from stage04 (only when input
                             was logs or a yaml)
  decisions.json             per-cell sel_eff (model + origami) +
                             pass/retrain decision against
                             --sel-eff-threshold
plus a global <out-dir>/summary.json comparing every dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import yaml  # noqa: E402

from lib import ui  # noqa: E402

# hipblaslt scaleA mode -> origami `*_mx_block_size` (32 for MXFP8).
from lib.dat import (  # noqa: E402
    mx_block_size_for_scale_mode as _mx_block_size_for_scale_mode,
)


# ── input-kind detection ────────────────────────────────────────────────────


def _detect_kind(path: Path) -> str:
    """Return "yaml" / "logs" / "enriched" / "unknown"."""
    if not path.exists():
        return "missing"
    if path.is_file():
        return "yaml" if path.suffix.lower() in (".yaml", ".yml") else "unknown"
    # Directory: look for block_*.log or chunk_*.csv.
    has_logs = any(
        p.name.startswith("block_") and p.name.endswith(".log") for p in path.iterdir()
    )
    has_chunks = any(
        p.name.startswith("chunk_") and p.name.endswith(".csv") for p in path.iterdir()
    )
    if has_chunks:
        return "enriched"
    if has_logs:
        return "logs"
    return "unknown"


# ── thin subprocess runner ──────────────────────────────────────────────────


def _run_subprocess(
    cmd: List[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        ui.info("run", " ".join(str(c) for c in cmd))
    t0 = time.time()
    # buffering=1 + per-line flush keeps the log file in sync with
    # stdout so `tail -f` works during long offline-tuning runs.
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
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
    if not quiet:
        (ui.ok if rc == 0 else ui.err)(
            "run", f"  completed in {ui.fmt_dur(el)} " f"(rc={rc})"
        )
    return rc


# ── one-dataset workflow ────────────────────────────────────────────────────


def _run_one_dataset(
    name: str,
    path: Path,
    out_root: Path,
    train_dir: Path,
    rocm_libraries: Optional[Path],
    bench_rocm_libraries: Optional[Path],
    devices: str,
    blocks_per_gpu: int,
    sel_eff_threshold: str,
    min_cell_gemms: int,
    arch: str,
    device: str,
    smart_k: int,
    prior_round_dirs: List[Path],
    origami_python_path: Optional[Path],
    workload_filter: Optional[Dict[str, str]],
    config_yaml: Optional[Path],
    quiet: bool,
    scale_mode: Optional[int] = None,
) -> Dict[str, Any]:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    kind = _detect_kind(path)
    if not quiet:
        ui.banner(f"dataset :: {name}  kind={kind}", ui.C.BLUE)
        ui.info("cfg", f"path : {path}")
        ui.info("cfg", f"out  : {out}")
    if kind in ("missing", "unknown"):
        ui.err(
            "err",
            f"dataset '{name}' at {path} is " f"missing or unrecognised (kind={kind})",
        )
        return {
            "name": name,
            "path": str(path),
            "rc": 1,
            "reason": f"input_kind={kind}",
        }

    # ── (1) yaml -> run hipblaslt-bench (Origami offline tune) ─────────────
    logs_dir: Path
    if kind == "yaml":
        if rocm_libraries is None:
            ui.err(
                "err",
                "dataset is a yaml but --rocm-libraries-"
                "to-deploy was not provided; cannot run "
                "hipblaslt-bench. Either provide it, or pass an "
                "already-tuned logs/enriched directory.",
            )
            return {
                "name": name,
                "path": str(path),
                "rc": 1,
                "reason": "yaml_requires_rocm_libraries",
            }
        bench_bin = (
            rocm_libraries / "projects/hipblaslt/build/release/clients/hipblaslt-bench"
        )
        if not bench_bin.exists():
            ui.err("err", f"hipblaslt-bench not found at {bench_bin}")
            return {
                "name": name,
                "path": str(path),
                "rc": 1,
                "reason": "missing_bench_binary",
            }
        logs_dir = out / "bench" / "logs"
        # Mirror the TRAINING pipeline's stage01 -> stage02 -> stage03
        # path rather than dumping a fixed `iters: 2`. A raw yaml has no
        # calibration: a constant iters either under-measures fast shapes
        # (noisy winners -> the true best kernel gets demoted to "tied for
        # second", tanking sel_eff) or wastes time on slow ones, and a
        # flat skip ratio either prunes real candidates or benches every
        # obviously-bad solution. So we run an actual stage02 probe to
        # measure each shape's winner `us`, then calibrate per-shape
        # `iters` / `cold_iters` / `skip_slow_solution_ratio` from the
        # config's `probe:` section -- exactly what stage02 does in
        # training. stage03 then benches the FULL candidate set
        # (requested_solution_num=-1) at those calibrated knobs.
        import yaml as _yaml  # local import; pyyaml is a pipeline dep

        bench_in = out / "bench_input"
        bench_in.mkdir(parents=True, exist_ok=True)
        with path.open() as _f:
            _entries = _yaml.safe_load(_f) or []
        if isinstance(_entries, dict):
            _entries = [_entries]
        # Does the config carry a usable `probe:` section?
        probe_cfg: Dict[str, Any] = {}
        if config_yaml is not None and Path(config_yaml).exists():
            try:
                with Path(config_yaml).open() as _cf:
                    probe_cfg = (_yaml.safe_load(_cf) or {}).get("probe", {}) or {}
            except Exception as _e:  # noqa: BLE001
                ui.warn(
                    "probe",
                    f"{name}: could not read probe: section "
                    f"from {config_yaml} ({_e}); using fixed iters",
                )
        do_probe = bool(probe_cfg)
        # Seed shapes.yaml = the FULL candidate sweep (requested_solution_
        # num=-1) + print_kernel_info=1 (stage04 enrichment keys on the
        # `--Solution index/name` lines; without it stage04 matches 0
        # gemms). The iters/cold_iters here are only placeholders: the
        # stage02 probe overwrites them per shape. If no probe config is
        # available they stay at this fixed fallback (with cold_iters>=1
        # so the GPU cold-start doesn't land on -- and deflate -- origami's
        # top pick, the first solution measured in the analytical order).
        shapes_yaml = bench_in / "shapes.yaml"
        n_written = 0
        with shapes_yaml.open("w") as _f:
            for _e in _entries:
                if not isinstance(_e, dict):
                    continue
                _e = dict(_e)
                _e["requested_solution_num"] = -1
                _e["print_kernel_info"] = 1
                _e.setdefault("iters", 2)
                _e.setdefault("cold_iters", 2)
                _f.write(
                    "- "
                    + _yaml.safe_dump(
                        _e, default_flow_style=True, sort_keys=False, width=10**9
                    ).strip()
                    + "\n"
                )
                n_written += 1
        if not quiet:
            ui.info(
                "bench",
                f"{name}: wrote {n_written} shapes "
                f"(requested_solution_num=-1, full candidate sweep)",
            )
        # ── stage02 probe -> shapes_probed.yaml (calibrated knobs) ──────
        if do_probe:
            s2_cmd = [
                sys.executable,
                "-u",
                str(THIS_DIR / "stage02_probe.py"),
                "--in-dir",
                str(bench_in),
                "--out-dir",
                str(bench_in),
                "--config-yaml",
                str(config_yaml),
                "--bench-binary",
                str(bench_bin),
                "--devices",
                devices,
            ]
            if quiet:
                s2_cmd.append("--quiet")
            rc = _run_subprocess(s2_cmd, out / "stage02.log", quiet=quiet)
            if rc != 0:
                ui.warn(
                    "probe",
                    f"{name}: stage02 probe failed (rc={rc}); "
                    f"stage03 falls back to the fixed-iters shapes.yaml",
                )
        elif not quiet:
            _why = (
                "--config-yaml not provided"
                if config_yaml is None
                else f"no probe: section in {config_yaml}"
            )
            ui.warn(
                "probe",
                f"{name}: skipping probe ({_why}); benching "
                f"the fixed-iters shapes.yaml",
            )
        # ── stage03 full bench (reads shapes_probed.yaml, else shapes.yaml)
        cmd = [
            sys.executable,
            "-u",
            str(THIS_DIR / "stage03_load_balance_offline_tuning.py"),
            "--in-dir",
            str(bench_in),
            "--out-dir",
            str(out / "bench"),
            "--bench-binary",
            str(bench_bin),
            "--devices",
            devices,
            "--blocks-per-gpu",
            str(blocks_per_gpu),
        ]
        rc = _run_subprocess(cmd, out / "stage03.log", quiet=quiet)
        if rc != 0:
            return {
                "name": name,
                "path": str(path),
                "rc": rc,
                "reason": "stage03_failed",
            }
    elif kind == "logs":
        logs_dir = path
    else:
        logs_dir = path  # enriched -- skip stage04 too

    # ── (2) stage04 enrichment (skip if input was already enriched) ────────
    enriched_dir: Path
    if kind == "enriched":
        enriched_dir = path
    else:
        # Enrichment MUST use the SAME rocm-libraries checkout that
        # produced the bench log -- the bench's `--Solution index`
        # numbers index into THAT library's MatchTable, and the
        # `--Solution name` strings only resolve in THAT library's
        # .dat files. Pointing the enricher at a newer/older
        # library means 100% of `sol_idx_global` lookups miss and
        # rows fall through with empty kernel params (later marked
        # `bad_kp` by stage 5). Each stage08 dataset can carry its
        # own `rocm_libraries_used` (the build the bench was run
        # against -- find the commit SHA on `hipBLASLt git version`
        # at the top of any bench log). Falls back to the deploy
        # checkout only when no per-dataset value is given.
        enrich_rocm = (
            bench_rocm_libraries if bench_rocm_libraries is not None else rocm_libraries
        )
        if enrich_rocm is None:
            ui.err(
                "err",
                "enrichment needs --bench-rocm-libraries "
                "(or --rocm-libraries-to-deploy) for dat_dir + "
                "match_table",
            )
            return {
                "name": name,
                "path": str(path),
                "rc": 1,
                "reason": "enrichment_needs_rocm_libraries",
            }
        dat_dir = (
            enrich_rocm / "projects/hipblaslt/build/release/Tensile/library" / arch
        )
        match_table = (
            enrich_rocm
            / "projects/hipblaslt/build/release/device-library"
            / "MatchTable.yaml"
        )
        if not dat_dir.exists() or not match_table.exists():
            ui.err(
                "err",
                f"bench-time library paths missing: "
                f"dat_dir={dat_dir} (exists={dat_dir.exists()}) "
                f"match_table={match_table} "
                f"(exists={match_table.exists()}). The library at "
                f"{enrich_rocm} probably isn't built yet -- run "
                f"the cmake + make there first, then retry.",
            )
            return {
                "name": name,
                "path": str(path),
                "rc": 1,
                "reason": "bench_library_not_built",
            }
        if not quiet:
            ui.info(
                "cfg",
                f"enrich library : {enrich_rocm} "
                f"({'per-dataset' if bench_rocm_libraries else 'deploy'})",
            )

        # Pick the origami python binding with a built .so. Same
        # logic as the eval-phase resolver below: explicit
        # `--origami-python-path` (training checkout) wins;
        # otherwise try the deploy. Without a built `.so`, the
        # enrichment falls back to "skip rows have blank kernel
        # params" which is recoverable -- but the eval phase needs
        # a real binding for `rank_configs`, so we use the same
        # resolution to keep the two phases consistent.
        def _has_built_so(pp: Path) -> bool:
            return bool(list(pp.glob("origami/origami.cpython-*.so")))

        if origami_python_path is not None and _has_built_so(origami_python_path):
            origami_pp = origami_python_path
        else:
            origami_pp = rocm_libraries / "shared/origami/python/src"
        enriched_dir = out / "stage04_enriched"
        env = os.environ.copy()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(origami_pp) + (":" + prev if prev else "")
        cmd = [
            sys.executable,
            "-u",
            str(THIS_DIR / "stage04_convert_to_enriched_dataset.py"),
            "--log-dir",
            str(logs_dir),
            "--out-dir",
            str(enriched_dir),
            "--dat-dir",
            str(dat_dir),
            "--match-table",
            str(match_table),
            "--progress-every",
            "60",
        ]
        # Same scale mode the pool filter uses, so the enriched rows and
        # the config pool below are built from the SAME libraries.
        if scale_mode is not None:
            cmd += ["--scale-mode", str(int(scale_mode))]
        rc = _run_subprocess(cmd, out / "stage04.log", env=env, quiet=quiet)
        if rc != 0:
            return {
                "name": name,
                "path": str(path),
                "rc": rc,
                "reason": "stage04_failed",
            }

    # ── (3) evaluate: model sel_eff + Origami sel_eff per cell ─────────────
    # No active-learning decision machinery here -- stage 8 is a pure
    # "how does the deployed pipeline do on this dataset?" report.
    # We call `stage05_train.evaluate_per_cell_held_out_sel_eff`
    # directly (not stage 04b) so the log stays focused on sel_eff
    # numbers and never mentions retrain/split/pass.
    return _evaluate_dataset(
        name=name,
        path=path,
        out=out,
        train_dir=train_dir,
        enriched_dir=enriched_dir,
        rocm_libraries=rocm_libraries,
        smart_k=smart_k,
        min_cell_gemms=min_cell_gemms,
        arch=arch,
        device=device,
        prior_round_dirs=prior_round_dirs,
        origami_python_path=origami_python_path,
        workload_filter=workload_filter,
        scale_mode=scale_mode,
        quiet=quiet,
    )


def _evaluate_dataset(
    *,
    name: str,
    path: Path,
    out: Path,
    train_dir: Path,
    enriched_dir: Path,
    rocm_libraries: Optional[Path],
    smart_k: int,
    min_cell_gemms: int,
    arch: str,
    device: str,
    prior_round_dirs: List[Path],
    origami_python_path: Optional[Path],
    workload_filter: Optional[Dict[str, str]],
    quiet: bool,
    scale_mode: Optional[int] = None,
) -> Dict[str, Any]:
    """Direct, in-process evaluator: loads `models.pt`, walks the
    enriched chunks, computes deployed-pipeline sel_eff + Origami
    sel_eff per cell, writes one focused JSON, prints a short table.
    Replaces the old `stage04b --quiet` subprocess + decisions.json
    parsing.

    Why in-process: stage 8 already needs `torch` and the origami
    binding to load models + score; sub-processing stage 04b spawned
    a second Python that did the same setup, then printed
    decision-flavored output that doesn't belong in this report.
    Calling the helper directly keeps the log clean and avoids paying
    the import cost a second time per dataset.
    """
    # ── load models bundle (torch) ─────────────────────────────────────────
    models_pt = train_dir / "models.pt"
    if not models_pt.exists():
        ui.err("err", f"models bundle missing: {models_pt}")
        return {"name": name, "path": str(path), "rc": 1, "reason": "missing_models_pt"}
    import torch  # noqa: E402 -- defer torch import until eval phase

    bundle = torch.load(str(models_pt), map_location=device, weights_only=False)
    if not quiet:
        ui.ok(
            "models",
            f"loaded bundle: cells={len(bundle.get('models', {}))} "
            f"feats={len(bundle.get('x_names', []))} "
            f"smart_k={smart_k}",
        )

    # ── load origami (analytical scorer + hardware descriptor) ─────────────
    if rocm_libraries is None:
        ui.err(
            "err",
            "stage 8 needs --rocm-libraries-to-deploy to "
            "find the DAT library; cannot evaluate without it",
        )
        return {
            "name": name,
            "path": str(path),
            "rc": 1,
            "reason": "missing_rocm_libraries",
        }

    # Pick the origami python binding that actually has a built .so.
    # `--origami-python-path` (preferred, usually the training
    # checkout) wins, then the deploy checkout's source dir. The
    # deploy checkout often has the .cpp source but no built .so, so
    # blindly importing from there fails with
    # "No module named 'origami.origami'".
    def _has_built_so(pp: Path) -> bool:
        return bool(list(pp.glob("origami/origami.cpython-*.so")))

    origami_pp: Optional[Path] = None
    if origami_python_path is not None and _has_built_so(origami_python_path):
        origami_pp = origami_python_path
    else:
        cand = rocm_libraries / "shared/origami/python/src"
        if _has_built_so(cand):
            origami_pp = cand
    if origami_pp is None:
        ui.err(
            "err",
            "no origami python extension found in either "
            f"--origami-python-path ({origami_python_path}) or "
            f"{rocm_libraries / 'shared/origami/python/src'}. "
            "Build it via `cmake --build .../shared/origami "
            "--target origami_py` or pass an explicit "
            "--origami-python-path.",
        )
        return {
            "name": name,
            "path": str(path),
            "rc": 1,
            "reason": "no_built_origami_extension",
        }
    if str(origami_pp) not in sys.path:
        sys.path.insert(0, str(origami_pp))
    try:
        import origami as _ori_mod  # type: ignore
    except ImportError as e:
        ui.err("err", f"origami import failed ({e}); origami_pp={origami_pp}")
        return {
            "name": name,
            "path": str(path),
            "rc": 1,
            "reason": "origami_import_failed",
        }
    from stage04_convert_to_enriched_dataset import (
        load_origami_configs_by_dtype as _load_cfgs,
        _origami_hardware as _ori_hw,
    )

    dat_dir = rocm_libraries / "projects/hipblaslt/build/release/Tensile/library" / arch
    # scale_mode keeps the MX and non-MX F8 TN pools apart (they share
    # one dtype/layout key on gfx1250); None = load everything, as before.
    _loaded = _load_cfgs(str(dat_dir), quiet=quiet, scale_mode=scale_mode)
    configs_by_key = _loaded[0] if isinstance(_loaded, tuple) else _loaded
    # `_origami_hardware(device_idx, quiet)` -- DEVICE INDEX (int),
    # not the arch string. Origami uses HIP to look up SMs/CUs from
    # the live GPU; device 0 is fine for any single-GPU
    # measurement.
    # Pass the run's arch: _origami_hardware reports it and uses it for the
    # fallback path. Omitting it leaves the gfx950 default, which is exactly
    # the silent wrong-hardware case the arch parameter was added to remove.
    hw = _ori_hw(0, quiet=quiet, arch=arch)
    if hw is None:
        ui.err(
            "err",
            "origami hardware descriptor unavailable; cannot " "score Origami baseline",
        )
        return {"name": name, "path": str(path), "rc": 1, "reason": "no_origami_hw"}

    # ── cumulative split tree across prior rounds (cell routing) ───────────
    import stage05_train as _train  # type: ignore  # noqa: E402

    _train.origami = _ori_mod
    _train._ORIGAMI_OK = True
    from lib import subcells as _sc  # noqa: E402

    split_tree = (
        _sc.load_cumulative_split_tree([str(p) for p in prior_round_dirs])
        if prior_round_dirs
        else {}
    )

    # ── per-cell held-out sel_eff (model + origami) ────────────────────────
    if not quiet:
        ui.banner("Per-cell held-out sel_eff (model + Origami)", ui.C.BLUE)
    result = _train.evaluate_per_cell_held_out_sel_eff(
        enriched_csv_dir=str(enriched_dir),
        bundle=bundle,
        split_tree=split_tree,
        configs_by_key=configs_by_key,
        hw=hw,
        arch=arch,
        device=device,
        smart_k=int(smart_k),
        min_cell_gemms=int(min_cell_gemms),
        quiet=quiet,
        workload_filter=workload_filter,
        mx_block_size=_mx_block_size_for_scale_mode(scale_mode),
    )

    # ── persist + print focused per-cell + dataset summary ─────────────────
    per_cell_path = out / "per_cell_sel_eff.json"
    with per_cell_path.open("w") as f:
        json.dump(result, f, indent=2)

    per_cell = result.get("per_cell", [])
    summary_obj = result.get("summary", {})
    skipped_no_gemms = result.get("skipped_no_gemms", [])
    skipped_no_model = result.get("skipped_no_model", [])
    via_fallback = result.get("via_bnone_fallback", [])
    if not quiet:
        if per_cell:
            # Size the cell-name column to the longest label so the numeric
            # columns line up regardless of how deep the adaptive splits made
            # the labels (clamped so short-name runs aren't over-padded).
            _w = min(max(max((len(r["cell"]) for r in per_cell), default=24), 24), 80)
            ui.info(
                "eval",
                f"{'cell':<{_w}} {'n':>6} {'model_sel_eff':>14} "
                f"{'origami_sel_eff':>16}  notes",
            )
            for r in per_cell:
                note = ""
                if r.get("fallback"):
                    note = f"-> via {r.get('model_cell', '?')} (Bnone fallback)"
                ui.info(
                    "eval",
                    f"{r['cell']:<{_w}} {r['n_gemms']:>6d} "
                    f"{r['model_sel_eff']:>14.4f} "
                    f"{r['origami_sel_eff']:>16.4f}  {note}",
                )
            _g = summary_obj
            ui.ok(
                "eval",
                f"GEOMEAN (all cells, equal-weight)  "
                f"model={_g.get('model_geomean_sel_eff', float('nan')):.4f}  "
                f"origami={_g.get('origami_geomean_sel_eff', float('nan')):.4f}  "
                f"(n_cells={_g.get('n_cells_evaluated', 0)})",
            )
            ui.info(
                "eval",
                f"GEOMEAN (GEMM-weighted, most representative)  "
                f"model={_g.get('model_geomean_gemm_weighted', float('nan')):.4f}  "
                f"origami={_g.get('origami_geomean_gemm_weighted', float('nan')):.4f}",
            )
            ui.info(
                "eval",
                f"GEOMEAN (primary cells, own/ancestor model)  "
                f"model={_g.get('model_geomean_primary', float('nan')):.4f}  "
                f"origami={_g.get('origami_geomean_primary', float('nan')):.4f}  "
                f"(n_cells={_g.get('n_cells_primary', 0)})",
            )
            n_fb_cells = _g.get("n_cells_via_bnone_fallback", 0)
            if n_fb_cells:
                ui.info(
                    "eval",
                    f"GEOMEAN (Bany->Bnone fallback, batched OOD diag)  "
                    f"model={_g.get('model_geomean_fallback', float('nan')):.4f}  "
                    f"origami={_g.get('origami_geomean_fallback', float('nan')):.4f}  "
                    f"(n_cells={n_fb_cells})",
                )
        else:
            ui.warn(
                "eval",
                "no populated cell could be scored for this "
                "dataset (no GEMMs survived the workload filter)",
            )
        # No min-GEMM skipping anymore -- every populated cell is scored.
        if via_fallback:
            ui.info(
                "eval",
                f"{len(via_fallback)} batched (Bany) cell(s) scored via "
                f"their Bnone fallback model -- "
                f"e.g. {sorted(set(via_fallback))[:5]}",
            )
        if skipped_no_model:
            ui.info(
                "eval",
                f"UNSCORED (no own/ancestor/Bnone model exists): "
                f"{len(set(skipped_no_model))} cells -- "
                f"e.g. {sorted(set(skipped_no_model))[:5]}",
            )

    return {
        "name": name,
        "path": str(path),
        "rc": 0,
        "per_cell_json": str(per_cell_path),
        "n_evaluated_cells": summary_obj.get("n_cells_evaluated", 0),
        "model_geomean_sel_eff": summary_obj.get("model_geomean_sel_eff", float("nan")),
        "origami_geomean_sel_eff": summary_obj.get(
            "origami_geomean_sel_eff", float("nan")
        ),
        "summary": summary_obj,
    }


# ── dataset spec parsing ───────────────────────────────────────────────────


def _parse_dataset_specs(
    spec_args: Optional[List[str]], yaml_file: Optional[Path]
) -> List[Tuple[str, Path]]:
    """spec_args entries are `name=path` or `name:path`. yaml_file is
    a yaml list of `{name, path}` dicts (the form used in config
    yamls)."""
    out: List[Tuple[str, Path]] = []
    if yaml_file is not None:
        with yaml_file.open() as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise SystemExit(f"datasets yaml must be a LIST; got {type(data).__name__}")
        for entry in data:
            if isinstance(entry, list) and len(entry) == 2:
                name, p = entry[0], entry[1]
            elif isinstance(entry, dict):
                name = entry.get("name")
                p = entry.get("path")
            else:
                raise SystemExit(f"bad dataset entry: {entry!r}")
            if not name or not p:
                raise SystemExit(f"dataset entry missing name/path: " f"{entry!r}")
            out.append((str(name), Path(str(p))))
    for s in spec_args or []:
        sep = "=" if "=" in s else ":"
        if sep not in s:
            raise SystemExit(f"--dataset must be NAME=PATH or " f"NAME:PATH; got {s!r}")
        name, p = s.split(sep, 1)
        out.append((name.strip(), Path(p.strip())))
    return out


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage08: validate trained per-cell models on one "
        "or more external datasets (yaml / bench logs / "
        "enriched chunks)."
    )
    ap.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="dir containing models.pt + cells.json (e.g. " "<run>/round_<R>/stage05).",
    )
    ap.add_argument(
        "--rocm-libraries-to-deploy",
        type=Path,
        default=None,
        help="rocm-libraries checkout. Required when ANY "
        "dataset is a yaml (needed to run hipblaslt-"
        "bench) or block_*.log dir (needed to enrich).",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="`NAME=PATH` (or NAME:PATH). PATH is a "
        "bench yaml, a logs dir, or an enriched dir. "
        "Repeatable. Use --dataset-rocm-libraries to "
        "pair each dataset with the rocm-libraries "
        "checkout that produced it.",
    )
    ap.add_argument(
        "--dataset-rocm-libraries",
        action="append",
        default=None,
        help="`NAME=PATH` (or NAME:PATH). PATH points at "
        "the rocm-libraries checkout the dataset's "
        "bench was run against -- stage 4 enrichment "
        "uses ITS `Tensile/library/<arch>` + "
        "`device-library/MatchTable.yaml` so the "
        "bench's `--Solution index` / `--Solution "
        "name` lookups hit. Find the SHA on "
        "`hipBLASLt git version` at the top of any "
        "bench log. Repeatable. Datasets without a "
        "matching entry fall back to "
        "`--rocm-libraries-to-deploy`.",
    )
    ap.add_argument(
        "--datasets-yaml",
        type=Path,
        default=None,
        help="YAML file with a list of [name, path] entries "
        "or [{name, path}] dicts.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated GPU device IDs for the "
        "offline-tune bench (only used when a dataset "
        "is a yaml).",
    )
    ap.add_argument("--blocks-per-gpu", type=int, default=20)
    ap.add_argument("--sel-eff-threshold", default="0.95")
    ap.add_argument("--min-cell-gemms", type=int, default=20)
    ap.add_argument(
        "--smart-k",
        type=int,
        default=10,
        help="ALWAYS-ON candidate cap after the "
        "check_lds_capacity + is_kernel_feasible "
        "pre-filter. Walks survivors in INPUT "
        "ORDER and keeps the first K. Match this "
        "to the deploy runtime's ML_SMART_K "
        "(default 10) so stage 8's reported "
        "sel_eff reflects what bench actually "
        "picks.",
    )
    ap.add_argument(
        "--prior-round-dir",
        action="append",
        type=Path,
        default=[],
        help="round directories whose cells.json should "
        "be union'd into the cumulative split tree "
        "used for sub-cell routing. Pass one "
        "--prior-round-dir per round (round_0, "
        "round_1, ...).",
    )
    ap.add_argument(
        "--origami-python-path",
        type=Path,
        default=None,
        help="Directory containing the COMPILED origami "
        "python extension (`origami/origami.cpython-"
        "*.so`). The deploy checkout has the binding "
        "sources but rarely a built `.so`; pass the "
        "training checkout's `shared/origami/python/"
        "src` here. When omitted, stage 8 falls "
        "back to `--rocm-libraries-to-deploy`'s "
        "shared/origami/python/src.",
    )
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument(
        "--device", default="cpu", help="torch device for the student inference pass."
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="cell config yaml whose `probe:` section drives "
        "per-shape iters / cold_iters / "
        "skip_slow_solution_ratio calibration when a "
        "dataset is a raw bench yaml. With it, stage 8 "
        "runs a stage02 probe (like the training "
        "pipeline) instead of dumping a fixed iters; "
        "without it, it falls back to fixed iters.",
    )
    # Workload filter: restrict the dataset to GEMMs matching the
    # config's dtype + transpose. A multi-dtype production trace
    # (fp16 + bf16 + fp8, mixed transposes) would otherwise count
    # every GEMM into the per-cell sel_eff -- inflating the cell
    # GEMM count past the workload-under-test (e.g. BBS_TN's bf16
    # cases) and punishing the trained model for picks on shapes
    # it never saw. Each flag is OPTIONAL; missing flags mean
    # "don't filter on that field".
    ap.add_argument(
        "--filter-a-type",
        default=None,
        help="Required value for the bench row's " "`a_type` column (e.g. `bf16_r`).",
    )
    ap.add_argument("--filter-b-type", default=None)
    ap.add_argument("--filter-c-type", default=None)
    ap.add_argument("--filter-d-type", default=None)
    ap.add_argument(
        "--filter-compute-type", default=None, help="With or without the `c_` prefix."
    )
    ap.add_argument("--filter-transA", default=None, help="`T` or `N`.")
    ap.add_argument("--filter-transB", default=None)
    # None == no filter == the legacy unfiltered pool. See lib/dat.py.
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA/scaleB mode of the run under "
        "test (0=none, 1=Scalar, 3=Block_32_UE8M0/MX); "
        "restricts the .dat libraries used for enrichment "
        "and for the Origami baseline pool",
    )
    ap.add_argument(
        "--ml-weights-path",
        type=Path,
        default=None,
        help="explicit path to tilewright_weights.bin. "
        "When set, this stage exports "
        "`TILEWRIGHT_WEIGHTS=<path>` AND `TILEWRIGHT_DIAG=1` "
        "to its environment so every subprocess "
        "(stage03 bench, stage04 enrich, stage04b "
        "evaluator) inherits the override. Without "
        "it, origami's loader searches `./` and "
        "`/opt/rocm/share/origami/` -- usually means "
        "the just-staged .bin isn't found and the "
        "ML path silently falls back to analytical "
        "Origami. The orchestrator's `--mode "
        "validate` derives this automatically.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    datasets = _parse_dataset_specs(args.dataset, args.datasets_yaml)
    if not datasets:
        ui.err(
            "err",
            "no datasets provided -- pass --dataset NAME=PATH "
            "or --datasets-yaml FILE",
        )
        return 1

    # Parse `--dataset-rocm-libraries NAME=PATH` into a name -> Path
    # dict. Datasets without an entry fall back to
    # --rocm-libraries-to-deploy at enrichment time (handled inside
    # _run_one_dataset). Format mirrors --dataset so the orchestrator
    # can pair them up trivially.
    bench_libs_by_name: Dict[str, Path] = {}
    for spec in args.dataset_rocm_libraries or []:
        sep = "=" if "=" in spec else ":"
        if sep not in spec:
            ui.err(
                "err",
                f"--dataset-rocm-libraries must be NAME=PATH or "
                f"NAME:PATH; got {spec!r}",
            )
            return 1
        nm, pth = spec.split(sep, 1)
        bench_libs_by_name[nm.strip()] = Path(pth.strip())

    # Plumb the ML recommender env vars through the WHOLE process tree
    # we're about to spawn (stage03 bench / stage04 enrich / stage04b
    # evaluator all inherit os.environ unless they pass an explicit
    # `env=` to subprocess.Popen). Setting these here means every
    # nested origami call -- C++ side via hipblaslt-bench AND Python
    # side via the `origami` import -- picks up the right .bin and
    # prints the load-diag line so the operator can audit the run.
    # F6: we no longer export TILEWRIGHT_WEIGHTS. The deployed origami
    # runtime auto-discovers tilewright_weights.bin (next to libOrigami /
    # in shared/tilewright/backends/hipblaslt/data) when the ML path is active, so the override is
    # unnecessary. We keep --ml-weights-path purely to (a) sanity-check the
    # deployed .bin exists and (b) turn on TILEWRIGHT_DIAG so the load is auditable.
    if args.ml_weights_path is not None:
        if not args.ml_weights_path.exists():
            ui.err(
                "err", f"--ml-weights-path does not exist: " f"{args.ml_weights_path}"
            )
            return 1
        os.environ.setdefault("TILEWRIGHT_DIAG", "1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        ui.banner("stage08  validate selection efficiency")
        ui.info("cfg", f"train_dir       : {args.train_dir}")
        ui.info("cfg", f"out_dir         : {args.out_dir}")
        if args.rocm_libraries_to_deploy:
            ui.info("cfg", f"rocm_libraries  : " f"{args.rocm_libraries_to_deploy}")
        ui.info("cfg", f"datasets        : " f"{[(n, str(p)) for n, p in datasets]}")

    # Build workload filter from individual CLI flags. Keep only
    # the keys actually set on the command line; unset keys mean
    # "no filter on that field".
    workload_filter = {
        k: v
        for k, v in {
            "a_type": args.filter_a_type,
            "b_type": args.filter_b_type,
            "c_type": args.filter_c_type,
            "d_type": args.filter_d_type,
            "compute_type": args.filter_compute_type,
            "transA": args.filter_transA,
            "transB": args.filter_transB,
        }.items()
        if v is not None
    }
    if workload_filter and not args.quiet:
        ui.info("cfg", f"workload_filter : {workload_filter}")

    summaries: List[Dict[str, Any]] = []
    n_failed = 0
    for name, path in datasets:
        s = _run_one_dataset(
            name,
            path,
            args.out_dir,
            train_dir=args.train_dir,
            rocm_libraries=args.rocm_libraries_to_deploy,
            bench_rocm_libraries=bench_libs_by_name.get(name),
            devices=args.devices,
            blocks_per_gpu=args.blocks_per_gpu,
            sel_eff_threshold=args.sel_eff_threshold,
            min_cell_gemms=args.min_cell_gemms,
            arch=args.arch,
            device=args.device,
            smart_k=args.smart_k,
            prior_round_dirs=list(args.prior_round_dir or []),
            origami_python_path=args.origami_python_path,
            workload_filter=workload_filter or None,
            config_yaml=args.config_yaml,
            scale_mode=args.scale_mode,
            quiet=args.quiet,
        )
        summaries.append(s)
        if s.get("rc") != 0:
            n_failed += 1

    out_json = args.out_dir / "summary.json"
    with out_json.open("w") as f:
        json.dump({"datasets": summaries}, f, indent=2)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN if n_failed == 0 else ui.C.YELLOW)
        for s in summaries:
            mse = s.get("model_geomean_sel_eff")
            ose = s.get("origami_geomean_sel_eff")
            n_eval = s.get("n_evaluated_cells", 0)
            if mse is None or ose is None:
                ui.warn(
                    "dataset",
                    f"{s['name']:20s}  rc={s.get('rc')}  "
                    f"reason={s.get('reason', '-')}",
                )
            else:
                delta = (mse - ose) * 100.0
                ui.ok(
                    "dataset",
                    f"{s['name']:20s}  n_cells={n_eval}  "
                    f"model={mse*100:6.2f}%  origami={ose*100:6.2f}%  "
                    f"(delta={delta:+5.2f} pp)",
                )
        ui.ok("done", f"summary.json -> {out_json}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
