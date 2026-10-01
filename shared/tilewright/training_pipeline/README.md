# tilewright training pipeline — per-cell ML kernel-selection recommender

A self-contained pipeline that trains one ML model per cell of a
96-cell categorical grid over GEMM problem space, then deploys the
trained set into a rocm-libraries checkout so the runtime can re-rank
Origami's top-K with the ML pick.

Everything the pipeline needs lives in `training_pipeline/`: every
helper in `lib/`, every stage in `stages/`, every per-run artifact
under `runs/`.

## Location, running and environment

The pipeline lives at `shared/tilewright/training_pipeline/` inside
rocm-libraries. It trains the weights that the tilewright runtime
(`shared/tilewright/`) loads; stage06 deploys them under
`shared/tilewright/backends/hipblaslt/weights/`.

Launch gfx1250v0 runs through `scripts/run_gfx1250v0.sh`. It reads the
variables below (exporting defaults for the bench and deploy roots),
unsets `HIPBLASLT_TENSILE_LIBPATH`, points
`LD_LIBRARY_PATH` at the bench root's libraries and `PYTHONPATH` at
origami's Python binding, restricts the candidate set to Origami
kernels, and passes every remaining argument to `run_orchestrator.py`
(the script header explains why each setting matters):

```bash
# round_0 + every active round (`full.total_rounds` from the config)
scripts/run_gfx1250v0.sh configs/<config>.yaml --mode full

# stages 7 + 8 only (post-deploy validation) on an existing run
scripts/run_gfx1250v0.sh configs/<config>.yaml --mode validate --run-dir <dir>
```

The script runs the orchestrator from `training_pipeline/`, so relative
paths (the config, `--run-dir`) resolve against that directory.

| Variable | Meaning | Default when launched via the script |
|---|---|---|
| `TILEWRIGHT_BENCH_ROOT` | Built rocm-libraries checkout, or a pinned snapshot of one, whose `hipblaslt-bench` and Tensile library the run benchmarks against. `PIN_DIR` is accepted as an alias. | the containing checkout |
| `TILEWRIGHT_DEPLOY_ROOT` | Checkout that receives the trained weights (stage06). | the containing checkout |
| `TILEWRIGHT_DATASETS` | Directory of stage08 dataset yamls. Only needed if the config lists datasets. | unset |
| `TILEWRIGHT_EXTRA_LD_LIBRARY_PATH` | Appended (never prepended) to `LD_LIBRARY_PATH`, for runtime libraries the deploy checkout's bench needs when it was built against a different ROCm. | unset |
| `TILEWRIGHT_ORIGAMI_BUILD` | Out-of-tree origami build providing a `liborigami` that matches the Python binding. | `training_pipeline/.origami_build_gfx1250` |

Configs refer to host-specific paths through these variables
(`${TILEWRIGHT_BENCH_ROOT}`, ...). `run_orchestrator.py` expands
`${VAR}`, `$VAR` and a leading `~` in every string of the config and
refuses to start if a referenced variable is unset, so a literal
`${VAR}` never reaches a stage as a path. When calling
`run_orchestrator.py` directly (see "Orchestrator usage" below), export
the variables your config references first.

For an ASIC-revision arch such as `gfx1250v0`, a build tree keeps that
library at `build/release/Tensile-gfx1250v0/library/gfx1250v0`, but
hipblaslt-bench's revision promotion only loads it from
`build/release/Tensile/library/gfx1250v0`. Link it there once before
benching against that tree (a pinned snapshot should carry the link
too); the orchestrator stops and prints the command if it is missing:

```bash
B=$TILEWRIGHT_BENCH_ROOT/projects/hipblaslt/build/release
cp -al $B/Tensile-gfx1250v0/library/gfx1250v0 $B/Tensile/library/gfx1250v0
```

Run outputs go to `training_pipeline/runs/` (gitignored). Never commit
them: the bench logs, metrics, manifests and reports in a run directory
carry measured performance numbers.

## Pipeline overview

```
┌───────────────────────────────────────────────────────────────────┐
│  stage01_generate_OOB_shapes        shapes.yaml + categorization  │
├───────────────────────────────────────────────────────────────────┤
│  stage02_probe                      shapes_probed.yaml            │
│      (per-shape iters/cold_iters/skip_slow_solution_ratio,        │
│       calibrated from a fast probe pass on every device;          │
│       skipped when `bench.adaptive.enabled` is true)              │
├───────────────────────────────────────────────────────────────────┤
│  stage03_load_balance_offline_tuning   logs/block_NNNN_*.log      │
│      (dynamic per-GPU work queue + per-block crash recovery)      │
├───────────────────────────────────────────────────────────────────┤
│  stage04_convert_to_enriched_dataset   chunk_*.csv                │
│      (bench logs -> MatchTable + .dat enrichment + skip rows)     │
├───────────────────────────────────────────────────────────────────┤
│  stage05_train                      models.pt + cells.json        │
│      (per-cell LightGBM teacher + GenericTwoTower distill student)│
├───────────────────────────────────────────────────────────────────┤
│  stage04b_iterate_active_learning   decisions.json + splits.json  │
│      (ACTIVE ROUNDS ONLY -- not run in round 0.                   │
│       Conceptually "stage 4.5"; the `b` suffix encodes "between   │
│       4 and 5" without a decimal in the filename. Validates the   │
│       PRIOR round's model on this round's NEW unknown shapes via  │
│       the DEPLOYED pipeline (Origami top-K + ML re-rank);         │
│       per-cell pass / retrain / split decision.)                  │
├───────────────────────────────────────────────────────────────────┤
│  stage05_train                                   (already above)  │
├───────────────────────────────────────────────────────────────────┤
│  stage06_deploy_to_origami    <liblogic>.tilewright.bin           │
│      (write trained models into a rocm-libraries CHECKOUT;        │
│       drops the .bin into shared/tilewright/backends/hipblaslt/   │
│       weights/<arch>/<arch>/; optionally relinks libOrigami +     │
│       hipblaslt-bench and copies the python binding)              │
├───────────────────────────────────────────────────────────────────┤
│  stage07_optional_validate_selection_time  (real-GPU; optional)   │
│      (hipblaslt-bench with TENSILE_DB=0x10000, ML-on vs ML-off,   │
│       end-to-end selection-time comparison; optional C++/Python   │
│       pick parity)                                                │
├───────────────────────────────────────────────────────────────────┤
│  stage08_optional_validate_selection_efficiency  (optional)       │
│      (Origami offline-tune any yaml-input dataset, then re-rank   │
│       with the ML recommender; sel_eff for both, multi-dataset)   │
└───────────────────────────────────────────────────────────────────┘
```

Stages 1–6 run automatically inside the orchestrator. Stages 7 + 8
are optional, post-deploy validation steps for a real GPU host: run
them with `--mode validate --run-dir <dir>`, or by hand (see
"Optional validation" below). A training run also runs them at the
end when its config lists `stage07.bench_yamls`,
`stage07.parity_sample_trained` or `stage08.datasets`.

## Terminology

| Term | Meaning |
|---|---|
| **cell** | one of the 96 categorical regions of (M, N, K, B) tier space (or a sub-cell, after splitting). Each leaf of the cumulative split tree is one trained model — the invariant `1 cell = 1 trained model` is preserved at every depth. |
| **config** | the workload definition (arch + dtype + layout + runtime settings). Identified by `config_id` inside `configs/<name>.yaml`. NOT the same thing as a grid cell. |
| **round** | one iteration of the active-learning loop: gen new shapes (excluding everything tested in any prior round), bench, enrich, validate, optionally retrain / split / deploy. Lives under `runs/<config_stem>_<ts>/round_<idx>/`. |
| `rocm_libraries_to_deploy` | path to a rocm-libraries CHECKOUT that stage06_deploy writes the trained weights into (`<repo>/shared/tilewright/backends/hipblaslt/weights/...`) and can optionally relink. |

## Grid (96 cells)

* **M tier** Tiny (≤32) · Small (≤128) · Mid (≤512) · Large (>512)
* **N tier** Tiny / Small / Mid / Large (same)
* **K tier** TinyK (≤32) · MidK (≤512) · LargeK (>512)
* **B tier** Bnone (B=1) · Bany (B>1)

Every synthesized shape must pass `is_rotating512_safe`
(`lib/shapes.py`), which keeps it benchable at `rotating: 512`: the
total device footprint (A+B+C+D, plus MX block scales and the rotating
overhead) must fit a per-arch HBM budget, FLOPs must fit a cost cap,
no 2D matrix may overflow int32 indexing, and batched (B>1) shapes get
tighter working-set, per-dimension and FLOP caps. An arch-specific
exclusion of known-faulting shape regions (`is_known_faulting`) applies
on top. A cell the filters leave empty has no model; Stage05 reports
`cells_unpopulated` explicitly so the operator knows which problem
regions have no model at all.

## Run-dir layout

```
runs/<config_stem>_<YYYYMMDD_HHMMSS>/
    config_used.yaml
    orchestrator.log
    round_0/                    INITIAL round (one-shot training)
        stage01/  stage01.log   gen shapes
        stage02/  stage02.log   probe (skipped when bench.adaptive
                                is enabled)
        stage03/  stage03.log   offline tuning (bench)
        stage04/  stage04.log   enrich
        stage05/  stage05.log   TRAIN initial models
        stage06/  stage06.log   (ONLY if --deploy is passed on the CLI;
                                 auto-deploy is GATED on stage04b's
                                 held-out sel_eff, which doesn't exist
                                 in round 0 -- stage04b runs from
                                 round 1 onwards. So unless the
                                 operator explicitly forces a deploy
                                 of the initial training-fit model,
                                 round 0 ends after stage 5.)
                                NB: NO stage04b in round 0 -- there's
                                no prior model to validate and no
                                held-out shapes (every GEMM was
                                training data).
    round_1/                    ACTIVE round (only in active / full mode)
        stage01/  stage01.log   gen NEW shapes (deduped vs every
                                prior round)
        stage02/  stage02.log   probe
        stage03/  stage03.log   offline tuning (bench)
        stage04/  stage04.log   enrich
        stage04b/ stage04b.log  VALIDATE round_0's model on round_1's
                                NEW shapes -> decide pass/retrain/split
                                (logical stage 4.5; runs BEFORE stage05
                                in the round)
        stage05/  stage05.log   RETRAIN failing cells with cross-round
                                data augmentation; fresh weights
        stage06/  ...           (only if deploy triggered)
    round_2/ ...                same pattern as round_1
    round_R/ ...
    validate/                   stage 7 + 8 output (--mode validate or
                                the end-of-run auto-validate)
    report.html                 end-of-run HTML report
```

Stage numbers encode the round's DATA-FLOW graph, not the temporal
order within a round. In active rounds, `stage04b` runs **before**
`stage05` because stage04b's retrain decision drives stage05's
`--only-cells` list. The `b` suffix on `04b` -- not `045` -- is
deliberate: it encodes "between 4 and 5" without a decimal in the
filename (`stage045` reads as "stage forty-five", which is wrong).
`stage04b` depends on a model from a PRIOR round's `stage05`;
`stage05` of this round in turn depends on `stage04b`'s decision.

Heavy bench / enrich / train artifacts (`runs/`) are gitignored.

## Active-learning policy (stage04b)

* Round 0 trains every populated cell.
* Round R ≥ 1 validates the prior round's model on this round's new
  shapes and decides per cell:
  - `sel_eff_new >= --sel-eff-threshold` (default 0.95, or the
    string `"origami"` meaning "match the per-cell Origami sel_eff")
    → **PASS**.
  - `sel_eff_new < --split-floor` (default 0.85, or `"origami"`) **OR**
    (round_idx ≥ 2 AND improvement < `--split-min-improvement`)
    → **SPLIT CANDIDATE**.
  - else → **RETRAIN**.
  Split candidates are demoted to retrain until the cell has been
  retrained `--split-after-attempts` times (default 2) — data
  augmentation gets first crack.
* When a cell does split: axis is chosen by **CART variance
  reduction** on per-GEMM log_ratios from the prior model on the new
  data, restricted to the cell's widest-tier axes (**Large** M/N,
  **LargeK** K) -- we never carve a cell whose tiers are all finite
  (Tiny/Small/Mid M and N, TinyK/MidK K; those finite-population
  spaces are better served by data augmentation). Threshold is the
  **log-median** of the cell's axis values (outlier-resistant). Tied
  gains break in favour of an axis NOT yet in this cell's split path
  (encourages isotropic refinement; no silent M-bias). A hard
  `max_split_depth` (default 6) caps the tree.

## Deployment (stage06_deploy)

stage06_deploy packs every trained per-cell model + the cumulative split
tree into a single weights `.bin` (magic `MLREC_v1`; the format is
specified in `shared/tilewright/src/tilewright/model.cpp`) and drops it
into a rocm-libraries CHECKOUT. The checkout is **assumed to already
be built** (kernels live under
`projects/hipblaslt/build/release/Tensile/library/<arch>/`) and the
**tilewright runtime is assumed to already be present** in
`shared/tilewright/` (engine) and `shared/tilewright/backends/hipblaslt/`
(hipBLASLt adapter); stage06_deploy NEVER edits C++ source.

What gets staged / written:

```
<round>/stage06/staged_for_deploy/          run output only
    shared/tilewright/backends/hipblaslt/weights/<arch>/<arch>/
        <liblogic>.tilewright.bin       feature-catalog hash, split
                                        tree, per-cell whitening
                                        stats + MLP weights
        ml_recommender.manifest.json    readable index (cells + splits)
        ml_recommender.summary.txt      human banner; ASCII split tree
    shared/origami/ML_RECOMMENDER_INSTALL.md
                                        operator notes for this drop

<repo>/shared/tilewright/backends/hipblaslt/weights/<arch>/<arch>/
    <liblogic>.tilewright.bin           (only with apply_patch: true)
    tilewright_index                    library-logic stem -> .bin;
                                        merged, so models for other
                                        dtypes/layouts are kept
```

The `.bin` is named after the Origami UserArgs library-logic file for
the config's dtype/layout (or, failing that, the first matching library
stem); with no matching library at all it falls back to
`tilewright_weights.bin` with no index entry. Only the `.bin` and its
`tilewright_index` entry go into the checkout: the manifest and summary
carry per-cell performance numbers, so they stay in the run output
together with the install notes.

The hipBLASLt dispatch is **already in place**: when
`Debug::Instance().useTilewright()` is true (env var
`TENSILE_USE_TILEWRIGHT=1`) and the library has a per-library model
(loaded via `tilewright_index`), tensilelite's `PredictionLibrary.hpp`
ranks the candidates with `tilewright::hipblaslt::rank_configs`;
otherwise it calls plain `origami::rank_configs`. stage06_deploy does
NOT touch any tensilelite header.

If a `.bin` can't load (wrong magic or version, truncated or missing
file) the runtime falls back to plain Origami for that library — no
crash, no garbage ranking.

### Triggering deployment

stage06_deploy fires automatically AFTER stage05 (retrain) whenever ANY
of:

1. `--deploy` was passed on the orchestrator CLI (forces every round,
   round 0 included — this is the **only** way to deploy round 0's
   training-fit initial model),
2. this round's stage04b **held-out** geomean sel_eff (weighted by
   n_eval over `decisions.json` per-cell `sel_eff_new`) is
   ≥ `deploy.target_sel_eff` — **active rounds only**, since round 0
   has no stage04b,
3. this is the LAST round of `--mode full` AND
   `deploy.on_final_round: true` — **active rounds only**, for the
   same reason.

The auto-triggers (2) and (3) are deliberately disabled in round 0:
round 0's `global_sel_eff` (stage05 metric) is **training-set**
accuracy, not generalization. Shipping based on that risks shipping
overfit weights; the active loop's whole point is to get a held-out
measurement before deploying.

### Two-step deploy (and an optional rebuild)

stage06_deploy ALWAYS stages the bundle under
`<round>/stage06/staged_for_deploy/`.
Whether it then writes into the rocm-libraries repo, and whether it
relinks the consumers, is driven by three independent knobs:

| `deploy.rocm_libraries_to_deploy` | `deploy.apply_patch` | `deploy.build_after_apply` | Behaviour |
|---|---|---|---|
| `null` | (any) | (any) | Stage only. Operator copies into the repo manually. |
| `/path/to/repo` | `false` | (any) | Stage only. Inspect, then re-run with `apply_patch: true`. |
| `/path/to/repo` | `true` | `false` | Stage AND copy the `.bin` (+ its `tilewright_index` entry) into `<repo>/shared/tilewright/backends/hipblaslt/weights/<arch>/<arch>/`. **No rebuild** — the operator runs the build commands below manually. |
| `/path/to/repo` | `true` | `true` | All of the above, then: (a) `cmake --build <repo>/shared/origami/build --parallel <jobs>` (plus its `_pyorigami` target) — relinks `libOrigami` and the origami Python binding; (b) `make -C <repo>/projects/hipblaslt/build/release/clients [-j <jobs>]` — relinks `hipblaslt-bench` against the new `libOrigami`; (c) `cmake --build <repo>/projects/hipblaslt/build/release --target tilewright_colocate` — copies the weights next to the runtime's Tensile library, where the loader looks for them; (d) copies the new `origami.cpython-*.so` into the venv's `site-packages/origami/` so stage08_validate_sel_eff's offline-eval Python path picks it up. Build output is tee'd to `<round>/stage06/build.log`. |

Kernels are **not** rebuilt — that's a one-time cost the operator
pays via `./install.sh -c` when setting up the deploy checkout.
After that, every model update is just a `.bin` drop + a fast relink
(libOrigami is small and `hipblaslt-bench` is a single binary).

### `TENSILE_PREDICTION_LIB=1` on every bench call; `TENSILE_USE_TILEWRIGHT=1` enables ML

Every `hipblaslt-bench` invocation in this pipeline — stage02 probe,
stage03 offline tuning, stage07 selection-time measurement, and any
in-line bench triggered by stage08_validate_sel_eff — runs with
`TENSILE_PREDICTION_LIB=1` in the environment. That env var makes
Tensile route `MasterSolutionLibrary::findTopSolutions` through
`ProblemPredictionLibrary`, which is the path the ML recommender
hooks into. Without it, hipblaslt-bench takes the legacy
Equality/Range-only path and we never exercise the deployed model.
The flag lives in one place (`lib/bench_env.py`) so it's impossible
to forget when adding a new bench call.

On top of `TENSILE_PREDICTION_LIB=1`, `TENSILE_USE_TILEWRIGHT=1`
specifically activates the tilewright ML scoring path (vs the plain
analytical Origami path). stage07_validate_sel_time sets it for the
ML-on run only, to compare selection time of the two configurations;
stage03's offline tuning deliberately leaves it unset (we want plain
Origami there to build the training labels).

## Optional validation (stages 7 + 8)

Both need the deployed checkout on a real GPU host. The orchestrator
runs them with `--mode validate --run-dir <dir>`, taking every knob
from the config and the latest round's models (`--from-round R` picks
another round; `--from-stage 8` skips stage 7), and at the end of a
training run whose config lists them. They can also be invoked by
hand, as below.

### stage07 — selection TIME

Measures wall-clock selection time end-to-end (the whole
`Caching → Equality → Range → Prediction` chain in `MasterSolutionLibrary`):

```bash
python stages/stage07_optional_validate_selection_time.py \
    --rocm-libraries-to-deploy /path/to/deployed/rocm-libraries \
    --yaml /path/to/workload_1.bench.yaml \
    --yaml /path/to/workload_2.bench.yaml \
    --out-dir results/seltime_$(date +%Y%m%d) \
    --devices 0
```

For each input yaml it runs `hipblaslt-bench` twice with
`TENSILE_DB=0x10000`, once with `TENSILE_USE_TILEWRIGHT=1` (ACTIVE)
and once without it (Origami-only). Both runs keep
`TENSILE_PREDICTION_LIB=1`. The bench tool emits one
`Solution selection time: <us> us` line per call — that's the
whole-call `findTopSolutions` span; there's no per-stage decomposition
in this code path. stage07_validate_sel_time parses every such line,
optionally drops the first per-yaml call (which carries the one-shot
library-init cost, far above the steady-state cost), and writes
`timings_summary.json` with the ML-overhead delta in mean / p50 / p99
microseconds.

### stage08 — selection EFFICIENCY

Validates per-cell `sel_eff` of the trained model AND of plain Origami
against one or more external datasets:

```bash
python stages/stage08_optional_validate_selection_efficiency.py \
    --train-dir runs/<config>_<ts>/round_<R>/stage05 \
    --rocm-libraries-to-deploy /path/to/deployed/rocm-libraries \
    --dataset oob=/path/to/oob_workload.bench.yaml \
    --dataset traces=/path/to/workload_logs_dir \
    --out-dir results/seleff_$(date +%Y%m%d)
```

Each `--dataset NAME=PATH` is auto-classified by stage08_validate_sel_eff:

| PATH | What stage08_validate_sel_eff does |
|---|---|
| `*.yaml` | Offline-tunes the workload via hipblaslt-bench (Origami's ranking), then enriches the logs, then evaluates. |
| dir of `block_*.log` | Skips bench, runs stage04 enrich, then evaluates. |
| dir of `chunk_*.csv` | Skips both bench and enrich; goes straight to per-cell evaluation. |

Multiple datasets can also be passed via `--datasets-yaml <file>`:

```yaml
- name: oob
  path: /path/to/oob_workload.bench.yaml
- name: traces
  path: /path/to/workload_logs_dir
```

Output: per-dataset `decisions.json` (one row per cell with both
model and Origami sel_eff) plus a top-level `summary.json` comparing
every dataset.

## Orchestrator usage

These call the orchestrator directly; export the variables the config
references first. For gfx1250v0, use `scripts/run_gfx1250v0.sh
<config>` instead, which takes the same arguments after the config.

```bash
# round_0 only (initial seed + train; deploy if target reached):
python run_orchestrator.py --config configs/gfx950_bbs_tn.yaml --mode initial

# ONE active round on top of an earlier run's round_0:
python run_orchestrator.py --config configs/gfx950_bbs_tn.yaml \
    --mode active --run-dir runs/<earlier_run>
    # (every `round_*/` already under <earlier_run> becomes a prior;
    #  the new round is appended as round_{N})

# every round in one shot (round_0 + active rounds up to
# `full.total_rounds` from the config):
python run_orchestrator.py --config configs/gfx950_bbs_tn.yaml --mode full

# resume a crashed full run:
python run_orchestrator.py --config configs/gfx950_bbs_tn.yaml --mode full \
    --run-dir runs/<existing> --from-round 2 --from-stage 4

# post-deploy validation only (stages 7 + 8) on an existing run:
python run_orchestrator.py --config configs/gfx950_bbs_tn.yaml --mode validate \
    --run-dir runs/<existing>
```

Each stage can be invoked standalone (helpful for debugging):

```bash
python stages/stage01_generate_OOB_shapes.py --help
python stages/stage02_probe.py --help
python stages/stage03_load_balance_offline_tuning.py --help
python stages/stage04_convert_to_enriched_dataset.py --help
python stages/stage05_train.py --help
python stages/stage04b_iterate_active_learning.py --help
python stages/stage06_deploy_to_origami.py --help
python stages/stage07_optional_validate_selection_time.py --help
python stages/stage08_optional_validate_selection_efficiency.py --help
```

## Layout

```
training_pipeline/
    run_orchestrator.py
    requirements.txt
    configs/
        gfx950_bbs_tn.yaml      one config per workload
    stages/
        stage01..stage08  (with stage04b as the active-round validate
                           step; logical stage 4.5)
    lib/
        arch.py         ASIC revision -> compiler target (gfx1250v0 -> gfx1250)
        grid.py         96-cell tier definitions + labels
        shapes.py       Shape dataclass + rotating-512 filter + synth
        bench_yaml.py   bench-yaml writer
        bench_probe.py  per-shape iters / skip ratio probe
        bench_env.py    common env (TENSILE_PREDICTION_LIB=1, etc.)
                          for every hipblaslt-bench invocation
        matchtable.py   sol_idx_global -> (yaml_path, local_idx)
        dat.py          .dat msgpack reader + kernel-param extraction
        features.py     generic v3 feature catalog (Q / I / X)
        model.py        GenericTwoTower + LightGBM LambdaRank teacher
        subcells.py     adaptive cell-splitting tree + leaf assignment
        origami.py      origami python binding loader + baseline helper
        ui.py           ANSI banner / colored progress / dur formatting
        fs.py           atomic write, JSON read/write, chunk paths
    scripts/
        run_gfx1250v0.sh           gfx1250v0 launcher (environment setup)
        make_report.py             self-contained HTML report for a run
        collect_bench_failures.py  list hipblaslt-bench failures in
                                   stage03 logs
        experiments/               rank_configs selection-time microbench
    tests/                      standalone kernel-param / model-pick checks
    runs/                       # gitignored; per-run artifacts
```
