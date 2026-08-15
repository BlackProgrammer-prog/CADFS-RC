# CADFS-RC

**Confidence-Aware Dynamic Focal Search with Risk Control**

CADFS-RC is a research framework for bounded-suboptimal path planning with learned guidance. It combines a C++17 search engine with a Python pipeline for synthetic dataset generation, optimal cost-to-go labeling, deep-ensemble training, uncertainty calibration, validation-only tuning, paired benchmarking, statistical analysis, and publication-ready figures.

The repository contains both the original **CADFS-RC** method and the newer **CADFS Next** architecture. CADFS Next adds heterogeneous experts, intra- and inter-expert uncertainty, composite confidence, interchangeable focal-width controllers, and a final safety projection.

> **Scope:** the current implementation targets grid maps in MovingAI format. It is a research prototype, not a complete road-navigation or Google-Maps-like application.

The evidence, mandatory baselines, ablations, timing rules, and paper-claim
gates are defined in [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).

## Table of contents

- [Core idea](#core-idea)
- [CADFS Next](#cadfs-next)
- [Bounded-suboptimality contract](#bounded-suboptimality-contract)
- [Implemented methods](#implemented-methods)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Build and test](#build-and-test)
- [Python environment](#python-environment)
- [Complete pipeline](#complete-pipeline)
- [Benchmark commands](#benchmark-commands)
- [Analysis and figure commands](#analysis-and-figure-commands)
- [Complete command reference](#complete-command-reference)
- [Data splits](#data-splits)
- [Generated artifacts](#generated-artifacts)
- [Python API example](#python-api-example)
- [Reproducibility protocol](#reproducibility-protocol)
- [Known limitations](#known-limitations)

## Core idea

Let `h_a(n)` be an admissible anchor heuristic:

```text
f_a(n) = g(n) + h_a(n)
f_min  = min_{n in OPEN} f_a(n)
```

At search step `t`, the algorithm builds a dynamic focal set:

```text
FOCAL_t = {n in OPEN | f_a(n) <= w_t * f_min}
```

The learned model ranks candidates and estimates uncertainty, while a controller selects the focal width `w_t`. Before use, the width passes through a safety projection:

```text
1 <= w_t <= W
```

This separation is deliberate: learned guidance can improve search order, but the admissible anchor and the projected focal width retain the bounded-search foundation.

### Legacy CADFS-RC

The original CADFS-RC controller combines confidence `C_t`, structural risk `R_t`, and fallback/instability signal `F_t`:

```text
w_t = 1 + (W - 1) * C_t * (1 - R_t) * (1 - F_t)
```

The engine supports multiplicative, linear, fixed, and tuned-fixed legacy controller modes, along with risk, confidence, and fallback ablations.

## CADFS Next

CADFS Next replaces the single guidance view with three heterogeneous experts:

- **Geometric expert:** wraps the learned deep ensemble and uses local occupancy plus goal-relative features.
- **Topological expert:** represents connectivity and structural properties of the grid.
- **Goal-distance expert:** provides a deterministic goal-relative distance signal.

Their outputs are fused as:

```text
H_fused(n) = sum_i alpha_i * H_i(n)
```

Three interchangeable learned-guidance backends are available:

- `EnsembleGuidance`: the complete independent CNN ensemble;
- adaptive `EnsembleGuidance`: evaluates two members first and runs the
  remainder only above a calibrated variance threshold;
- `FastEnsembleGuidance`: a distilled shared MLP backbone with three
  uncertainty heads, intended for final low-latency C++ inference.

All CADFS Next node evaluations are cached per query. Telemetry separates node
evaluations, member/head evaluations, cache hits, cache-hit rate, and model
time. This makes it possible to report where runtime is actually spent rather
than attributing the whole search cost to node expansions.

The confidence module combines:

- **intra-expert uncertainty:** uncertainty within an expert, especially ensemble variance;
- **inter-expert uncertainty:** disagreement between heterogeneous experts;
- optional risk, reference, and OOD terms supported by the C++ interface.

The fused confidence state is passed to one of these controllers:

- multiplicative;
- linear;
- threshold-based;
- fixed;
- MLP regression or classification.

Every controller shares the same final `[1, W]` safety projection. The benchmark suite includes controlled ablations for expert fusion, uncertainty terms, confidence, and controller choice.

## Bounded-suboptimality contract

Under the standard focal-search assumptions:

- the anchor heuristic is admissible;
- edge costs are positive;
- standard focal termination is used;
- every dynamic width satisfies `1 <= w_t <= W`;

the returned solution satisfies:

```text
Cost(solution) <= W * C*
```

where `C*` is the optimal path cost. The benchmark runner checks this bound for every successful search and raises an error if it observes a violation.

## Implemented methods

The C++ engine implements A*, Weighted A*, fixed focal search, learned focal search, legacy CADFS-RC, and CADFS Next.

The Python registry exposes the following benchmark names:

| Method | Description |
|---|---|
| `astar` | Optimal A* with weight 1 |
| `wastar` | Weighted A* with the global bound `W` |
| `focal_plain` | Fixed-width focal search without learned guidance |
| `learn_focal_W` | Learned focal search with width `W` |
| `learn_focal_wstar` | Learned focal search with validation-selected `w*` |
| `cadfs` | Legacy CADFS-RC |
| `cadfs_linear` | Legacy CADFS-RC with a linear controller |
| `cadfs_norisk` | Legacy CADFS-RC without the risk signal |
| `cadfs_randomrisk` | Legacy diagnostic with random risk |
| `cadfs_permutedrisk` | Legacy diagnostic with permuted risk |
| `cadfs_nofallback` | Legacy CADFS-RC without fallback |
| `cadfs_noconf` | Legacy CADFS-RC without confidence |
| `cadfs_next` | Complete CADFS Next configuration |
| `cadfs_next_geometry` | Geometric expert only |
| `cadfs_next_uniform` | Uniform expert fusion |
| `cadfs_next_nointra` | Intra-expert uncertainty disabled |
| `cadfs_next_nointer` | Inter-expert disagreement disabled |
| `cadfs_next_noconf` | Composite confidence disabled |
| `cadfs_next_linear` | Linear CADFS Next controller |
| `cadfs_next_threshold` | Threshold CADFS Next controller |
| `cadfs_next_fixed` | Fixed-width CADFS Next controller |
| `cadfs_next_mlp` | Optional MLP controller; available only when its JSON model exists |

### Benchmark suites

| Suite | Included comparison |
|---|---|
| `main` | A*, WA*, plain focal, both learned-focal baselines, legacy CADFS-RC, and CADFS Next |
| `next` | CADFS Next, legacy CADFS-RC, tuned learned focal, and all CADFS Next ablations |
| `legacy` | Classical/learned baselines, legacy CADFS-RC, and legacy ablations |
| `full` | Every method currently available, including MLP when its model exists |

## Repository structure

```text
CADFS-RC/
├── CMakeLists.txt
├── vcpkg.json
├── requirements.txt
├── configs/
│   └── default.yaml
├── cpp/
│   ├── include/cadfs/
│   │   ├── grid_map.hpp
│   │   ├── guidance.hpp
│   │   ├── heuristics.hpp
│   │   ├── risk.hpp
│   │   ├── mlp.hpp
│   │   ├── expert.hpp
│   │   ├── confidence.hpp
│   │   ├── controller.hpp
│   │   ├── telemetry.hpp
│   │   ├── search.hpp
│   │   ├── search_astar.hpp
│   │   ├── search_focal.hpp
│   │   ├── search_cadfs.hpp
│   │   └── search_cadfs_next.hpp
│   ├── src/
│   │   ├── grid_map.cpp
│   │   ├── heuristics.cpp
│   │   ├── risk.cpp
│   │   ├── mlp.cpp
│   │   ├── expert.cpp
│   │   ├── confidence.cpp
│   │   ├── controller.cpp
│   │   ├── search_astar.cpp
│   │   ├── search_focal.cpp
│   │   ├── search_cadfs.cpp
│   │   ├── search_cadfs_next.cpp
│   │   └── instance_io.cpp
│   ├── bindings/
│   │   └── bindings.cpp
│   └── tests/
│       └── test_main.cpp
├── python/
│   ├── cadfs_py/
│   │   ├── __init__.py              # C++ extension discovery
│   │   └── experiments.py           # method and suite registry
│   ├── scripts/
│   │   ├── gen_synthetic.py
│   │   ├── make_labels.py
│   │   ├── tune_validation.py
│   │   ├── tune_next_validation.py
│   │   ├── smoke_next.py
│   │   ├── run_experiments.py
│   │   └── fetch_movingai.py        # placeholder; not implemented yet
│   ├── ml/
│   │   ├── model.py
│   │   ├── train_ensemble.py
│   │   ├── export_weights.py
│   │   └── check_cpp_parity.py
│   └── analysis/
│       ├── stats.py
│       ├── tables.py
│       ├── figures.py               # legacy figures
│       └── figures_next.py          # CADFS Next figures
├── data/
│   ├── synthetic/
│   ├── instances/
│   └── labels/
└── results/
    ├── models/
    ├── logs/
    ├── tables/
    └── figures/
```

`main.cpp` is an unused CLion template and is not part of any CMake target. The project currently exposes the search engine through the `cadfs_core` library and the `cadfs_engine` Python module; it does not provide a standalone C++ command-line planner.

## Requirements

Recommended environment:

- Linux or WSL2;
- CMake 3.21 or newer;
- Ninja;
- a C++17 compiler such as GCC or Clang;
- Python 3.10 or newer;
- vcpkg;
- optional CUDA for faster PyTorch training.

The vcpkg manifest declares:

- `pybind11`;
- `nlohmann-json`.

Python dependencies are split by hardware:

- requirements-cpu.txt: CPU PyTorch wheels and all common dependencies;
- requirements-gpu.txt: CUDA 12.8 PyTorch wheels (suitable for an RTX 3080 Ti) and all common dependencies;
- requirements.txt: aliases the CPU profile, because CPU is the safe default.

CUDA is used for teacher/student training. Search-time inference uses the
distilled C++ student in both profiles: launching one small CUDA kernel per
node is slower than CPU inference. A future native CUDA search backend must
batch nodes (for example, K-Focal-style evaluation) before it can be expected
to help.

## Build and test

Clone the repository and enter its root:

```bash
git clone <YOUR_REPOSITORY_URL>
cd CADFS-RC
```

Install or bootstrap vcpkg separately, then set `VCPKG_ROOT` to its absolute path. Configure a Release build:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

Build all default targets:

```bash
cmake --build build -j
```

Run the C++ test suite:

```bash
ctest --test-dir build --output-on-failure
```

IPO/LTO is enabled in Release builds when the compiler supports it. For
machine-specific paper benchmarks, configure a separate non-portable build:

```bash
cmake -S . -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCADFS_NATIVE_ARCH=ON \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-native -j
```

Record the CPU model and compiler in the adjacent experiment manifest; never
compare a native build against portable baselines built with different flags.

### Useful CMake variants

Build only the dependency-free C++ core and tests, without Python bindings:

```bash
cmake -S . -B build-core -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCADFS_BUILD_PYTHON=OFF
cmake --build build-core -j
ctest --test-dir build-core --output-on-failure
```

Build the Python module without C++ tests:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCADFS_BUILD_TESTS=OFF
cmake --build build -j
```

The newer Python tools search `CADFS_ENGINE_DIR`, `cmake-build-debug`, `build`, other `cmake-build-*` directories, and the repository root. For a custom build location:

```bash
export CADFS_ENGINE_DIR=/absolute/path/to/build
```

For the complete pipeline, using the standard `build/` directory is recommended because the older data-generation and legacy-tuning scripts import the extension from that location directly.

## Python environment

CPU installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

GPU installation (CUDA 12.8 PyTorch wheel):

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

Run every command from the repository root. Generated data, model checkpoints, logs, tables, figures, virtual environments, and build artifacts are excluded by `.gitignore`.

configs/cpu.yaml and configs/gpu.yaml document the two reproducible profiles.
They are manifests, not universal CLI configuration files; pass the shown
settings to the corresponding scripts.

## Complete pipeline

The following sequence reproduces the implemented workflow from an empty `data/` and `results/` directory.

### 1. Generate synthetic maps, queries, and optimal costs

```bash
python python/scripts/gen_synthetic.py \
  --seed 1234 \
  --maps-per-split 40 \
  --queries-per-map 10 \
  --starts-per-goal 2 \
  --size 64 \
  --shift-size 128 \
  --workers 8 \
  --max-map-attempts 100
```

This creates MovingAI-format `.map` files in `data/synthetic/` and one instance CSV per split in `data/instances/`. Reverse Dijkstra searches generate several reachable, well-separated queries per goal and record exact `optimal_cost` values. Maps are generated in parallel worker processes, and each completed split is checkpointed immediately to its CSV.

To add the structural domains to an existing random-grid dataset without regenerating the completed random splits:

```bash
python python/scripts/gen_synthetic.py \
  --seed 1234 \
  --maps-per-split 120 \
  --queries-per-map 12 \
  --starts-per-goal 2 \
  --size 128 \
  --shift-size 256 \
  --splits train_structural val_structural val_shift shift_family \
  --workers 12
```

`shift_family` is regenerated because its corridor/passage parameters are deliberately harder than those used by structural training. It remains map-disjoint and test-only.

For a small development dataset:

```bash
python python/scripts/gen_synthetic.py \
  --seed 1234 \
  --maps-per-split 8 \
  --queries-per-map 3 \
  --starts-per-goal 1 \
  --size 32 \
  --shift-size 48 \
  --workers 4
```

### 2. Generate cost-to-go labels

Generate labels for all default splits:

```bash
python python/scripts/make_labels.py \
  --samples-per-goal 300 \
  --seed 7 \
  --workers 12
```

Generate only the training and validation labels:

```bash
python python/scripts/make_labels.py \
  --splits train train_structural val val_structural val_shift \
  --samples-per-goal 300 \
  --seed 7 \
  --workers 12
```

For each unique `(map, goal)` pair, the script runs `dijkstra_all` once and samples reachable cells. It writes compressed arrays to `data/labels/<split>.npz`:

```text
patch : (N, 1, 31, 31) uint8 local occupancy patches (converted per batch)
extra : (N, 10)         goal geometry, line obstruction, multi-scale density, and mobility
y     : (N,)            normalized optimal cost-to-go
meta  : (N, 3)          map/cell/goal identifiers
```

### 3. Train and calibrate the deep ensemble

```bash
python python/ml/train_ensemble.py \
  --K 7 \
  --epochs 80 \
  --batch-size 256 \
  --lr 3e-4 \
  --patience 10 \
  --structural-weight 1.0 \
  --hidden 96
```

Each member receives a different seed and a replacement-sampled, domain-balanced stream from `train` and `train_structural`. Early stopping minimizes a worst-domain score over `val` and `val_structural`, so structural accuracy cannot improve by silently sacrificing random-grid accuracy. AdamW, gradient clipping, learning-rate reduction, and patience-based early stopping provide regularization. Ensemble-variance calibration uses the disjoint `val_shift` proxy. The script accepts `--device cpu` or `--device cuda`; `--amp --compile` is the recommended GPU profile.

The network predicts `log1p(d*/diagonal)`, which represents long structural detours without clipping. At search time Python and C++ apply the same monotone bounded priority `1-exp(-prediction)`. The exported non-negative affine calibration, `scale * ensemble_variance + floor`, is applied by the C++ engine itself; its floor captures shared ensemble bias that scale-only calibration cannot represent.

This command also evaluates the trained ensemble and creates the ML diagnostics:

```text
results/models/member_0.pt ... member_<K-1>.pt
results/models/train_log.csv
results/models/calibration.json
results/models/training_manifest.json
results/tables/ml_metrics.csv
results/figures/ml_training_curves.png
results/figures/ml_pred_vs_true.png
results/figures/ml_uncertainty_shift.png
results/figures/ml_calibration.png
results/figures/ml_error_vs_std.png
```

For a quick training smoke run:

```bash
python python/ml/train_ensemble.py \
  --K 2 \
  --epochs 2 \
  --batch-size 128 \
  --patience 1
```

### 3b. Train the low-latency C++ student

The publication backend is a shared-backbone, three-head student. It sees the
complete 31x31 patch and the same ten structural features as the teacher.
Exact Dijkstra labels supervise cost-to-go, paired cells from the same
map-goal problem provide ranking loss, and bootstrap masks keep the heads
different enough to estimate uncertainty.

CPU-only training without teacher distillation:

```bash
python python/ml/train_student.py \
  --device cpu \
  --epochs 50 \
  --batch-size 512 \
  --supervised-weight 0.8 \
  --rank-weight 0.2 \
  --teacher-weight 0
```

GPU training with CNN-teacher distillation:

```bash
python python/ml/train_student.py \
  --device cuda \
  --amp \
  --compile \
  --epochs 50 \
  --batch-size 2048 \
  --supervised-weight 0.60 \
  --rank-weight 0.15 \
  --teacher-weight 0.25
```

This writes `fast_student.pt`, `calibration_fast.json`, and the
directly loadable C++ file `fast_ensemble.txt`. Verify it before tuning:

```bash
python python/ml/check_cpp_parity.py \
  --backend fast \
  --split val_structural \
  --samples 64
```

### 4. Export PyTorch weights for C++ inference

```bash
python python/ml/export_weights.py
```

The exporter loads all `results/models/member_*.pt` checkpoints and writes the dependency-free C++ format:

```text
results/models/ensemble.txt
```

Verify exact PyTorch/C++ numerical parity before tuning:

```bash
python python/ml/check_cpp_parity.py \
  --split val_structural \
  --samples 64
```

### 5. Tune the legacy CADFS-RC configuration

```bash
python python/scripts/tune_validation.py
```

This script uses 40 shuffled instances from `data/instances/val.csv`. It selects:

- `w_star` for the tuned learned-focal baseline;
- confidence temperature `tau_c`;
- fallback confidence threshold `theta_c`.

The selected configuration is written to:

```text
results/models/tuned.json
```

### 6. Tune CADFS Next

```bash
python python/scripts/tune_next_validation.py \
  --splits val val_structural val_shift \
  --per-split 20 \
  --workers 1 \
  --seed 17 \
  --guidance fast \
  --out results/models/tuned_next.json
```

Tuning jointly evaluates expert fusion, confidence, and controller choices.
Candidates with mean focal width at most 1.02 or fallback rate at least 0.90
are rejected as collapsed. The weighted objective includes expansions,
serial runtime, fallback, suboptimality, and a collapse penalty. It writes a
balanced selection to `tuned_next.json` and a quality-weighted selection
to `tuned_next_conservative.json`; every trial is retained for audit.
The script rejects `test`, `shift_density`, `shift_size`, and
`shift_family` to prevent test leakage. Use `--workers 1` for timing
selection; parallel workers are useful only for throughput while developing.

If `tuned_next.json` is absent, the smoke test and benchmark runner use the documented CADFS Next defaults. `tuned.json` and `ensemble.txt` are still required.

### 7. Run a fast integration and bound check

```bash
python python/scripts/smoke_next.py \
  --split val \
  --instances 3
```

Run selected methods:

```bash
python python/scripts/smoke_next.py \
  --split val_shift \
  --instances 5 \
  --methods astar learn_focal_wstar cadfs cadfs_next
```

The smoke script prints expansions, runtime, suboptimality ratio, and mean focal width, and fails immediately on search failure or a bound violation.

## Benchmark commands

### List available methods

```bash
python python/scripts/run_experiments.py --list-methods
```

The extension, `results/models/ensemble.txt`, and `results/models/tuned.json` must exist even when listing methods. `cadfs_next_mlp` appears only if the configured MLP JSON exists.

### Small validation benchmark

```bash
python python/scripts/run_experiments.py \
  --splits val \
  --per-split 5 \
  --suite main \
  --seed 3 \
  --out results/logs/dev_main.csv
```

### Main paper-style benchmark

```bash
python python/scripts/run_experiments.py \
  --splits test shift_density shift_size shift_family \
  --per-split 40 \
  --suite main \
  --seed 3 \
  --out results/logs/bench_next.csv
```

### CADFS Next ablation benchmark

```bash
python python/scripts/run_experiments.py \
  --splits test shift_density shift_size shift_family \
  --per-split 40 \
  --suite next \
  --seed 3 \
  --out results/logs/bench_next_ablation.csv
```

### Legacy CADFS-RC benchmark

Use `bench.csv` if you also want to run the hard-coded legacy figure script:

```bash
python python/scripts/run_experiments.py \
  --splits test shift_density shift_size shift_family \
  --per-split 40 \
  --suite legacy \
  --seed 3 \
  --out results/logs/bench.csv
```

### Full benchmark

```bash
python python/scripts/run_experiments.py \
  --splits test shift_density shift_size shift_family \
  --per-split 40 \
  --suite full \
  --seed 3 \
  --out results/logs/bench_full.csv
```

### Explicit method selection

`--methods` overrides the selected suite:

```bash
python python/scripts/run_experiments.py \
  --splits test shift_family \
  --per-split 20 \
  --methods astar wastar learn_focal_wstar cadfs cadfs_next cadfs_next_fixed \
  --seed 3 \
  --out results/logs/custom_comparison.csv
```

### Custom tuning and MLP model files

```bash
python python/scripts/run_experiments.py \
  --suite full \
  --tuned-next results/models/my_tuned_next.json \
  --mlp-model results/models/controller_mlp.json \
  --out results/logs/bench_custom_models.csv
```

The repository supports loading an MLP controller but does not currently include an end-to-end MLP-controller training script. The expected JSON contains `w1`, `b1`, `w2`, `b2`, and optionally `actions`.

### Append to or replace an existing log

By default, the runner refuses to overwrite a CSV.

Append rows while preserving the schema:

```bash
python python/scripts/run_experiments.py \
  --splits shift_family \
  --suite main \
  --out results/logs/bench_next.csv \
  --append
```

Explicitly replace an existing log:

```bash
python python/scripts/run_experiments.py \
  --suite main \
  --out results/logs/bench_next.csv \
  --overwrite
```

Every benchmark writes a JSON manifest beside its CSV. Appended runs receive timestamped manifest names.

## Analysis and figure commands

### CADFS Next tables and statistical analysis

```bash
python python/analysis/tables.py \
  --input results/logs/bench_next.csv \
  --tag next
```

Generate the systems table that separates model overhead from search:

```bash
python python/analysis/systems_table.py \
  --input results/logs/bench_next.csv \
  --tag next \
  --baseline astar
```

This table reports model-time share, head/member work, cache behavior, paired
speedup, and paired expansion reduction. Small grids where A* completes in
fractions of a millisecond must remain in the report; larger 512/1024/2048
size-shift tiers should be added as predeclared runtime regimes, not used to
hide negative small-map results.

This generates:

```text
results/tables/next_main_results.csv
results/tables/next_paired_tests.csv
results/tables/next_risk_correlation.csv
results/tables/next_ablation.csv
```

The analysis includes means, medians, 95th percentiles, success and fallback rates, maximum observed ratios, paired bootstrap confidence intervals, Wilcoxon signed-rank tests, paired rank-biserial effect sizes, percent changes, and Spearman correlations.

Analyze another benchmark without replacing the `next_*` tables:

```bash
python python/analysis/tables.py \
  --input results/logs/bench_next_ablation.csv \
  --tag next_ablation
```

### CADFS Next figures

```bash
python python/analysis/figures_next.py \
  --input results/logs/bench_next.csv \
  --tag next \
  --bound 2.0
```

Depending on the methods present in the input CSV, this produces:

```text
results/figures/next_main_expansions.png
results/figures/next_runtime.png
results/figures/next_width_fallback.png
results/figures/next_suboptimality.png
results/figures/next_ablations.png
```

Generate a separate set for the ablation log:

```bash
python python/analysis/figures_next.py \
  --input results/logs/bench_next_ablation.csv \
  --tag next_ablation \
  --bound 2.0
```

### Legacy figures

The legacy figure script has no CLI arguments and always reads `results/logs/bench.csv`:

```bash
python python/analysis/figures.py
```

It creates:

```text
results/figures/fig_main_expansions.png
results/figures/fig_degradation_density.png
results/figures/fig_width_fallback.png
results/figures/fig_risk_vs_difficulty.png
results/figures/fig_ablation_appendix.png
results/figures/fig_suboptimality.png
results/figures/fig_calibration_comparison.png
```

The legacy calibration-comparison figure contains recorded constants in `CALIBRATION_COMPARISON`. Regenerate or update those values in `python/analysis/figures.py` if the dataset, model, seed, or thresholds change.

### ML training figures

No separate plotting command is required for ML diagnostics. They are generated automatically by:

```bash
python python/ml/train_ensemble.py
```

## Complete command reference

### `python/scripts/gen_synthetic.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--seed` | `1234` | Master random seed |
| `--maps-per-split` | `40` | Base number of maps used to derive split sizes |
| `--queries-per-map` | `10` | Reachable start-goal queries per map |
| `--starts-per-goal` | `2` | Queries sharing one reverse-Dijkstra goal; smaller values increase goal diversity |
| `--size` | `64` | Standard square-map size |
| `--shift-size` | `128` | Size-shift square-map size |
| `--splits` | all splits | Optional subset to generate without touching other split CSVs |
| `--workers` | up to `8` | Parallel map-generation processes; use `1` to disable multiprocessing |
| `--max-map-attempts` | `100` | Maximum rerolls per difficult map before failing explicitly |

### `python/scripts/make_labels.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--splits` | all nine splits | Splits for which `.npz` labels are generated |
| `--samples-per-goal` | `150` | Maximum reachable cells sampled per map-goal pair |
| `--seed` | `7` | Sampling seed |
| `--workers` | up to `8` | Parallel map-level label workers |

### `python/ml/train_ensemble.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--K` | `7` | Number of ensemble members |
| `--epochs` | `80` | Maximum epochs per member |
| `--batch-size` | `256` | Training batch size |
| `--lr` | `3e-4` | AdamW learning rate |
| `--patience` | `10` | Worst-domain early-stopping patience |
| `--structural-weight` | `1.0` | Structural-domain mass relative to random-domain mass |
| `--hidden` | `96` | Hidden-layer width recorded in each checkpoint |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--amp` | false | Mixed precision on CPU/CUDA |
| `--compile` | false | Compile each member with torch.compile |

### `python/ml/train_student.py`

Trains and exports the shared-backbone fast student. Important options are
`--device`, `--amp`, `--compile`, `--heads`,
`--hidden1`, `--hidden2`, the three loss weights,
`--teacher-dir`, and `--out`. The loss weights must sum to one.
`--max-samples`, `--max-val-samples`, and `--artifacts-dir`
exist for isolated smoke tests and must not be used for final paper training.

### `python/ml/export_weights.py`

With no arguments, reads every `results/models/member_*.pt`, embeds `calibration.json`, and writes `results/models/ensemble.txt`. `--models-dir` and `--out` allow isolated exports.

### `python/ml/check_cpp_parity.py`

Checks exported outputs against PyTorch on real map cells. Use
`--backend cnn` for the teacher or `--backend fast` for the
student. Other useful options are `--split`, `--samples`,
`--seed`, `--atol`, `--model`, and
`--checkpoints-dir`.

### `python/scripts/tune_validation.py`

No arguments. Reads `val.csv` and `ensemble.txt`, then writes `results/models/tuned.json`.

### `python/scripts/tune_next_validation.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--splits` | `val val_structural val_shift` | Validation-only selection splits |
| `--per-split` | `20` | Instances sampled from each split |
| `--workers` | up to `8` | Parallel C++ searches per candidate; use `1` for serial timing |
| `--seed` | `17` | Deterministic selection seed |
| `--out` | `results/models/tuned_next.json` | Audit and selected settings JSON |
| `--out-conservative` | `results/models/tuned_next_conservative.json` | Conservative profile |
| `--guidance` | `auto` | `fast`, `cnn`, or `cnn-adaptive`; auto prefers fast |
| `--max-candidates` | `0` | Development-only deterministic cap; zero runs the full grid |

### `python/scripts/smoke_next.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--split` | `val` | Instance split used by the smoke test |
| `--instances` | `3` | Number of first CSV rows to run |
| `--methods` | `learn_focal_wstar cadfs cadfs_next` | Methods to compare |

### `python/scripts/run_experiments.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--splits` | `test shift_density shift_size shift_family` | Evaluation splits |
| `--per-split` | `40` | Shuffled instances selected per split |
| `--seed` | `3` | Shuffle and experiment seed |
| `--suite` | `main` | `main`, `next`, `legacy`, or `full` |
| `--methods` | unset | Explicit method list; overrides the suite |
| `--list-methods` | false | Print currently available methods |
| `--tuned-next` | `results/models/tuned_next.json` | CADFS Next tuning file |
| `--mlp-model` | `results/models/controller_mlp.json` | Optional MLP-controller file |
| `--guidance` | `auto` | Guidance backend; auto prefers fast_ensemble.txt |
| `--early-exit-members` | `2` | First CNN members used by the adaptive backend |
| `--early-exit-variance` | `0.01` | Calibrated variance threshold for running remaining members |
| `--out` | `results/logs/bench_next.csv` | Benchmark CSV path |
| `--append` | false | Append if the existing schema matches |
| `--overwrite` | false | Explicitly replace an existing CSV |

`--append` and `--overwrite` are mutually exclusive.

### `python/analysis/tables.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--input` | `results/logs/bench_next.csv` | Benchmark CSV |
| `--tag` | `next` | Prefix for generated table filenames |

### `python/analysis/systems_table.py`

Reads a v2 telemetry CSV and writes a systems summary. Useful options are
`--input`, `--tag`, `--baseline`, and `--out`.

### `python/analysis/figures_next.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--input` | `results/logs/bench_next.csv` | Benchmark CSV |
| `--tag` | `next` | Prefix for generated figure filenames |
| `--bound` | `2.0` | Reference bound drawn in the suboptimality plot |

### `python/analysis/figures.py`

No arguments. Reads `results/logs/bench.csv` and writes legacy `fig_*.png` figures.

### `python/scripts/fetch_movingai.py`

This file is currently a `TODO` placeholder and has no implemented download workflow. Synthetic MovingAI-format maps are fully supported; automatic download of external MovingAI benchmark datasets is not yet available.

## Data splits

| Split | Role | Distribution |
|---|---|---|
| `train` | Model fitting | Random grids, density 0.10-0.30 |
| `train_structural` | Model fitting | Moderate maze/narrow maps; corridor/passage settings exclude the hard test parameters |
| `val` | Early stopping and legacy tuning | Disjoint random grids, density 0.10-0.30 |
| `val_structural` | Early stopping | Disjoint moderate maze/narrow maps |
| `val_shift` | Variance calibration and CADFS Next tuning | Mild, disjoint maze/narrow geometry shift |
| `test` | Final in-distribution evaluation | Held-out random grids, density 0.10-0.30 |
| `shift_density` | OOD test | Random grids, density 0.35-0.50 |
| `shift_size` | OOD test | Larger random grids |
| `shift_family` | OOD test | Seen structural families with held-out hard corridor/passage and clutter parameters |

The generator asserts map-level disjointness between every split.

## Generated artifacts

| Path | Produced by | Purpose |
|---|---|---|
| `data/synthetic/<family>/<split>/*.map` | `gen_synthetic.py` | MovingAI-format maps |
| `data/instances/<split>.csv` | `gen_synthetic.py` | Queries, metadata, and optimal costs |
| `data/labels/<split>.npz` | `make_labels.py` | Deep-learning samples and oracle targets |
| `results/models/member_*.pt` | `train_ensemble.py` | PyTorch ensemble checkpoints |
| `results/models/train_log.csv` | `train_ensemble.py` | Per-member epoch losses |
| `results/models/calibration.json` | `train_ensemble.py` | Variance-calibration diagnostics |
| `results/models/training_manifest.json` | `train_ensemble.py` | Model architecture and split protocol |
| `results/models/ensemble.txt` | `export_weights.py` | Versioned weights, target metadata, and calibration for C++ |
| `results/models/fast_student.pt` | `train_student.py` | Shared-backbone PyTorch student checkpoint |
| `results/models/calibration_fast.json` | `train_student.py` | Student head-variance calibration |
| `results/models/fast_ensemble.txt` | `train_student.py` | Dependency-free low-latency C++ student |
| `results/models/tuned.json` | `tune_validation.py` | Legacy parameters and learned-focal `w*` |
| `results/models/tuned_next.json` | `tune_next_validation.py` | CADFS Next selection plus all audited trials |
| `results/logs/*.csv` | `run_experiments.py` | Per-instance, per-method benchmark rows |
| `results/logs/*.manifest.json` | `run_experiments.py` | Reproducibility metadata |
| `results/tables/*.csv` | training or `tables.py` | ML metrics and statistical summaries |
| `results/figures/*.png` | training or figure scripts | ML, benchmark, and ablation plots |

### Benchmark CSV schema

The runner records:

```text
problem_id, split, family, density, map_id, instance, seed,
algorithm_version, method, found, cost, cstar, ratio, expansions,
generated, runtime_ms, fallback_rate, mean_w, min_w, max_w,
mean_abs_dw, mean_C, mean_R, model_eval_count, model_member_evals,
model_cache_hits, model_cache_hit_rate, model_eval_time_ms
```

## Python API example

After building the extension and exporting `ensemble.txt`:

```python
import math
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "python"))

from cadfs_py import load_engine

engine = load_engine(required=("run_cadfs_next",))
grid = engine.GridMap.load_movingai("data/synthetic/random/val/example.map")
model = engine.FastEnsembleGuidance("results/models/fast_ensemble.txt")

config = {
    "W": 2.0,
    "L": 16,
    "K": 50,
    "connectivity": 8,
    "h_min": 0.0,
    "h_max": math.hypot(grid.width, grid.height),
    "expert_weights": [1 / 3, 1 / 3, 1 / 3],
    "confidence_intra_weight": 1.0,
    "confidence_inter_weight": 1.0,
    "confidence_temperature": 0.05,
    "next_controller": "multiplicative",
}

result = engine.run_cadfs_next(
    grid,
    (1, 1),
    (grid.width - 2, grid.height - 2),
    config,
    model,
)

print(result)
```

Replace the map path and endpoints with a valid generated instance. The result dictionary contains `found`, `cost`, `path`, `expansions`, `generated`, `runtime_ms`, fallback statistics, width statistics, mean confidence, and mean risk.
It also reports model evaluations, member/head evaluations, cache hits, cache
hit rate, and measured model-evaluation time.

## Configuration notes

`configs/default.yaml` documents the legacy search, controller, risk, ranking, fallback, confidence, and normalization defaults. The current experiment scripts do not use it as a universal command-line configuration file: legacy tuning writes `tuned.json`, CADFS Next tuning writes `tuned_next.json`, and the benchmark registry constructs C++ configuration dictionaries from those artifacts.

Default search values include:

```text
W = 2.0
L = 16
K = 50
connectivity = 8
```

Use only `4` or `8` for grid connectivity.

## Reproducibility protocol

- Run commands from the repository root.
- Keep training, validation, calibration, and test maps disjoint.
- Fit network weights on `train` and `train_structural` only.
- Use `val` and `val_structural` for model early stopping; use `val` for legacy search selection.
- Use `val_shift` only as the documented mild shift/calibration proxy.
- Never tune on `test`, `shift_density`, `shift_size`, or `shift_family`.
- Compare methods on identical shuffled instances and the same seed.
- Preserve the CSV and its adjacent manifest together.
- Use a new output filename for each experiment, or explicitly choose `--append`/`--overwrite`.
- Report solution ratio and bound violations alongside speed and expansion metrics.
- For timing claims, prefer a Release build; CMake uses `-O3 -DNDEBUG` for Release.
- Use `--workers 1`, warm up each backend, pin the CPU if possible, and
  report cold-start, search-only, model-only, and amortized timings separately.
- Treat `--max-samples`, `--max-val-samples`, and
  `--max-candidates` as development controls, never paper settings.

## Known limitations

- Grid planning is implemented; road graphs, turn restrictions, map matching, live traffic, contraction hierarchies, and geographic rendering are outside the current scope.
- `fetch_movingai.py` is not implemented.
- The MLP controller can be loaded and executed, but its training/data-generation pipeline is not included.
- The current topological expert is a deterministic degree/density baseline,
  not a GNN. The repository must not claim HGNN results until a real region
  graph model, its training code, and its ablations are implemented.
- No native CUDA search kernel is included. GPU accelerates training; final
  search uses the portable C++ student. This is intentional until batched
  K-Focal evaluation can be tested on real GPU hardware.
- The fast student code has end-to-end and parity tests, but a final full-data
  student checkpoint and its held-out benchmark must be generated before any
  speed/quality superiority claim.
- `tune_validation.py`, `gen_synthetic.py`, and `make_labels.py` use the standard `build/` import path rather than the newer extension-discovery helper.
- The legacy `figures.py` script has a fixed input path and includes recorded calibration-comparison constants.
- Generated data and result artifacts are ignored by Git and must be reproduced locally.
- The bounded-suboptimality claim depends on an admissible anchor, positive edge costs, the focal termination rule, and the safety projection. Arbitrary changes to those components may invalidate it.

## Using the core in a navigation product

The C++ library can be embedded as a search component in a larger routing system. A production map application would still need road-graph ingestion, geospatial indexing, snapping/map matching, turn restrictions, dynamic traffic costs, hierarchical routing, rerouting, backend APIs, and a renderer. CADFS-RC currently implements the search and learned-control layer only.

## Citation

If you use this repository in academic work, cite the accompanying paper or thesis. A BibTeX entry can be added here when the publication metadata is available.

## Contributing

Issues and focused pull requests are welcome. Algorithmic changes should include tests and state whether they affect admissibility assumptions, the focal bound, experiment comparability, or output schemas.
