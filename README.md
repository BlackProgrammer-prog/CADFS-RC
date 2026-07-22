# CADFS — Confidence-Aware Dynamic Focal Search

C++ search core (honest runtime) + Python for datasets, ML training, experiments, analysis.

## Layout
```
cadfs/
├── CMakeLists.txt
├── vcpkg.json                  # C++ deps: pybind11, catch2, nlohmann-json
├── requirements.txt
├── configs/default.yaml        # W, L, K, thresholds, lambdas, ...
├── cpp/
│   ├── include/cadfs/
│   │   ├── grid_map.hpp        # .map loader (MovingAI) + synthetic grids, neighbors
│   │   ├── heuristics.hpp      # octile/manhattan anchor h_a
│   │   ├── mlp.hpp             # tiny MLP forward pass (weights exported from PyTorch)
│   │   ├── risk.hpp            # R_obs, R_mob, R_dev
│   │   ├── search.hpp          # Instance, Config, SearchResult (stats schema)
│   │   ├── search_astar.hpp    # A*, Weighted A*
│   │   ├── search_focal.hpp    # fixed-width focal: width = W or tuned w*
│   │   └── search_cadfs.hpp    # CADFS: B_t → Q_t → controller → FOCAL_t → fallback
│   ├── src/                    # implementations (stubs for now)
│   ├── bindings/bindings.cpp   # pybind11 module `cadfs_engine`
│   └── tests/                  # Catch2: bound never violated, w_t ∈ [1,W], etc.
├── python/
│   ├── cadfs_py/               # thin wrapper over cadfs_engine
│   ├── scripts/
│   │   ├── gen_synthetic.py    # random/maze/narrow + queries + MAP-LEVEL splits + shift sets
│   │   ├── fetch_movingai.py   # download & index MovingAI subset (.map/.scen)
│   │   ├── make_labels.py      # Dijkstra-from-goal d*(cell,goal) labels
│   │   ├── run_experiments.py  # baselines/ablations/seeds → results/logs/*.csv
│   │   └── tune_validation.py  # w*, thresholds, lambdas, y/h normalization stats
│   ├── ml/
│   │   ├── model.py            # patch CNN/MLP cost-to-go regressor
│   │   ├── train_ensemble.py   # K_e=5 (different seeds/bootstraps)
│   │   └── export_weights.py   # PyTorch → engine weight format (npz/json)
│   └── analysis/
│       ├── stats.py            # Wilcoxon, paired bootstrap CI, Spearman
│       ├── tables.py           # paper Tables 1–8
│       └── figures.py          # Figures 1–5 + shift-severity degradation curve
├── data/       (generated, git-ignored)
└── results/    (logs, figures, tables)
```

## Build
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -G Ninja
cmake --build build && ctest --test-dir build
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Design decisions
- **Ensemble inference lives in C++** (`mlp.hpp`): runtime comparisons stay honest.
  PyTorch only trains; `export_weights.py` dumps weights the engine loads.
- **Map-level splits** (train/val/test/shift maps disjoint) enforced in `gen_synthetic.py`.
- Instance CSV: `map_id,start_x,start_y,goal_x,goal_y,family,density,width,height,split,optimal_cost`
- Per-instance log: expansions, runtime, cost, C*, success, fallback_rate,
  mean/min/max w_t, mean |Δw_t| — one schema feeds every paper table/figure.
