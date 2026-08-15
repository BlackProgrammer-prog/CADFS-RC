# CADFS-RC research protocol

This document separates implemented contributions from future hypotheses. It
is a gate for paper claims, not a promise that a model is state of the art.

## Evidence behind the architecture

- [Neural A* (ICML 2021)](https://proceedings.mlr.press/v139/yonetani21a.html)
  encodes an instance into a guidance map before search. It supports the design
  rule that expensive global perception should run once per query.
- [PHIL / Learning Graph Search Heuristics (LoG 2022)](https://proceedings.mlr.press/v198/pandy22a.html)
  targets constant-time learned-heuristic evaluation and reports fewer explored
  nodes on several graph domains. It motivates search-aware ranking and cheap
  node-time inference.
- [K-Focal Search for Slow Learned Heuristics (SoCS 2022)](https://ojs.aaai.org/index.php/SOCS/article/download/21785/21549/25828)
  obtains GPU benefit by batching several focal nodes. It is the required
  reference design for any future CUDA search backend.
- [Dynamic Potential Search (SoCS 2019)](https://ojs.aaai.org/index.php/SOCS/article/download/18392/18183/21908)
  is a classical bounded-suboptimal baseline and explains the focal-search
  bound used by CADFS.
- [NeuroMP (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4c7912516fdb339b12bad45eefda523c-Abstract-Conference.html)
  uses learned graph construction for continuous-space motion planning. It is
  relevant inspiration, but its domain and hardware results are not directly
  comparable to this grid C++ engine.

These papers do not establish that a GNN is automatically best for this
repository. Current measured latency is dominated by repeated CNN evaluation,
so the first implemented contribution is teacher-to-student compression with
ranking-aware training, caching, adaptive evaluation, and auditable telemetry.

## Implemented paper candidate

Working name: **CADFS-SH: Confidence-Aware Dynamic Focal Search with a
Shared-Head Student**.

Implemented components:

1. admissible anchor and final width projection preserving the global bound;
2. exact per-query node cache;
3. full CNN, adaptive CNN, and shared-backbone student backends;
4. Dijkstra-supervised cost-to-go plus within-query pairwise ranking;
5. optional CNN-teacher distillation;
6. bootstrap uncertainty heads and affine validation-shift calibration;
7. joint expert/confidence/controller tuning with collapse rejection;
8. per-instance model-time, cache, member/head, search, and quality telemetry.

The topological expert is currently hand-crafted. **Do not call the implemented
system a hierarchical GNN.** A future CADFS-HGNN paper needs a real region
graph builder, message-passing model, export/runtime path, and ablations.

## CPU and GPU roles

| Phase | CPU profile | GPU profile |
|---|---|---|
| Labels | parallel C++ reverse Dijkstra | same |
| CNN teacher | optional/slow | CUDA AMP + optional torch.compile |
| Student | direct supervised + ranking | supervised + ranking + distillation |
| Final search | C++ fast student | C++ fast student |
| Future GPU search | not applicable | batched K-Focal only |

Native scalar CUDA inference is intentionally absent: per-node transfers and
kernel launches would hide the arithmetic savings. CUDA becomes a valid
search contribution only after the batch size, queue semantics, bound, device
transfer time, and cold/warm timing are measured on real hardware.

## Mandatory baselines

Grid experiments:

- Dijkstra, A*, Weighted A*, plain focal search;
- learned focal with full CNN, adaptive CNN, and fast student;
- legacy CADFS-RC and complete CADFS Next;
- Dynamic Potential Search if the paper claims broad bounded-search novelty;
- Neural A* or TransPath only if reproduced under the same data, map sizes,
  hardware, and timing boundary.

Predeclare size regimes: 64/128 for overhead and quality, 512/1024 for the main
scaling curve, and 2048 as a stress tier when memory permits. Report every
regime. A method that loses to A* on small maps but wins when search work
dominates is a crossover result, not universal superiority.

Road-network claims require a separate graph-native engine and at least
bidirectional Dijkstra/A*, ALT, and Contraction Hierarchies. The current grid
engine cannot support road-routing superiority claims.

## Required ablations

1. supervised student versus supervised plus ranking;
2. no teacher versus teacher distillation;
3. one head versus three bootstrap heads;
4. full CNN versus adaptive CNN versus student;
5. cache disabled versus enabled;
6. geometry only, deterministic topology, and fused experts;
7. intra uncertainty, inter disagreement, confidence, fallback, and controller;
8. balanced versus conservative tuned profile.

## Timing protocol

- Release build with compiler and flags recorded.
- One serial benchmark worker for latency claims.
- Fixed CPU affinity and performance governor when available.
- Warm-up runs excluded and reported separately from cold start.
- Report preprocessing, model load, query preparation, model evaluation,
  search, total query, and amortized multi-query time.
- Report peak RAM and, on GPU, peak device memory and host-device transfer time.
- Every method receives the same paired query order.
- Timeouts and failures remain in the denominator.

## Statistical protocol

- Select every hyperparameter on `val`, `val_structural`, and
  `val_shift` only.
- Freeze code, model, and tuning JSON before opening test results.
- Report per-family and aggregate median, mean, 95% bootstrap confidence
  intervals, and paired effect sizes.
- Use paired Wilcoxon tests with Holm correction for multiple method
  comparisons; include the number of zero differences.
- Report expansions, generated nodes, total runtime, model runtime, cache-hit
  rate, success, path-cost ratio, and maximum observed ratio.
- Archive the CSV, manifest, model hashes, commit hash, environment, and exact
  command together.

## Claim gates

The following claims are forbidden until their gate is satisfied:

- **Faster:** statistically significant total wall-clock improvement on held-out
  maps, including model overhead.
- **Fewer nodes:** significant paired expansion reduction without hiding
  failures or using different instance subsets.
- **Better quality:** lower paired cost ratio at the same declared bound.
- **Robust OOD:** improvement on all predeclared shift families, not only the
  best one.
- **GPU acceleration:** end-to-end GPU-server timing including transfers and
  cold start.
- **State of the art / suitable for an indexed journal:** reproduced external
  baselines and a complete held-out statistical report. Code changes alone do
  not satisfy this gate.
