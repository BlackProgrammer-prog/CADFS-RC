#pragma once
#include <algorithm>
#include <cmath>
#include "grid_map.hpp"
#include "search.hpp"

namespace cadfs {

// R_obs: obstacle density in local window (Sec. 6.2).
double risk_obs(const GridMap& m, int node, int radius);

// R_mob = 1 - deg_free/deg_max (Sec. 6.3).
double risk_mob(const GridMap& m, int node, int connectivity);

// H_ref: normalized anchor heuristic with validation stats (Sec. 7.3).
inline double h_ref_norm(double h_a, const Config& cfg) {
    double v = (h_a - cfg.h_min) / (cfg.h_max - cfg.h_min + cfg.eps);
    return std::min(1.0, std::max(0.0, v));
}

// R_dev = |H_L - H_ref|, both already in [0,1] (Sec. 6.4).
inline double risk_dev(double H_L, double H_ref) { return std::abs(H_L - H_ref); }

// Combined R_s (Sec. 6.5); honors ablation modes.
double risk_combined(const GridMap& m, int node, double H_L, double h_a,
                     const Config& cfg, uint64_t permute_salt);

} // namespace cadfs
