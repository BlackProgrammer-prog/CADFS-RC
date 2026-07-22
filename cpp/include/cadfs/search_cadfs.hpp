#pragma once
#include "grid_map.hpp"
#include "guidance.hpp"
#include "search.hpp"

namespace cadfs {
// Full CADFS (paper Algorithm 1): B_t -> Q_t (anchor Top-L) -> (C_t,R_t,F_t)
// -> w_t -> FOCAL_t -> risk-aware selection + threshold fallback.
// Controller variants (multiplicative/linear/fixed/tuned_fixed) via cfg.
SearchResult cadfs_search(const GridMap& m, const Instance& ins, const Config& cfg,
                          const GuidanceModel& model, double tau_c);
} // namespace cadfs
