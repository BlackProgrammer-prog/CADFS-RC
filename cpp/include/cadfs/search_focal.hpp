#pragma once
#include "grid_map.hpp"
#include "guidance.hpp"
#include "search.hpp"

namespace cadfs {
// Fixed-width Focal Search with secondary ordering S(n) = alpha*H_L + beta*R_s.
// width = cfg.W (standard) or cfg.tuned_fixed_w (w* baseline).
// model == nullptr -> plain focal with secondary = h_a (ties toward goal).
SearchResult focal_fixed(const GridMap& m, const Instance& ins, const Config& cfg,
                         const GuidanceModel* model, double width);
} // namespace cadfs
