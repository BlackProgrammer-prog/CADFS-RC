#pragma once
#include "grid_map.hpp"
#include "guidance.hpp"
#include "search.hpp"

namespace cadfs {
// Frozen legacy CADFS-RC baseline. Keep its behaviour unchanged so every new
// search/controller variant can be compared against the original method.
SearchResult cadfs_search(
        const GridMap& m,
        const Instance& ins,
        const Config& cfg,
        const GuidanceModel& model,
        double tau_c);
} // namespace cadfs
