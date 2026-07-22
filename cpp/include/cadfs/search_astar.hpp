#pragma once
#include "grid_map.hpp"
#include "search.hpp"

namespace cadfs {
// weight = 1.0 -> A*; weight > 1 -> Weighted A* (bound: cost <= weight * C*).
SearchResult astar(const GridMap& m, const Instance& ins, const Config& cfg,
                   double weight = 1.0);
// Exact optimal cost via Dijkstra (label generation / test oracle).
SearchResult dijkstra(const GridMap& m, const Instance& ins, const Config& cfg);
// Dijkstra from a single source to ALL cells (cost-to-go labels d*(cell, goal)).
std::vector<double> dijkstra_all(const GridMap& m, int source, int connectivity);
} // namespace cadfs
