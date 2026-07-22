#pragma once
#include <algorithm>
#include <cmath>
#include "grid_map.hpp"

namespace cadfs {

inline double manhattan(const GridMap& m, int a, int b) {
    return std::abs(m.x_of(a) - m.x_of(b)) + std::abs(m.y_of(a) - m.y_of(b));
}

// Admissible for 8-connected grids with unit cardinal and sqrt(2) diagonal costs.
inline double octile(const GridMap& m, int a, int b) {
    const double dx = std::abs(m.x_of(a) - m.x_of(b));
    const double dy = std::abs(m.y_of(a) - m.y_of(b));
    static const double SQRT2 = 1.4142135623730951;
    return (dx + dy) + (SQRT2 - 2.0) * std::min(dx, dy);
}

// Anchor heuristic h_a: admissible for the given connectivity.
inline double anchor_h(const GridMap& m, int a, int b, int connectivity) {
    return connectivity == 4 ? manhattan(m, a, b) : octile(m, a, b);
}

} // namespace cadfs
