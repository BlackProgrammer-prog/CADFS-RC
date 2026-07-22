#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include "grid_map.hpp"
#include "heuristics.hpp"

namespace cadfs {

// Learned-guidance interface. eval() returns the ensemble-mean normalized
// priority H_L(n) in [0,1] and the ensemble variance sigma^2(n).
struct GuidanceModel {
    virtual ~GuidanceModel() = default;
    virtual void eval(const GridMap& m, int node, int goal,
                      double& H_L, double& variance) const = 0;
};

// Confidence C(n) = exp(-sigma^2 / tau_c)  (paper Sec. 7.2)
inline double confidence_from_variance(double variance, double tau_c) {
    return std::exp(-variance / tau_c);
}

// Mock guidance for engine development and unit tests: H_L is the normalized
// anchor heuristic corrupted by deterministic pseudo-noise; variance grows with
// the noise scale so that CADFS's confidence machinery is exercised end-to-end.
class MockGuidance final : public GuidanceModel {
public:
    MockGuidance(double h_norm, double noise, uint64_t seed = 1)
        : h_norm_(h_norm), noise_(noise), seed_(seed) {}
    void eval(const GridMap& m, int node, int goal,
              double& H_L, double& variance) const override {
        const double base = std::min(1.0, anchor_h(m, node, goal, 8) / h_norm_);
        // splitmix64 hash -> uniform in [-1,1], deterministic per (node, seed)
        uint64_t z = (uint64_t)node * 0x9E3779B97F4A7C15ull + seed_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const double u = 2.0 * ((z >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
        H_L = std::min(1.0, std::max(0.0, base + noise_ * u));
        variance = noise_ * noise_ * 0.25 * (0.5 + 0.5 * std::abs(u));
    }
private:
    double h_norm_;   // normalization scale (e.g., map diagonal)
    double noise_;    // 0 = perfect-ish guidance, larger = degraded (OOD proxy)
    uint64_t seed_;
};

} // namespace cadfs
