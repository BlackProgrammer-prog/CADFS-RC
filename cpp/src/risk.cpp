#include "cadfs/risk.hpp"
#include "cadfs/heuristics.hpp"

namespace cadfs {

double risk_obs(const GridMap& m, int node, int radius) {
    return m.obstacle_density(node, radius);
}

double risk_mob(const GridMap& m, int node, int connectivity) {
    const int deg_max = connectivity == 4 ? 4 : 8;
    return 1.0 - (double)m.degree(node, connectivity) / (double)deg_max;
}

static double unit_hash(uint64_t z) { // splitmix64 -> [0,1)
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (z >> 11) * (1.0 / 9007199254740992.0);
}

double risk_combined(const GridMap& m, int node, double H_L, double h_a,
                     const Config& cfg, uint64_t permute_salt) {
    using RM = Config::RiskMode;
    switch (cfg.risk_mode) {
        case RM::Off:    return 0.0;
        case RM::Random: return unit_hash((uint64_t)node ^ cfg.risk_seed);
        case RM::Permuted: {
            // Instance-level destruction of the state<->risk association: the true
            // risk of a pseudo-random *other* node stands in for this node's risk.
            const int other = (int)(unit_hash((uint64_t)node ^ permute_salt) * m.size());
            const double ho = anchor_h(m, other, node, cfg.connectivity); // arbitrary pairing
            const double Ro = cfg.lambda_obs * risk_obs(m, other, cfg.risk_radius)
                            + cfg.lambda_mob * risk_mob(m, other, cfg.connectivity)
                            + cfg.lambda_dev * risk_dev(H_L, h_ref_norm(ho, cfg));
            return std::min(1.0, std::max(0.0, Ro));
        }
        case RM::Normal: default: {
            const double R = cfg.lambda_obs * risk_obs(m, node, cfg.risk_radius)
                           + cfg.lambda_mob * risk_mob(m, node, cfg.connectivity)
                           + cfg.lambda_dev * risk_dev(H_L, h_ref_norm(h_a, cfg));
            return std::min(1.0, std::max(0.0, R));
        }
    }
}

} // namespace cadfs
