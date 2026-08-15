#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cadfs {

constexpr double INF = std::numeric_limits<double>::infinity();

struct Instance {
    int start_x = 0, start_y = 0, goal_x = 0, goal_y = 0;
};

enum class ControllerType { Multiplicative, Linear, Fixed, TunedFixed };

struct Config {
    // search
    double W = 2.0;
    int L = 16;
    int K = 50;
    int connectivity = 8;
    // controller
    ControllerType controller = ControllerType::Multiplicative;
    double lin_a = 0.34, lin_b = 0.33, lin_c = 0.33; // a+b+c = 1
    double tuned_fixed_w = 1.5;                       // w* baseline
    // risk
    int risk_radius = 3;
    double lambda_obs = 0.34, lambda_mob = 0.33, lambda_dev = 0.33;
    // risk ablations: Off => R_s = 0; Random ~ U(0,1); Permuted = shuffled per run
    enum class RiskMode { Normal, Off, Random, Permuted } risk_mode = RiskMode::Normal;
    uint64_t risk_seed = 0;
    // ranking S_t = alpha*H_L + beta*R_s, alpha+beta = 1
    double alpha = 1.0, beta = 0.0;  // validated: risk in controller only (beta>0 evaluated as ablation)
    // fallback thresholds
    double theta_c = 0.3, theta_r = 0.8, theta_dev = 0.5;
    bool fallback_enabled = true;
    // confidence ablation: force C_t = 1
    bool confidence_enabled = true;
    // normalization for H_ref (validation-set stats)
    double h_min = 0.0, h_max = 1.0, eps = 1e-6;
};

struct SearchResult {
    bool found = false;
    double cost = INF;
    int64_t expansions = 0;
    int64_t generated = 0;
    double runtime_ms = 0.0;
    std::vector<int> path;          // node ids, start..goal
    // CADFS diagnostics (0/NaN-free defaults for non-CADFS searches)
    double fallback_rate = 0.0;
    double mean_w = 0.0, min_w = 0.0, max_w = 0.0;
    double mean_abs_dw = 0.0;       // oscillation statistic mean |w_t - w_{t-1}|
    double mean_C = 0.0, mean_R = 0.0;
    // Learned-guidance systems telemetry. A model evaluation is one cache
    // miss (one node); member evaluations expose adaptive/full ensembles.
    int64_t model_eval_count = 0;
    int64_t model_member_evals = 0;
    int64_t model_cache_hits = 0;
    double model_cache_hit_rate = 0.0;
    double model_eval_time_ms = 0.0;
};

} // namespace cadfs
