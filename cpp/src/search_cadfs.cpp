#include "cadfs/search_cadfs.hpp"
#include "cadfs/heuristics.hpp"
#include "cadfs/risk.hpp"
#include <chrono>
#include <cmath>
#include <set>
#include <unordered_map>

namespace cadfs {

namespace {
struct OpenKey {
    double f; int id;
    bool operator<(const OpenKey& o) const { return f != o.f ? f < o.f : id < o.id; }
};
struct NodeEval { double H_L, C, R; };

std::vector<int> reconstruct(const std::vector<int>& parent, int goal) {
    std::vector<int> path;
    for (int v = goal; v != -1; v = parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}

double controller_width(const Config& cfg, double C_t, double R_t, double F_t) {
    switch (cfg.controller) {
        case ControllerType::Fixed:      return cfg.W;
        case ControllerType::TunedFixed: return cfg.tuned_fixed_w;
        case ControllerType::Linear: {
            const double s = cfg.lin_a * C_t + cfg.lin_b * (1.0 - R_t) + cfg.lin_c * (1.0 - F_t);
            return 1.0 + (cfg.W - 1.0) * std::min(1.0, std::max(0.0, s));
        }
        case ControllerType::Multiplicative: default:
            return 1.0 + (cfg.W - 1.0) * C_t * (1.0 - R_t) * (1.0 - F_t);
    }
}
} // namespace

SearchResult cadfs_search(const GridMap& m, const Instance& ins, const Config& cfg,
                          const GuidanceModel& model, double tau_c) {
    const auto t0 = std::chrono::steady_clock::now();
    SearchResult res;
    const int start = m.idx(ins.start_x, ins.start_y);
    const int goal  = m.idx(ins.goal_x, ins.goal_y);
    if (!m.passable(ins.start_x, ins.start_y) || !m.passable(ins.goal_x, ins.goal_y)) return res;

    std::vector<double> g(m.size(), INF), fval(m.size(), INF);
    std::vector<int> parent(m.size(), -1);
    std::vector<uint8_t> closed(m.size(), 0);
    std::set<OpenKey> open;
    auto h = [&](int n) { return anchor_h(m, n, goal, cfg.connectivity); };

    // per-node model/risk cache: eval each generated node at most once
    std::unordered_map<int, NodeEval> cache;
    auto eval_node = [&](int n) -> const NodeEval& {
        auto it = cache.find(n);
        if (it != cache.end()) return it->second;
        double H_L, var; model.eval(m, n, goal, H_L, var);
        NodeEval e;
        e.H_L = H_L;
        e.C = cfg.confidence_enabled ? confidence_from_variance(var, tau_c) : 1.0;
        e.R = risk_combined(m, n, H_L, h(n), cfg,
                            /*permute_salt=*/(uint64_t)start * 1315423911u ^ (uint64_t)goal);
        return cache.emplace(n, e).first->second;
    };

    // fallback sliding window of length K
    std::vector<uint8_t> fb_win((size_t)std::max(1, cfg.K), 0);
    int fb_pos = 0, fb_sum = 0;
    int64_t fb_events = 0;

    g[start] = 0.0; fval[start] = h(start);
    open.insert({fval[start], start});
    int nbr[8]; double nc[8];

    double sum_w = 0, min_w = INF, max_w = -INF, prev_w = -1, sum_dw = 0, sum_C = 0, sum_R = 0;
    int64_t iters = 0;

    while (!open.empty()) {
        const double f_min = open.begin()->f;

        // ---- Q_t: anchor Top-L prefix of B_t (Sec. 5.2 / 7.4) ----
        double C_t = 0, R_t = 0; int q = 0;
        const double base_bound = cfg.W * f_min;
        for (auto it = open.begin(); it != open.end() && it->f <= base_bound && q < cfg.L; ++it, ++q) {
            const NodeEval& e = eval_node(it->id);
            C_t += e.C; R_t += e.R;
        }
        C_t = q ? C_t / q : 0.0; R_t = q ? R_t / q : 0.0;
        const double F_t = (double)fb_sum / (double)fb_win.size();

        // ---- adaptive width ----
        const double w_t = controller_width(cfg, C_t, R_t, F_t);
        ++iters; sum_w += w_t; min_w = std::min(min_w, w_t); max_w = std::max(max_w, w_t);
        if (prev_w >= 0) sum_dw += std::abs(w_t - prev_w);
        prev_w = w_t; sum_C += C_t; sum_R += R_t;

        // ---- FOCAL_t and learned-risk candidate ----
        const double bound = w_t * f_min;
        int n_learned = -1; double best_s = INF; bool goal_in_focal = false;
        for (auto it = open.begin(); it != open.end() && it->f <= bound; ++it) {
            const int n = it->id;
            if (n == goal) { goal_in_focal = true; n_learned = n; break; }
            const NodeEval& e = eval_node(n);
            const double s = cfg.alpha * e.H_L + cfg.beta * e.R;
            if (s < best_s) { best_s = s; n_learned = n; }
        }

        // ---- fixed threshold fallback rule (Sec. 7.5) ----
        int n_expand = n_learned; int fb = 0;
        if (!goal_in_focal && cfg.fallback_enabled) {
            const NodeEval& e = eval_node(n_learned);
            const double Rdev = risk_dev(e.H_L, h_ref_norm(h(n_learned), cfg));
            if (e.C < cfg.theta_c || e.R > cfg.theta_r || Rdev > cfg.theta_dev) {
                n_expand = open.begin()->id; // anchor-best node, trivially in FOCAL
                fb = 1; ++fb_events;
            }
        }
        fb_sum += fb - fb_win[fb_pos]; fb_win[fb_pos] = (uint8_t)fb;
        fb_pos = (fb_pos + 1) % (int)fb_win.size();

        // ---- expand ----
        const int u = n_expand;
        open.erase({fval[u], u});
        closed[u] = 1;
        ++res.expansions;
        if (u == goal) {
            res.found = true; res.cost = g[u]; res.path = reconstruct(parent, goal);
            break;
        }
        const int k = m.neighbors(u, cfg.connectivity, nbr, nc);
        for (int i = 0; i < k; ++i) {
            const int v = nbr[i];
            if (closed[v]) continue;
            const double t = g[u] + nc[i];
            if (t < g[v]) {
                if (g[v] < INF) open.erase({fval[v], v});
                g[v] = t; fval[v] = t + h(v); parent[v] = u;
                open.insert({fval[v], v});
                ++res.generated;
            }
        }
    }

    res.runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (iters > 0) {
        res.fallback_rate = (double)fb_events / (double)iters;
        res.mean_w = sum_w / iters; res.min_w = min_w; res.max_w = max_w;
        res.mean_abs_dw = iters > 1 ? sum_dw / (iters - 1) : 0.0;
        res.mean_C = sum_C / iters; res.mean_R = sum_R / iters;
    }
    return res;
}

} // namespace cadfs
