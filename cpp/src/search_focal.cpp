#include "cadfs/search_focal.hpp"
#include "cadfs/heuristics.hpp"
#include "cadfs/risk.hpp"
#include <chrono>
#include <set>
#include <unordered_map>

namespace cadfs {

namespace {
struct OpenKey {
    double f; int id;
    bool operator<(const OpenKey& o) const { return f != o.f ? f < o.f : id < o.id; }
};
std::vector<int> reconstruct(const std::vector<int>& parent, int goal) {
    std::vector<int> path;
    for (int v = goal; v != -1; v = parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}
} // namespace

SearchResult focal_fixed(const GridMap& m, const Instance& ins, const Config& cfg,
                         const GuidanceModel* model, double width) {
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
    // secondary-score cache: evaluate the model at most once per node
    std::unordered_map<int, double> scache;
    auto secondary = [&](int n) {
        auto it = scache.find(n);
        if (it != scache.end()) {
            if (model) ++res.model_cache_hits;
            return it->second;
        }
        double s;
        if (model) {
            const auto eval_started = std::chrono::steady_clock::now();
            const GuidanceEvaluation evaluation =
                    model->eval_detailed(m, n, goal);
            res.model_eval_time_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        eval_started).count();
            ++res.model_eval_count;
            res.model_member_evals += evaluation.member_evaluations;
            const double R = risk_combined(
                    m, n, evaluation.priority, h(n), cfg, 0);
            s = cfg.alpha * evaluation.priority + cfg.beta * R;
        } else {
            s = h(n);
        }
        return scache.emplace(n, s).first->second;
    };
    g[start] = 0.0; fval[start] = h(start);
    open.insert({fval[start], start});
    int nbr[8]; double nc[8];

    while (!open.empty()) {
        const double f_min = open.begin()->f;
        // termination: goal expandable within bound
        // (goal is expanded when selected below; standard condition)
        // --- select argmin secondary within FOCAL = prefix f <= width*f_min ---
        const double bound = width * f_min;
        int best = -1; double best_s = INF;
        for (auto it = open.begin(); it != open.end() && it->f <= bound; ++it) {
            const int n = it->id;
            if (n == goal) { best = n; break; } // goal in FOCAL -> expand it, bound holds
            const double s = secondary(n);
            if (s < best_s) { best_s = s; best = n; }
        }
        const int u = best;
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
            const double t = g[u] + nc[i];
            if (t < g[v]) {
                if (g[v] < INF && !closed[v])
                    open.erase({fval[v], v});
                // Focal ordering is not f-ordering. A previously expanded
                // state can therefore receive a better path and must reopen.
                closed[v] = 0;
                g[v] = t; fval[v] = t + h(v); parent[v] = u;
                open.insert({fval[v], v});
                ++res.generated;
            }
        }
    }
    res.runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const int64_t requests = res.model_eval_count + res.model_cache_hits;
    res.model_cache_hit_rate = requests > 0
            ? static_cast<double>(res.model_cache_hits) / requests : 0.0;
    return res;
}

} // namespace cadfs
