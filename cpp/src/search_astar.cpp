#include "cadfs/search_astar.hpp"
#include "cadfs/heuristics.hpp"
#include <chrono>
#include <queue>

namespace cadfs {

namespace {
struct QItem { double f; int id; };
struct QCmp { bool operator()(const QItem& a, const QItem& b) const { return a.f > b.f; } };

std::vector<int> reconstruct(const std::vector<int>& parent, int goal) {
    std::vector<int> path;
    for (int v = goal; v != -1; v = parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}
} // namespace

SearchResult astar(const GridMap& m, const Instance& ins, const Config& cfg, double weight) {
    const auto t0 = std::chrono::steady_clock::now();
    SearchResult res;
    const int start = m.idx(ins.start_x, ins.start_y);
    const int goal  = m.idx(ins.goal_x, ins.goal_y);
    if (!m.passable(ins.start_x, ins.start_y) || !m.passable(ins.goal_x, ins.goal_y)) return res;

    std::vector<double> g(m.size(), INF);
    std::vector<int> parent(m.size(), -1);
    std::vector<uint8_t> closed(m.size(), 0);
    std::priority_queue<QItem, std::vector<QItem>, QCmp> open;

    g[start] = 0.0;
    open.push({weight * anchor_h(m, start, goal, cfg.connectivity), start});
    int nbr[8]; double nc[8];

    while (!open.empty()) {
        const QItem top = open.top(); open.pop();
        const int u = top.id;
        if (closed[u]) continue;           // lazy deletion
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
                g[v] = t; parent[v] = u;
                open.push({t + weight * anchor_h(m, v, goal, cfg.connectivity), v});
                ++res.generated;
            }
        }
    }
    res.runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return res;
}

SearchResult dijkstra(const GridMap& m, const Instance& ins, const Config& cfg) {
    return astar(m, ins, cfg, 0.0); // weight 0 disables the heuristic
}

std::vector<double> dijkstra_all(const GridMap& m, int source, int connectivity) {
    std::vector<double> d(m.size(), INF);
    std::priority_queue<QItem, std::vector<QItem>, QCmp> open;
    d[source] = 0.0; open.push({0.0, source});
    int nbr[8]; double nc[8];
    while (!open.empty()) {
        const QItem top = open.top(); open.pop();
        if (top.f > d[top.id]) continue;
        const int k = m.neighbors(top.id, connectivity, nbr, nc);
        for (int i = 0; i < k; ++i)
            if (top.f + nc[i] < d[nbr[i]]) { d[nbr[i]] = top.f + nc[i]; open.push({d[nbr[i]], nbr[i]}); }
    }
    return d;
}

} // namespace cadfs
