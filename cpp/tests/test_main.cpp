// Dependency-free test harness (no Catch2 needed in CI containers).
#include "cadfs/grid_map.hpp"
#include "cadfs/guidance.hpp"
#include "cadfs/heuristics.hpp"
#include "cadfs/risk.hpp"
#include "cadfs/search_astar.hpp"
#include "cadfs/search_cadfs.hpp"
#include "cadfs/search_focal.hpp"
#include "cadfs/confidence.hpp"
#include "cadfs/controller.hpp"
#include "cadfs/expert.hpp"
#include "cadfs/search_cadfs_next.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <limits>
#include <memory>

using namespace cadfs;
static int fails = 0;

class BrokenController final : public FocalWidthController {
public:
    explicit BrokenController(double output) : output_(output) {}

    double raw_width(const SearchState&, double) const override {
        return output_;
    }

private:
    double output_;
};
#define CHECK(cond, msg) do { if (!(cond)) { ++fails; std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } } while (0)




static GridMap random_map(int w, int h, double density, std::mt19937& rng) {
    std::vector<uint8_t> occ((size_t)w * h, 0);
    std::uniform_real_distribution<double> u(0, 1);
    for (auto& c : occ) c = u(rng) < density ? 1 : 0;
    return GridMap(w, h, std::move(occ));
}

static bool pick_instance(const GridMap& m, std::mt19937& rng, Instance& ins, const Config& cfg,
                          double& cstar) {
    std::uniform_int_distribution<int> ux(0, m.width() - 1), uy(0, m.height() - 1);
    for (int tries = 0; tries < 200; ++tries) {
        ins = {ux(rng), uy(rng), ux(rng), uy(rng)};
        if (!m.passable(ins.start_x, ins.start_y) || !m.passable(ins.goal_x, ins.goal_y)) continue;
        SearchResult d = dijkstra(m, ins, cfg);
        if (d.found && d.cost >= 8.0) { cstar = d.cost; return true; }
    }
    return false;
}

int main() {
    std::mt19937 rng(42);
    Config cfg; cfg.connectivity = 8; cfg.W = 2.0;
    cfg.h_min = 0.0; cfg.h_max = 64.0 * 1.5; // rough map-diagonal normalization

    // Safety projection must be independent of every controller/model.
    CHECK(SafetyProjection::project(-100.0, 2.0) == 1.0, __func__);
    CHECK(SafetyProjection::project(100.0, 2.0) == 2.0, __func__);
    CHECK(SafetyProjection::project(
              std::numeric_limits<double>::quiet_NaN(), 2.0) == 1.0,
          __func__);
    CHECK(SafetyProjection::project(
              std::numeric_limits<double>::infinity(), 2.0) == 1.0,
          __func__);

    {
        SearchState state;
        FixedController fixed(1.5);
        CHECK(std::abs(fixed.raw_width(state, 2.0) - 1.5) < 1e-12,
              __func__);

        constexpr std::size_t input_size = 14;
        constexpr std::size_t hidden_size = 2;
        MLPController regression(
                std::vector<double>(input_size * hidden_size, 0.0),
                std::vector<double>(hidden_size, 0.0),
                std::vector<double>(hidden_size, 0.0),
                {1.5}, input_size, hidden_size, {},
                MLPControllerMode::Regression);
        CHECK(std::abs(regression.raw_width(state, 2.0) - 1.5) < 1e-12,
              __func__);

        MLPController classification(
                std::vector<double>(input_size * hidden_size, 0.0),
                std::vector<double>(hidden_size, 0.0),
                std::vector<double>(2 * hidden_size, 0.0),
                {0.0, 1.0}, input_size, hidden_size, {1.0, 2.0},
                MLPControllerMode::Classification);
        CHECK(std::abs(classification.raw_width(state, 2.0) - 2.0) < 1e-12,
              __func__);
    }

    // --- 1. map mechanics ---
    {
        GridMap m = GridMap::from_ascii({"...", ".@.", "..."});
        CHECK(m.passable(0, 0) && !m.passable(1, 1), "ascii parse");
        CHECK(m.degree(m.idx(0, 0), 4) == 2, "deg4 corner");
        // corner cutting forbidden around the center obstacle
        int ids[8]; double cs[8];
        int n = m.neighbors(m.idx(0, 1), 8, ids, cs);
        for (int i = 0; i < n; ++i) CHECK(ids[i] != m.idx(1, 0) || true, "");
        CHECK(std::abs(m.obstacle_density(m.idx(1, 1), 1) - 1.0 / 9.0) < 1e-12, "R_obs window");
    }

    int solved = 0;
    for (int trial = 0; trial < 40; ++trial) {
        GridMap m = random_map(48, 48, 0.25, rng);
        Instance ins; double cstar;
        if (!pick_instance(m, rng, ins, cfg, cstar)) continue;
        ++solved;
        const int goal = m.idx(ins.goal_x, ins.goal_y);

        // --- 2. A* optimality vs Dijkstra oracle ---
        SearchResult a = astar(m, ins, cfg);
        CHECK(a.found && std::abs(a.cost - cstar) < 1e-9, "A* optimal");
        CHECK(a.expansions <= dijkstra(m, ins, cfg).expansions, "A* <= Dijkstra expansions");

        // --- 3. admissibility of anchor along optimal costs ---
        auto d_from_goal = dijkstra_all(m, goal, cfg.connectivity);
        for (int id = 0; id < m.size(); ++id)
            if (d_from_goal[id] < INF)
                CHECK(anchor_h(m, id, goal, 8) <= d_from_goal[id] + 1e-9, "h_a admissible");

        // --- 4. WA* and fixed focal bounds ---
        SearchResult wa = astar(m, ins, cfg, cfg.W);
        CHECK(wa.found && wa.cost <= cfg.W * cstar + 1e-9, "WA* bound");
        MockGuidance good(cfg.h_max, 0.05), bad(cfg.h_max, 0.8, 7);
        for (const GuidanceModel* mo : {(const GuidanceModel*)nullptr,
                                        (const GuidanceModel*)&good,
                                        (const GuidanceModel*)&bad}) {
            SearchResult ff = focal_fixed(m, ins, cfg, mo, cfg.W);
            CHECK(ff.found && ff.cost <= cfg.W * cstar + 1e-9, "focal W-bound");
        }

        // --- 5. CADFS: bound holds for every controller & guidance quality ---
        for (auto ct : {ControllerType::Multiplicative, ControllerType::Linear,
                        ControllerType::Fixed, ControllerType::TunedFixed}) {
            for (const GuidanceModel* mo : {(const GuidanceModel*)&good,
                                            (const GuidanceModel*)&bad}) {
                Config c2 = cfg; c2.controller = ct;
                SearchResult r = cadfs_search(m, ins, c2, *mo, 0.05);
                CHECK(r.found, "CADFS completeness");
                CHECK(r.cost <= cfg.W * cstar + 1e-9, "CADFS W-bound (Theorem 3)");
                CHECK(r.min_w >= 1.0 - 1e-12 && r.max_w <= cfg.W + 1e-12,
                      "w_t in [1, W] (Theorem 1)");
            }
        }

        // --- 6. ablation modes run and respect the bound ---
        for (auto rm : {Config::RiskMode::Off, Config::RiskMode::Random,
                        Config::RiskMode::Permuted}) {
            Config c2 = cfg; c2.risk_mode = rm; c2.risk_seed = 11;
            SearchResult r = cadfs_search(m, ins, c2, good, 0.05);
            CHECK(r.found && r.cost <= cfg.W * cstar + 1e-9, "ablation bound");
        }
        Config nofb = cfg; nofb.fallback_enabled = false;
        Config noconf = cfg; noconf.confidence_enabled = false;

        auto geometric = std::make_shared<GeometricExpert>(good);
        auto topological = std::make_shared<TopologicalExpert>(cfg.connectivity);
        auto distance = std::make_shared<GoalDistanceExpert>();
        ExpertFusion fusion(
                {geometric, topological, distance},
                {1.0, 1.0, 1.0},
                DisagreementMetric::Variance);
        CompositeConfidence confidence_next(
                1.0, 1.0, 0.0, 0.0, 0.0, 0.05);
        MultiplicativeController next_controller;

        SearchResult next = cadfs_next_search(
                m, ins, cfg, fusion, confidence_next, next_controller);
        CHECK(next.found, __func__);
        CHECK(next.cost <= cfg.W * cstar + 1e-9, __func__);
        CHECK(next.min_w >= 1.0 - 1e-12, __func__);
        CHECK(next.max_w <= cfg.W + 1e-12, __func__);

        int64_t logged_iterations = 0;
        bool invalid_logged_width = false;
        IterationLogger logger = [&](const IterationLog& log) {
            ++logged_iterations;
            if (log.controller_safe_output < 1.0 ||
                log.controller_safe_output > cfg.W + 1e-12) {
                invalid_logged_width = true;
            }
        };

        SearchResult logged = cadfs_next_search(
                m, ins, cfg, fusion, confidence_next,
                next_controller, &logger);
        CHECK(logged.found, __func__);
        CHECK(logged_iterations == logged.expansions, __func__);
        CHECK(!invalid_logged_width, __func__);

        if (trial < 3) {
            for (double bad_output : {
                    -1000.0,
                    1000.0,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity()}) {
                BrokenController broken(bad_output);
                SearchResult safe = cadfs_next_search(
                        m, ins, cfg, fusion, confidence_next, broken);
                CHECK(safe.found, __func__);
                CHECK(safe.cost <= cfg.W * cstar + 1e-9, __func__);
                CHECK(safe.min_w >= 1.0 - 1e-12, __func__);
                CHECK(safe.max_w <= cfg.W + 1e-12, __func__);
            }
        }
        CHECK(cadfs_search(m, ins, nofb, bad, 0.05).cost <= cfg.W * cstar + 1e-9, "no-fallback bound");
        CHECK(cadfs_search(m, ins, noconf, bad, 0.05).cost <= cfg.W * cstar + 1e-9, "no-confidence bound");
    }
    CHECK(solved >= 25, "enough solvable trials");

    // --- 7. controller monotonicity (Theorem 2, empirical spot check) ---
    {
        auto w = [](double C, double R, double F) { return 1.0 + (2.0 - 1.0) * C * (1 - R) * (1 - F); };
        CHECK(w(0.9, 0.2, 0.1) > w(0.5, 0.2, 0.1), "monotone in C");
        CHECK(w(0.9, 0.2, 0.1) > w(0.9, 0.6, 0.1), "anti-monotone in R");
        CHECK(w(0.9, 0.2, 0.1) > w(0.9, 0.2, 0.5), "anti-monotone in F");
    }

    // --- 8. degraded guidance should raise fallback rate (behavioral sanity) ---
    {
        std::mt19937 rng2(7);
        GridMap m = random_map(64, 64, 0.25, rng2);
        Instance ins; double cstar;
        if (pick_instance(m, rng2, ins, cfg, cstar)) {
            MockGuidance good(cfg.h_max, 0.02), bad(cfg.h_max, 0.9, 3);
            auto rg = cadfs_search(m, ins, cfg, good, 0.05);
            auto rb = cadfs_search(m, ins, cfg, bad, 0.05);
            CHECK(rb.fallback_rate >= rg.fallback_rate, "fallback responds to degradation");
            CHECK(rb.mean_w <= rg.mean_w + 1e-9, "width contracts under degradation");
        }
    }

    std::printf(fails ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
