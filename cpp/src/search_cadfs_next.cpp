#include <cadfs/search_cadfs_next.hpp>

#include <cadfs/heuristics.hpp>
#include <cadfs/risk.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cadfs {

namespace {

struct OpenKey {
    double f;
    int id;

    bool operator<(const OpenKey& other) const {
        return f != other.f ? f < other.f : id < other.id;
    }
};

struct NodeEval {
    FusionResult fusion;
    double confidence = 0.0;
    double structural_risk = 0.0;
};

double unit(double value, double invalid_value = 1.0) {
    if (!std::isfinite(value)) return invalid_value;
    return std::clamp(value, 0.0, 1.0);
}

double normalized(double value, double scale) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
        return 0.0;
    }
    return unit(value / scale, 0.0);
}

std::vector<int> reconstruct(const std::vector<int>& parent, int goal) {
    std::vector<int> path;
    for (int node = goal; node != -1; node = parent[node]) {
        path.push_back(node);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace

SearchResult cadfs_next_search(
        const GridMap& m,
        const Instance& ins,
        const Config& cfg,
        const ExpertFusion& fusion,
        const ConfidenceEstimator& confidence_estimator,
        const FocalWidthController& controller,
        const IterationLogger* logger) {
    const auto started_at = std::chrono::steady_clock::now();
    SearchResult result;

    if (!std::isfinite(cfg.W) || cfg.W < 1.0) {
        throw std::invalid_argument(__func__);
    }
    if (cfg.connectivity != 4 && cfg.connectivity != 8) {
        throw std::invalid_argument(__func__);
    }
    if (!m.passable(ins.start_x, ins.start_y) ||
        !m.passable(ins.goal_x, ins.goal_y)) {
        return result;
    }

    const int start = m.idx(ins.start_x, ins.start_y);
    const int goal = m.idx(ins.goal_x, ins.goal_y);
    const double map_scale = std::max(
            1.0, std::hypot(static_cast<double>(m.width()),
                            static_cast<double>(m.height())));

    std::vector<double> g(m.size(), INF);
    std::vector<double> f_value(m.size(), INF);
    std::vector<int> parent(m.size(), -1);
    std::vector<uint8_t> closed(m.size(), 0);
    std::set<OpenKey> open;

    auto h = [&](int node) {
        return anchor_h(m, node, goal, cfg.connectivity);
    };

    std::unordered_map<int, NodeEval> cache;

    auto evaluate_node = [&](int node) -> const NodeEval& {
        const auto cached = cache.find(node);
        if (cached != cache.end()) return cached->second;

        const double anchor = h(node);
        const double anchor_normalized = normalized(anchor, map_scale);
        NodeEval evaluation;

        try {
            evaluation.fusion = fusion.evaluate(
                    ExpertContext{m, node, start, goal,
                                  anchor, anchor_normalized});
        } catch (const std::exception&) {
            evaluation.fusion.fused_prediction = anchor_normalized;
            evaluation.fusion.intra_uncertainty = 1.0;
            evaluation.fusion.inter_disagreement = 1.0;
            evaluation.fusion.predictions.push_back(
                    ExpertPrediction{std::string{}, anchor_normalized, 1.0, true});
            evaluation.fusion.normalized_weights.push_back(1.0);
        }

        evaluation.fusion.fused_prediction =
                unit(evaluation.fusion.fused_prediction, anchor_normalized);
        evaluation.fusion.intra_uncertainty =
                unit(evaluation.fusion.intra_uncertainty);
        evaluation.fusion.inter_disagreement =
                unit(evaluation.fusion.inter_disagreement);

        evaluation.structural_risk = unit(risk_combined(
                m, node, evaluation.fusion.fused_prediction, anchor, cfg,
                static_cast<uint64_t>(start) * 1315423911u ^
                static_cast<uint64_t>(goal)));

        ConfidenceSignals signals;
        signals.intra_uncertainty = evaluation.fusion.intra_uncertainty;
        signals.inter_disagreement = evaluation.fusion.inter_disagreement;
        signals.structural_risk = evaluation.structural_risk;
        signals.model_reference_disagreement = std::abs(
                evaluation.fusion.fused_prediction - anchor_normalized);

        evaluation.confidence = cfg.confidence_enabled
                ? unit(confidence_estimator.estimate(signals), 0.0)
                : 1.0;

        return cache.emplace(node, std::move(evaluation)).first->second;
    };

    const int window_size = std::max(1, cfg.K);
    std::vector<uint8_t> fallback_window(
            static_cast<std::size_t>(window_size), 0);
    int fallback_position = 0;
    int fallback_sum = 0;
    int64_t fallback_events = 0;

    g[start] = 0.0;
    f_value[start] = h(start);
    open.insert({f_value[start], start});

    double sum_width = 0.0;
    double minimum_width = INF;
    double maximum_width = -INF;
    double previous_width = -1.0;
    double sum_width_change = 0.0;
    double sum_confidence = 0.0;
    double sum_risk = 0.0;
    int64_t iterations = 0;
    std::size_t previous_focal_size = 0;

    int neighbor_ids[8];
    double neighbor_costs[8];

    while (!open.empty()) {
        const int anchor_node = open.begin()->id;
        const double f_min = open.begin()->f;
        const double base_bound = cfg.W * f_min;

        double confidence_mean = 0.0;
        double risk_mean = 0.0;
        double intra_mean = 0.0;
        double inter_mean = 0.0;
        int sample_count = 0;
        std::size_t base_focal_size = 0;

        for (auto it = open.begin();
             it != open.end() && it->f <= base_bound;
             ++it) {
            ++base_focal_size;
            if (sample_count >= std::max(1, cfg.L)) continue;

            const NodeEval& evaluation = evaluate_node(it->id);
            confidence_mean += evaluation.confidence;
            risk_mean += evaluation.structural_risk;
            intra_mean += evaluation.fusion.intra_uncertainty;
            inter_mean += evaluation.fusion.inter_disagreement;
            ++sample_count;
        }

        if (sample_count > 0) {
            const double denominator = static_cast<double>(sample_count);
            confidence_mean /= denominator;
            risk_mean /= denominator;
            intra_mean /= denominator;
            inter_mean /= denominator;
        }

        const double fallback_frequency =
                static_cast<double>(fallback_sum) /
                static_cast<double>(fallback_window.size());

        SearchState state;
        state.confidence = unit(confidence_mean, 0.0);
        state.intra_uncertainty = unit(intra_mean);
        state.inter_disagreement = unit(inter_mean);
        state.expert_disagreement = state.inter_disagreement;
        state.structural_risk = unit(risk_mean);
        state.fallback_frequency = unit(fallback_frequency, 0.0);
        state.open_size_normalized = normalized(
                static_cast<double>(open.size()),
                static_cast<double>(std::max(1, m.size())));
        state.base_focal_size_normalized = normalized(
                static_cast<double>(base_focal_size),
                static_cast<double>(std::max<std::size_t>(1, open.size())));
        state.previous_focal_size_normalized = normalized(
                static_cast<double>(previous_focal_size),
                static_cast<double>(std::max<std::size_t>(1, open.size())));
        state.f_min_normalized = normalized(f_min, map_scale);
        state.current_g_normalized = normalized(g[anchor_node], map_scale);
        state.current_h_normalized = normalized(h(anchor_node), map_scale);
        state.branching_normalized = normalized(
                static_cast<double>(m.degree(anchor_node, cfg.connectivity)),
                cfg.connectivity == 4 ? 4.0 : 8.0);
        state.search_progress = normalized(
                static_cast<double>(result.expansions),
                static_cast<double>(result.expansions + open.size()));

        const double raw_width = controller.raw_width(state, cfg.W);
        const double width = SafetyProjection::project(raw_width, cfg.W);
        const double focal_bound = width * f_min;

        int learned_node = -1;
        double best_score = INF;
        bool goal_in_focal = false;
        std::size_t focal_size = 0;

        for (auto it = open.begin();
             it != open.end() && it->f <= focal_bound;
             ++it) {
            ++focal_size;
            const int node = it->id;

            if (node == goal) {
                learned_node = node;
                goal_in_focal = true;
                continue;
            }

            if (goal_in_focal) continue;

            const NodeEval& evaluation = evaluate_node(node);
            const double score =
                    cfg.alpha * evaluation.fusion.fused_prediction +
                    cfg.beta * evaluation.structural_risk;

            if (score < best_score) {
                best_score = score;
                learned_node = node;
            }
        }

        // The anchor node is always in FOCAL because width >= 1.
        if (learned_node < 0) learned_node = anchor_node;

        int selected_node = learned_node;
        bool fallback_triggered = false;

        if (!goal_in_focal && cfg.fallback_enabled) {
            const NodeEval& learned = evaluate_node(learned_node);
            const double reference_disagreement = std::abs(
                    learned.fusion.fused_prediction -
                    normalized(h(learned_node), map_scale));

            if (learned.confidence < cfg.theta_c ||
                learned.structural_risk > cfg.theta_r ||
                reference_disagreement > cfg.theta_dev) {
                selected_node = anchor_node;
                fallback_triggered = true;
                ++fallback_events;
            }
        }

        const int fallback_value = fallback_triggered ? 1 : 0;
        fallback_sum += fallback_value -
                        fallback_window[fallback_position];
        fallback_window[fallback_position] =
                static_cast<uint8_t>(fallback_value);
        fallback_position =
                (fallback_position + 1) % window_size;

        ++iterations;
        sum_width += width;
        minimum_width = std::min(minimum_width, width);
        maximum_width = std::max(maximum_width, width);
        if (previous_width >= 0.0) {
            sum_width_change += std::abs(width - previous_width);
        }
        previous_width = width;
        previous_focal_size = focal_size;
        sum_confidence += state.confidence;
        sum_risk += state.structural_risk;

        if (logger && *logger) {
            const NodeEval& selected = evaluate_node(selected_node);
            IterationLog log;
            log.iteration = iterations;
            log.open_size = open.size();
            log.base_focal_size = base_focal_size;
            log.focal_size = focal_size;
            log.f_min = f_min;
            log.controller_raw_output = raw_width;
            log.controller_safe_output = width;
            log.expert_weights = selected.fusion.normalized_weights;
            for (const ExpertPrediction& prediction :
                 selected.fusion.predictions) {
                log.expert_predictions.push_back(prediction.mean);
            }
            log.intra_expert_uncertainty =
                    selected.fusion.intra_uncertainty;
            log.inter_expert_disagreement =
                    selected.fusion.inter_disagreement;
            log.confidence = selected.confidence;
            log.structural_risk = selected.structural_risk;
            log.fallback_frequency = fallback_frequency;
            log.selected_node = selected_node;
            log.selected_by_anchor = fallback_triggered;
            log.fallback_triggered = fallback_triggered;
            log.g = g[selected_node];
            log.h = h(selected_node);
            log.f = f_value[selected_node];
            (*logger)(log);
        }

        open.erase({f_value[selected_node], selected_node});
        closed[selected_node] = 1;
        ++result.expansions;

        if (selected_node == goal) {
            result.found = true;
            result.cost = g[goal];
            result.path = reconstruct(parent, goal);
            break;
        }

        const int neighbor_count = m.neighbors(
                selected_node, cfg.connectivity,
                neighbor_ids, neighbor_costs);

        for (int i = 0; i < neighbor_count; ++i) {
            const int neighbor = neighbor_ids[i];
            const double tentative =
                    g[selected_node] + neighbor_costs[i];

            if (tentative >= g[neighbor]) continue;

            if (g[neighbor] < INF && !closed[neighbor]) {
                open.erase({f_value[neighbor], neighbor});
            }

            // Reopen a node when a better path is discovered. This keeps the
            // safety/search layer independent of learned node ordering.
            closed[neighbor] = 0;
            g[neighbor] = tentative;
            f_value[neighbor] = tentative + h(neighbor);
            parent[neighbor] = selected_node;
            open.insert({f_value[neighbor], neighbor});
            ++result.generated;
        }
    }

    result.runtime_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started_at).count();

    if (iterations > 0) {
        const double count = static_cast<double>(iterations);
        result.fallback_rate =
                static_cast<double>(fallback_events) / count;
        result.mean_w = sum_width / count;
        result.min_w = minimum_width;
        result.max_w = maximum_width;
        result.mean_abs_dw = iterations > 1
                ? sum_width_change /
                  static_cast<double>(iterations - 1)
                : 0.0;
        result.mean_C = sum_confidence / count;
        result.mean_R = sum_risk / count;
    }

    return result;
}

} // namespace cadfs
