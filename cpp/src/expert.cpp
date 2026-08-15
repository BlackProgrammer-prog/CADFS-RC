//
// Created by HOME on 8/14/2026.
//
#include "cadfs/expert.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cadfs {

    namespace {

        double unit_clip(double value) {
            return std::clamp(value, 0.0, 1.0);
        }

    } // namespace

    ExpertPrediction GeometricExpert::predict(
            const ExpertContext& context) const {

        const GuidanceEvaluation evaluation = guidance_.eval_detailed(
                context.map, context.node, context.goal);

        return {
                name(),
                unit_clip(evaluation.priority),
                std::max(0.0, evaluation.variance),
                true,
                std::max(0, evaluation.member_evaluations)
        };
    }

    ExpertPrediction TopologicalExpert::predict(
            const ExpertContext& context) const {

        const int degree =
                context.map.degree(context.node, connectivity_);

        const int max_degree = connectivity_ == 4 ? 4 : 8;
        const double mobility_risk =
                1.0 - static_cast<double>(degree) / max_degree;

        const double obstacle_risk =
                context.map.obstacle_density(context.node, 3);

        // baseline ساده؛ بعداً می‌تواند با MLP یا GNN جایگزین شود.
        // anchor component باعث می‌شود خروجی معنای cost-to-go خود را حفظ کند.
        const double prediction =
                0.70 * context.h_normalized +
                0.15 * mobility_risk +
                0.15 * obstacle_risk;

        return {
                name(),
                unit_clip(prediction),
                0.0,
                true
        };
    }

    ExpertPrediction GoalDistanceExpert::predict(
            const ExpertContext& context) const {

        return {
                name(),
                unit_clip(context.h_normalized),
                0.0,
                true
        };
    }

    ExpertFusion::ExpertFusion(
            std::vector<std::shared_ptr<const Expert>> experts,
            std::vector<double> weights,
            DisagreementMetric metric)
            : experts_(std::move(experts)),
              weights_(std::move(weights)),
              metric_(metric) {

        if (experts_.empty()) {
            throw std::invalid_argument(
                    "ExpertFusion requires at least one expert");
        }

        if (weights_.empty()) {
            weights_.assign(
                    experts_.size(),
                    1.0 / static_cast<double>(experts_.size()));
        }

        if (weights_.size() != experts_.size()) {
            throw std::invalid_argument(
                    "expert weights size mismatch");
        }

        for (double weight : weights_) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument(
                        "expert weights must be finite and non-negative");
            }
        }

        const double sum =
                std::accumulate(weights_.begin(), weights_.end(), 0.0);

        if (sum <= 0.0) {
            throw std::invalid_argument(
                    "sum of expert weights must be positive");
        }

        for (double& weight : weights_) {
            weight /= sum;
        }
    }

    FusionResult ExpertFusion::evaluate(
            const ExpertContext& context) const {

        FusionResult result;

        for (std::size_t i = 0; i < experts_.size(); ++i) {
            ExpertPrediction prediction =
                    experts_[i]->predict(context);

            if (!prediction.available) {
                continue;
            }

            prediction.mean = unit_clip(prediction.mean);
            prediction.intra_uncertainty =
                    std::max(0.0, prediction.intra_uncertainty);

            result.predictions.push_back(std::move(prediction));
            result.normalized_weights.push_back(weights_[i]);
        }

        if (result.predictions.empty()) {
            throw std::runtime_error("no expert is available");
        }

        double active_weight_sum =
                std::accumulate(
                        result.normalized_weights.begin(),
                        result.normalized_weights.end(),
                        0.0);

        for (double& weight : result.normalized_weights) {
            weight /= active_weight_sum;
        }

        for (std::size_t i = 0;
             i < result.predictions.size();
             ++i) {

            const double weight = result.normalized_weights[i];

            result.fused_prediction +=
                    weight * result.predictions[i].mean;

            result.intra_uncertainty +=
                    weight * result.predictions[i].intra_uncertainty;
        }

        const double mean = result.fused_prediction;

        if (metric_ == DisagreementMetric::Variance) {
            for (std::size_t i = 0;
                 i < result.predictions.size();
                 ++i) {

                const double delta =
                        result.predictions[i].mean - mean;

                result.inter_disagreement +=
                        result.normalized_weights[i] * delta * delta;
            }
        } else if (
                metric_ == DisagreementMetric::MeanAbsoluteDeviation) {

            for (std::size_t i = 0;
                 i < result.predictions.size();
                 ++i) {

                result.inter_disagreement +=
                        result.normalized_weights[i] *
                        std::abs(result.predictions[i].mean - mean);
            }
        } else {
            auto [minimum, maximum] =
                    std::minmax_element(
                            result.predictions.begin(),
                            result.predictions.end(),
                            [](const auto& a, const auto& b) {
                                return a.mean < b.mean;
                            });

            result.inter_disagreement =
                    maximum->mean - minimum->mean;
        }

        result.fused_prediction =
                unit_clip(result.fused_prediction);

        result.inter_disagreement =
                unit_clip(result.inter_disagreement);

        return result;
    }

} // namespace cadfs
