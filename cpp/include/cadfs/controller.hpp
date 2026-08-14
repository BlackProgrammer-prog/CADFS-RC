#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cadfs {

    struct SearchState {
        double confidence = 0.0;
        double intra_uncertainty = 0.0;
        double inter_disagreement = 0.0;
        double expert_disagreement = 0.0;
        double structural_risk = 0.0;
        double fallback_frequency = 0.0;

        double open_size_normalized = 0.0;
        double base_focal_size_normalized = 0.0;


        double previous_focal_size_normalized = 0.0;

        double f_min_normalized = 0.0;
        double current_g_normalized = 0.0;
        double current_h_normalized = 0.0;
        double branching_normalized = 0.0;
        double search_progress = 0.0;

        std::vector<double> to_vector() const {
            return {
                    confidence,
                    intra_uncertainty,
                    inter_disagreement,
                    expert_disagreement,
                    structural_risk,
                    fallback_frequency,
                    open_size_normalized,
                    base_focal_size_normalized,
                    previous_focal_size_normalized,
                    f_min_normalized,
                    current_g_normalized,
                    current_h_normalized,
                    branching_normalized,
                    search_progress
            };
        }
    };

    class FocalWidthController {
    public:
        virtual ~FocalWidthController() = default;

        virtual double raw_width(
                const SearchState& state,
                double maximum_width) const = 0;
    };

    class MultiplicativeController final
            : public FocalWidthController {
    public:
        double raw_width(
                const SearchState& state,
                double maximum_width) const override {

            return 1.0 +
                   (maximum_width - 1.0) *
                   state.confidence *
                   (1.0 - state.structural_risk) *
                   (1.0 - state.fallback_frequency);
        }
    };

    class LinearController final : public FocalWidthController {
    public:
        LinearController(double a, double b, double c)
                : a_(a), b_(b), c_(c) {}

        double raw_width(
                const SearchState& state,
                double maximum_width) const override {

            const double score =
                    a_ * state.confidence +
                    b_ * (1.0 - state.structural_risk) +
                    c_ * (1.0 - state.fallback_frequency);

            return 1.0 +
                   (maximum_width - 1.0) * score;
        }

    private:
        double a_;
        double b_;
        double c_;
    };

    class ThresholdController final
            : public FocalWidthController {
    public:
        ThresholdController(
                double confidence_threshold,
                double disagreement_threshold,
                double conservative_width)
                : confidence_threshold_(confidence_threshold),
                  disagreement_threshold_(disagreement_threshold),
                  conservative_width_(conservative_width) {}

        double raw_width(
                const SearchState& state,
                double maximum_width) const override {

            if (state.confidence < confidence_threshold_ ||
                state.inter_disagreement >
                disagreement_threshold_) {
                return conservative_width_;
            }

            return maximum_width;
        }

    private:
        double confidence_threshold_;
        double disagreement_threshold_;
        double conservative_width_;
    };

    class SafetyProjection {
    public:
        static double project(
                double raw_width,
                double maximum_width) {

            if (!std::isfinite(raw_width)) {
                return 1.0;
            }

            maximum_width = std::max(1.0, maximum_width);

            return std::clamp(
                    raw_width,
                    1.0,
                    maximum_width);
        }
    };

} // namespace cadfs


