#pragma once

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

        std::vector<double> to_vector() const;
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
        double raw_width(const SearchState& state,
                         double maximum_width) const override;
    };

    class LinearController final : public FocalWidthController {
    public:
        LinearController(double a, double b, double c);

        double raw_width(const SearchState& state,
                         double maximum_width) const override;

    private:
        double a_;
        double b_;
        double c_;
    };

    class FixedController final : public FocalWidthController {
    public:
        explicit FixedController(double width);
        double raw_width(const SearchState& state,
                         double maximum_width) const override;
    private:
        double width_;
    };

    class ThresholdController final
            : public FocalWidthController {
    public:
        ThresholdController(
                double confidence_threshold,
                double disagreement_threshold,
                double conservative_width);

        double raw_width(const SearchState& state,
                         double maximum_width) const override;

    private:
        double confidence_threshold_;
        double disagreement_threshold_;
        double conservative_width_;
    };

    enum class MLPControllerMode {
        Regression,
        Classification
    };

    class MLPController final : public FocalWidthController {
    public:
        MLPController(
                std::vector<double> input_hidden_weights,
                std::vector<double> hidden_bias,
                std::vector<double> hidden_output_weights,
                std::vector<double> output_bias,
                std::size_t input_size,
                std::size_t hidden_size,
                std::vector<double> actions,
                MLPControllerMode mode);

        double raw_width(const SearchState& state,
                         double maximum_width) const override;

    private:
        std::vector<double> w1_;
        std::vector<double> b1_;
        std::vector<double> w2_;
        std::vector<double> b2_;
        std::size_t input_size_;
        std::size_t hidden_size_;
        std::vector<double> actions_;
        MLPControllerMode mode_;
    };

    class SafetyProjection {
    public:
        static double project(double raw_width, double maximum_width);
    };

} // namespace cadfs


