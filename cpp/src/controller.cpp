#include <cadfs/controller.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace cadfs {

namespace {

double unit(double value) {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

} // namespace

std::vector<double> SearchState::to_vector() const {
    return {
        confidence, intra_uncertainty, inter_disagreement,
        expert_disagreement, structural_risk, fallback_frequency,
        open_size_normalized, base_focal_size_normalized,
        previous_focal_size_normalized, f_min_normalized,
        current_g_normalized, current_h_normalized,
        branching_normalized, search_progress
    };
}

double MultiplicativeController::raw_width(
        const SearchState& state, double maximum_width) const {
    maximum_width = std::max(1.0, maximum_width);
    return 1.0 + (maximum_width - 1.0) *
           unit(state.confidence) *
           (1.0 - unit(state.structural_risk)) *
           (1.0 - unit(state.fallback_frequency));
}

LinearController::LinearController(double a, double b, double c)
        : a_(a), b_(b), c_(c) {
    if (!std::isfinite(a_) || !std::isfinite(b_) || !std::isfinite(c_) ||
        a_ < 0.0 || b_ < 0.0 || c_ < 0.0) {
        throw std::invalid_argument(__func__);
    }
    const double sum = a_ + b_ + c_;
    if (sum <= 0.0) {
        throw std::invalid_argument(__func__);
    }
    a_ /= sum;
    b_ /= sum;
    c_ /= sum;
}

double LinearController::raw_width(
        const SearchState& state, double maximum_width) const {
    maximum_width = std::max(1.0, maximum_width);
    const double score =
            a_ * unit(state.confidence) +
            b_ * (1.0 - unit(state.structural_risk)) +
            c_ * (1.0 - unit(state.fallback_frequency));
    return 1.0 + (maximum_width - 1.0) * score;
}

FixedController::FixedController(double width)
        : width_(width) {
    if (!std::isfinite(width_)) {
        throw std::invalid_argument(__func__);
    }
}

double FixedController::raw_width(
        const SearchState&, double) const {
    return width_;
}

ThresholdController::ThresholdController(
        double confidence_threshold,
        double disagreement_threshold,
        double conservative_width)
        : confidence_threshold_(confidence_threshold),
          disagreement_threshold_(disagreement_threshold),
          conservative_width_(conservative_width) {
    if (!std::isfinite(confidence_threshold_) ||
        !std::isfinite(disagreement_threshold_) ||
        !std::isfinite(conservative_width_)) {
        throw std::invalid_argument(__func__);
    }
    confidence_threshold_ = unit(confidence_threshold_);
    disagreement_threshold_ = unit(disagreement_threshold_);
}

double ThresholdController::raw_width(
        const SearchState& state, double maximum_width) const {
    if (unit(state.confidence) < confidence_threshold_ ||
        unit(state.inter_disagreement) > disagreement_threshold_) {
        return conservative_width_;
    }
    return maximum_width;
}

MLPController::MLPController(
        std::vector<double> input_hidden_weights,
        std::vector<double> hidden_bias,
        std::vector<double> hidden_output_weights,
        std::vector<double> output_bias,
        std::size_t input_size,
        std::size_t hidden_size,
        std::vector<double> actions,
        MLPControllerMode mode)
        : w1_(std::move(input_hidden_weights)),
          b1_(std::move(hidden_bias)),
          w2_(std::move(hidden_output_weights)),
          b2_(std::move(output_bias)),
          input_size_(input_size),
          hidden_size_(hidden_size),
          actions_(std::move(actions)),
          mode_(mode) {
    if (input_size_ == 0 || hidden_size_ == 0) {
        throw std::invalid_argument(__func__);
    }
    if (w1_.size() != input_size_ * hidden_size_ ||
        b1_.size() != hidden_size_) {
        throw std::invalid_argument(__func__);
    }

    const std::size_t output_size =
            mode_ == MLPControllerMode::Regression
            ? 1 : actions_.size();

    if (output_size == 0 ||
        w2_.size() != output_size * hidden_size_ ||
        b2_.size() != output_size) {
        throw std::invalid_argument(__func__);
    }

    const auto all_finite = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    if (!all_finite(w1_) || !all_finite(b1_) ||
        !all_finite(w2_) || !all_finite(b2_)) {
        throw std::invalid_argument(__func__);
    }

    for (double action : actions_) {
        if (!std::isfinite(action)) {
            throw std::invalid_argument(__func__);
        }
    }
}

double MLPController::raw_width(
        const SearchState& state, double) const {
    const std::vector<double> input = state.to_vector();
    if (input.size() != input_size_) {
        throw std::runtime_error(__func__);
    }

    std::vector<double> hidden(hidden_size_, 0.0);
    for (std::size_t row = 0; row < hidden_size_; ++row) {
        double value = b1_[row];
        for (std::size_t column = 0; column < input_size_; ++column) {
            value += w1_[row * input_size_ + column] * input[column];
        }
        hidden[row] = std::max(0.0, value);
    }

    const std::size_t output_size =
            mode_ == MLPControllerMode::Regression
            ? 1 : actions_.size();
    std::vector<double> output(output_size, 0.0);

    for (std::size_t row = 0; row < output_size; ++row) {
        double value = b2_[row];
        for (std::size_t column = 0; column < hidden_size_; ++column) {
            value += w2_[row * hidden_size_ + column] * hidden[column];
        }
        output[row] = value;
    }

    if (mode_ == MLPControllerMode::Regression) {
        return output.front();
    }

    const auto best = std::max_element(output.begin(), output.end());
    const std::size_t action_index = static_cast<std::size_t>(
            std::distance(output.begin(), best));
    return actions_[action_index];
}

double SafetyProjection::project(double raw_width, double maximum_width) {
    if (!std::isfinite(raw_width)) return 1.0;
    if (!std::isfinite(maximum_width) || maximum_width < 1.0) return 1.0;
    return std::clamp(raw_width, 1.0, maximum_width);
}

} // namespace cadfs
