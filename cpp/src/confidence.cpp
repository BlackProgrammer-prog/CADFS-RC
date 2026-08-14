#include <cadfs/confidence.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cadfs {

namespace {

double unit_signal(double value) {
    if (!std::isfinite(value)) return 1.0;
    return std::clamp(value, 0.0, 1.0);
}

void validate_weight(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(__func__);
    }
}

} // namespace

CompositeConfidence::CompositeConfidence(
        double intra_weight,
        double inter_weight,
        double risk_weight,
        double reference_weight,
        double ood_weight,
        double temperature)
        : intra_weight_(intra_weight),
          inter_weight_(inter_weight),
          risk_weight_(risk_weight),
          reference_weight_(reference_weight),
          ood_weight_(ood_weight),
          temperature_(temperature) {
    validate_weight(intra_weight_);
    validate_weight(inter_weight_);
    validate_weight(risk_weight_);
    validate_weight(reference_weight_);
    validate_weight(ood_weight_);
    if (!std::isfinite(temperature_) || temperature_ <= 0.0) {
        throw std::invalid_argument(__func__);
    }
}

double CompositeConfidence::estimate(const ConfidenceSignals& signals) const {
    const double penalty =
            intra_weight_ * unit_signal(signals.intra_uncertainty) +
            inter_weight_ * unit_signal(signals.inter_disagreement) +
            risk_weight_ * unit_signal(signals.structural_risk) +
            reference_weight_ * unit_signal(signals.model_reference_disagreement) +
            ood_weight_ * unit_signal(signals.ood_score);

    const double result = std::exp(-penalty / temperature_);
    if (!std::isfinite(result)) return 0.0;
    return std::clamp(result, 0.0, 1.0);
}

} // namespace cadfs
