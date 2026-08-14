#pragma once

#include <algorithm>
#include <cmath>

namespace cadfs {
    struct ConfidenceSignals {
        double intra_uncertainty = 0.0;
        double inter_disagreement = 0.0;
        double structural_risk = 0.0;
        double model_reference_disagreement = 0.0;
        double ood_score = 0.0;
    };

    class ConfidenceEstimator {
    public:
        virtual ~ConfidenceEstimator() = default;

        virtual double estimate(const ConfidenceSignals &signals) const = 0;
    };

    class CompositeConfidence final : public ConfidenceEstimator {
    public:
        CompositeConfidence(
                double intra_weight,
                double inter_weight,
                double risk_weight,
                double reference_weight,
                double ood_weight,
                double temperature
        ) : intra_weight_(intra_weight),
            inter_weight_(inter_weight),
            risk_weight_(risk_weight),
            reference_weight_(reference_weight),
            ood_weight_(ood_weight),
            temperature_(temperature) {}

        double estimate(const ConfidenceSignals &s) const override {
            const double penalty = intra_weight_ * s.intra_uncertainty +
                                   inter_weight_ * s.inter_disagreement +
                                   risk_weight_ * s.structural_risk +
                                   reference_weight_ *
                                   s.model_reference_disagreement +
                                   ood_weight_ * s.ood_score;

            const double safe_temperature = std::max(temperature_, 1e-12);
            const double confidence = std::exp(-penalty / safe_temperature);

            return std::clamp(confidence, 0.0, 1.0);
        }

    private:
        double intra_weight_;
        double inter_weight_;
        double risk_weight_;
        double reference_weight_;
        double ood_weight_;
        double temperature_;
    };
}// namespace cadfs

