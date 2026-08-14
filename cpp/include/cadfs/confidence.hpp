#pragma once

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
                double temperature);

        double estimate(const ConfidenceSignals &s) const override;

    private:
        double intra_weight_;
        double inter_weight_;
        double risk_weight_;
        double reference_weight_;
        double ood_weight_;
        double temperature_;
    };
}// namespace cadfs

