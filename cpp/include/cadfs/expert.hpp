//
// Created by HOME on 8/14/2026.
//
#ifndef CADFS_EXPERT_H
#define CADFS_EXPERT_H

#include "grid_map.hpp"
#include "guidance.hpp"
#include "search.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cadfs {
    struct ExpertContext {
        const GridMap& map;
        int node;
        int start;
        int goal;
        double anchor_h;
        double h_normalized;
    };

    struct ExpertPrediction{
        std::string name;
        double mean = 0.0;
        double intra_uncertainty = 0.0;
        bool available = true;
        int member_evaluations = 0;
    };

    class Expert {
    public:
        virtual ~Expert() = default;
        virtual std::string name() const = 0;
        virtual ExpertPrediction predict(const ExpertContext& context) const = 0;
    };

    class GeometricExpert final : public Expert {
    public:
        explicit GeometricExpert(const GuidanceModel& guidance)
                : guidance_(guidance) {}
        std::string name() const override {
            return "geometry";
        }
        ExpertPrediction predict(const ExpertContext& context) const override;

    private:
        const GuidanceModel& guidance_;
    };

    class TopologicalExpert final : public Expert {
    public:
        explicit TopologicalExpert(int connectivity = 8) : connectivity_(connectivity) {}

        std::string name() const override {
            return "topology";
        }

        ExpertPrediction predict(const ExpertContext& context) const override;
    private:
        int connectivity_;
    };

    class GoalDistanceExpert final : public Expert {
    public:
        std::string name() const override {
            return "goal_distance";
        }

        ExpertPrediction predict (const ExpertContext& context) const override;
    };

    enum class DisagreementMetric {
        Variance,
        MeanAbsoluteDeviation,
        Range
    };

    struct FusionResult {
        double fused_prediction = 0.0;
        double intra_uncertainty = 0.0;
        double inter_disagreement = 0.0;
        std::vector<ExpertPrediction> predictions;
        std::vector<double> normalized_weights;
    };

    class ExpertFusion {
    public:
        ExpertFusion(std::vector<std::shared_ptr<const Expert>> experts ,
                     std::vector<double> weights = {},
                     DisagreementMetric metric = DisagreementMetric::Variance);

        FusionResult evaluate(const ExpertContext& context) const;

    private:
        std::vector<std::shared_ptr<const Expert>>  experts_;
        std::vector<double> weights_;
        DisagreementMetric metric_;
    };

}


#endif //CADFS_EXPERT_H
