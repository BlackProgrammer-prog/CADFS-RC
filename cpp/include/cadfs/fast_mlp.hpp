#pragma once

#include "guidance.hpp"
#include "mlp.hpp"

#include <string>
#include <vector>

namespace cadfs {

// Shared-backbone, multi-head student for low-latency search-time guidance.
// The full occupancy patch is flattened once, two small FC layers are shared,
// and only the final scalar heads differ. Head disagreement estimates
// epistemic uncertainty without repeating the expensive feature backbone.
class FastEnsembleGuidance final : public GuidanceModel {
public:
    explicit FastEnsembleGuidance(const std::string& path);

    void eval(const GridMap& map, int node, int goal,
              double& priority, double& variance) const override;
    GuidanceEvaluation eval_detailed(const GridMap& map, int node,
                                     int goal) const override;
    void raw_eval(const GridMap& map, int node, int goal,
                  std::vector<float>& encoded_outputs) const;

    int heads() const { return static_cast<int>(heads_.size()); }
    int patch_size() const { return patch_; }
    int extra_features() const { return extra_; }
    double variance_scale() const { return variance_scale_; }
    double variance_floor() const { return variance_floor_; }

private:
    std::vector<float> encode(const std::vector<float>& patch,
                              const std::vector<float>& extra) const;

    int patch_ = 31;
    int extra_ = 10;
    FcLayer fc1_;
    FcLayer fc2_;
    std::vector<FcLayer> heads_;
    double variance_scale_ = 1.0;
    double variance_floor_ = 0.0;
};

} // namespace cadfs
