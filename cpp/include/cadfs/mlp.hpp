#pragma once
#include <string>
#include <vector>
#include "grid_map.hpp"
#include "guidance.hpp"

namespace cadfs {

// CPU inference for the trained CostToGoNet ensemble (exported by
// python/ml/export_weights.py). Mirrors exactly:
//   Conv3x3(pad1)->ReLU->MaxPool2 -> Conv3x3(pad1)->ReLU->MaxPool2
//   -> flatten + extra(10) -> FC->ReLU -> FC -> scalar
// Version 2 applies Softplus and predicts log1p(d*/diagonal), then converts
// it to the bounded monotone priority 1-exp(-z). Version 1 remains readable
// for backward compatibility.
struct ConvLayer { int cin, cout; std::vector<float> w, b; };
struct FcLayer   { int in, out;  std::vector<float> w, b; };

struct MemberNet {
    ConvLayer c1, c2;
    FcLayer f1, f2;
    float forward(const std::vector<float>& patch,      // PATCH*PATCH, ch=1
                  const float* extra, int n_extra) const;
};

class EnsembleGuidance final : public GuidanceModel {
public:
    // path: ensemble.txt. Node features are built from the map on the fly.
    explicit EnsembleGuidance(const std::string& path);
    void eval(const GridMap& m, int node, int goal,
              double& H_L, double& variance) const override;
    int members() const { return (int)nets_.size(); }
    int format_version() const { return format_version_; }
    int patch_size() const { return patch_; }
    double variance_scale() const { return variance_scale_; }
    double variance_floor() const { return variance_floor_; }
    // raw ensemble outputs for parity tests
    void raw_eval(const GridMap& m, int node, int goal,
                  std::vector<float>& outs) const;
private:
    std::vector<MemberNet> nets_;
    int patch_ = 15, extra_ = 4;
    int format_version_ = 1;
    bool log1p_target_ = false;
    double variance_scale_ = 1.0;
    double variance_floor_ = 0.0;
};

} // namespace cadfs
