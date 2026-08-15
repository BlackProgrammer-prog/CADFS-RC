#include "cadfs/fast_mlp.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace cadfs {
namespace {

float relu_fast(float value) {
    return value > 0.0f ? value : 0.0f;
}

float softplus_fast(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return std::exp(value);
    return std::log1p(std::exp(value));
}

double priority_from_log1p(double encoded) {
    return -std::expm1(-std::max(0.0, encoded));
}

void read_fc(std::ifstream& stream, FcLayer& layer) {
    std::string token;
    stream >> token >> layer.in >> layer.out;
    if (token != "FC" || layer.in <= 0 || layer.out <= 0)
        throw std::runtime_error("invalid fast-ensemble FC layer");
    layer.w.resize(static_cast<std::size_t>(layer.in) * layer.out);
    layer.b.resize(static_cast<std::size_t>(layer.out));
    for (float& value : layer.w) stream >> value;
    for (float& value : layer.b) stream >> value;
}

std::vector<float> linear_relu(const FcLayer& layer,
                               const std::vector<float>& input) {
    if (static_cast<int>(input.size()) != layer.in)
        throw std::runtime_error("fast-ensemble input size mismatch");
    std::vector<float> output(static_cast<std::size_t>(layer.out));
    for (int out = 0; out < layer.out; ++out) {
        float value = layer.b[static_cast<std::size_t>(out)];
        const float* weights =
                &layer.w[static_cast<std::size_t>(out) * layer.in];
        for (int in = 0; in < layer.in; ++in)
            value += weights[in] * input[static_cast<std::size_t>(in)];
        output[static_cast<std::size_t>(out)] = relu_fast(value);
    }
    return output;
}

} // namespace

FastEnsembleGuidance::FastEnsembleGuidance(const std::string& path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open fast ensemble file: " + path);

    std::string token;
    int version = 0;
    int hidden1 = 0;
    int hidden2 = 0;
    int head_count = 0;
    std::string target;
    stream >> token >> version;
    if (token != "CADFS_FAST_ENSEMBLE" || version != 1)
        throw std::runtime_error("unsupported fast ensemble format");
    auto expect = [&](const char* expected) {
        stream >> token;
        if (token != expected)
            throw std::runtime_error(
                std::string("expected fast ensemble token ") + expected);
    };
    expect("PATCH"); stream >> patch_;
    expect("EXTRA"); stream >> extra_;
    expect("HIDDEN1"); stream >> hidden1;
    expect("HIDDEN2"); stream >> hidden2;
    expect("HEADS"); stream >> head_count;
    expect("TARGET"); stream >> target;
    expect("VARIANCE_SCALE"); stream >> variance_scale_;
    expect("VARIANCE_FLOOR"); stream >> variance_floor_;
    if (patch_ <= 0 || patch_ % 2 == 0 || extra_ <= 0 ||
        hidden1 <= 0 || hidden2 <= 0 || head_count < 2 ||
        target != "LOG1P" ||
        !std::isfinite(variance_scale_) || variance_scale_ < 0.0 ||
        !std::isfinite(variance_floor_) || variance_floor_ < 0.0)
        throw std::runtime_error("invalid fast ensemble header");

    read_fc(stream, fc1_);
    read_fc(stream, fc2_);
    heads_.resize(static_cast<std::size_t>(head_count));
    for (int index = 0; index < head_count; ++index) {
        int stored_index = -1;
        stream >> token >> stored_index;
        if (token != "HEAD" || stored_index != index)
            throw std::runtime_error("invalid fast ensemble head");
        read_fc(stream, heads_[static_cast<std::size_t>(index)]);
    }
    const int expected_input = patch_ * patch_ + extra_;
    if (fc1_.in != expected_input || fc1_.out != hidden1 ||
        fc2_.in != hidden1 || fc2_.out != hidden2)
        throw std::runtime_error("fast ensemble architecture mismatch");
    for (const FcLayer& head : heads_) {
        if (head.in != hidden2 || head.out != 1)
            throw std::runtime_error("fast ensemble head mismatch");
    }
    if (!stream)
        throw std::runtime_error("truncated fast ensemble file");
}

std::vector<float> FastEnsembleGuidance::encode(
        const std::vector<float>& patch,
        const std::vector<float>& extra) const {
    std::vector<float> input;
    input.reserve(patch.size() + extra.size());
    input.insert(input.end(), patch.begin(), patch.end());
    input.insert(input.end(), extra.begin(), extra.end());
    return linear_relu(fc2_, linear_relu(fc1_, input));
}

void FastEnsembleGuidance::raw_eval(
        const GridMap& map, int node, int goal,
        std::vector<float>& encoded_outputs) const {
    std::vector<float> patch, extra;
    build_guidance_inputs(
            map, node, goal, patch_, extra_, patch, extra);
    const std::vector<float> hidden = encode(patch, extra);
    encoded_outputs.resize(heads_.size());
    for (std::size_t index = 0; index < heads_.size(); ++index) {
        const FcLayer& head = heads_[index];
        float value = head.b[0];
        for (int in = 0; in < head.in; ++in)
            value += head.w[static_cast<std::size_t>(in)] *
                     hidden[static_cast<std::size_t>(in)];
        encoded_outputs[index] = softplus_fast(value);
    }
}

GuidanceEvaluation FastEnsembleGuidance::eval_detailed(
        const GridMap& map, int node, int goal) const {
    std::vector<float> outputs;
    raw_eval(map, node, goal, outputs);
    double mean = 0.0;
    for (float output : outputs)
        mean += priority_from_log1p(output);
    mean /= static_cast<double>(outputs.size());
    double variance = 0.0;
    for (float output : outputs) {
        const double priority = priority_from_log1p(output);
        variance += (priority - mean) * (priority - mean);
    }
    variance /= static_cast<double>(outputs.size());
    return {
        std::clamp(mean, 0.0, 1.0),
        std::max(0.0, variance * variance_scale_ + variance_floor_),
        static_cast<int>(outputs.size())
    };
}

void FastEnsembleGuidance::eval(
        const GridMap& map, int node, int goal,
        double& priority, double& variance) const {
    const GuidanceEvaluation result = eval_detailed(map, node, goal);
    priority = result.priority;
    variance = result.variance;
}

} // namespace cadfs
