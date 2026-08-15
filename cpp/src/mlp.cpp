#include "cadfs/mlp.hpp"
#include "cadfs/heuristics.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cadfs {

namespace {
inline float relu(float x) { return x > 0.f ? x : 0.f; }

inline float softplus(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline double log1p_priority(double encoded) {
    return -std::expm1(-std::max(0.0, encoded));
}

double line_obstacle_fraction(const GridMap& map,
                              int x0, int y0, int x1, int y1) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    int blocked = 0;
    int total = 0;
    while (true) {
        ++total;
        if (map.blocked(x0, y0)) ++blocked;
        if (x0 == x1 && y0 == y1) break;
        const int twice = 2 * error;
        if (twice >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice <= dx) {
            error += dx;
            y0 += sy;
        }
    }
    return static_cast<double>(blocked) / std::max(1, total);
}

// Conv3x3 pad1 + ReLU + MaxPool2 fused pipeline on a (C,H,W) buffer.
void conv_relu(const ConvLayer& L, const std::vector<float>& in, int H, int W,
               std::vector<float>& out) {
    out.assign((size_t)L.cout * H * W, 0.f);
    for (int co = 0; co < L.cout; ++co) {
        const float* wk = &L.w[(size_t)co * L.cin * 9];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float acc = L.b[co];
                for (int ci = 0; ci < L.cin; ++ci) {
                    const float* w9 = wk + ci * 9;
                    const float* ip = &in[(size_t)ci * H * W];
                    for (int ky = -1; ky <= 1; ++ky) {
                        const int yy = y + ky;
                        if (yy < 0 || yy >= H) continue;
                        for (int kx = -1; kx <= 1; ++kx) {
                            const int xx = x + kx;
                            if (xx < 0 || xx >= W) continue;
                            acc += w9[(ky + 1) * 3 + (kx + 1)] * ip[yy * W + xx];
                        }
                    }
                }
                out[((size_t)co * H + y) * W + x] = relu(acc);
            }
    }
}

void maxpool2(const std::vector<float>& in, int C, int H, int W,
              std::vector<float>& out, int& oH, int& oW) {
    oH = H / 2; oW = W / 2;                     // floor, matches PyTorch
    out.assign((size_t)C * oH * oW, 0.f);
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < oH; ++y)
            for (int x = 0; x < oW; ++x) {
                const float* ip = &in[(size_t)c * H * W];
                float v = ip[(2 * y) * W + 2 * x];
                v = std::max(v, ip[(2 * y) * W + 2 * x + 1]);
                v = std::max(v, ip[(2 * y + 1) * W + 2 * x]);
                v = std::max(v, ip[(2 * y + 1) * W + 2 * x + 1]);
                out[((size_t)c * oH + y) * oW + x] = v;
            }
}
} // namespace

float MemberNet::forward(const std::vector<float>& patch,
                         const float* extra, int n_extra) const {
    const int P = (int)std::lround(std::sqrt((double)patch.size()));
    std::vector<float> a, b;
    int H = P, W = P, oH, oW;
    conv_relu(c1, patch, H, W, a);
    maxpool2(a, c1.cout, H, W, b, oH, oW); H = oH; W = oW;
    conv_relu(c2, b, H, W, a);
    maxpool2(a, c2.cout, H, W, b, oH, oW); H = oH; W = oW;
    // flatten + extras -> fc1(ReLU) -> fc2
    std::vector<float> z((size_t)b.size() + n_extra);
    std::copy(b.begin(), b.end(), z.begin());
    for (int i = 0; i < n_extra; ++i) z[b.size() + i] = extra[i];
    std::vector<float> h1(f1.out);
    for (int o = 0; o < f1.out; ++o) {
        float acc = f1.b[o];
        const float* wr = &f1.w[(size_t)o * f1.in];
        for (int i = 0; i < f1.in; ++i) acc += wr[i] * z[i];
        h1[o] = relu(acc);
    }
    float y = f2.b[0];
    for (int i = 0; i < f2.in; ++i) y += f2.w[i] * h1[i];
    return y;
}

EnsembleGuidance::EnsembleGuidance(const std::string& path,
                                   int early_exit_members,
                                   double early_exit_variance)
        : early_exit_members_(early_exit_members),
          early_exit_variance_(early_exit_variance) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open ensemble file: " + path);
    std::string tok; int ver, K, hidden;
    f >> tok >> ver;
    if (tok != "CADFS_ENSEMBLE") throw std::runtime_error("bad magic");
    if (ver != 1 && ver != 2)
        throw std::runtime_error("unsupported ensemble format version");
    format_version_ = ver;
    f >> tok >> K >> tok >> patch_ >> tok >> extra_ >> tok >> hidden;
    if (K <= 0 || patch_ <= 0 || patch_ % 2 == 0 || extra_ <= 0 || hidden <= 0)
        throw std::runtime_error("invalid ensemble dimensions");
    if (format_version_ >= 2) {
        std::string target;
        f >> tok >> target;
        if (tok != "TARGET" || target != "LOG1P")
            throw std::runtime_error("unsupported ensemble target transform");
        log1p_target_ = true;
        f >> tok >> variance_scale_;
        if (tok != "VARIANCE_SCALE" || !std::isfinite(variance_scale_) ||
            variance_scale_ < 0.0)
            throw std::runtime_error("invalid ensemble variance scale");
        f >> tok >> variance_floor_;
        if (tok != "VARIANCE_FLOOR" || !std::isfinite(variance_floor_) ||
            variance_floor_ < 0.0)
            throw std::runtime_error("invalid ensemble variance floor");
    }
    nets_.resize(K);
    auto read_floats = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        for (size_t i = 0; i < n; ++i) f >> v[i];
    };
    for (int k = 0; k < K; ++k) {
        int idx; f >> tok >> idx;                       // MEMBER i
        if (tok != "MEMBER" || idx != k)
            throw std::runtime_error("invalid ensemble member header");
        MemberNet& net = nets_[k];
        for (ConvLayer* c : {&net.c1, &net.c2}) {
            f >> tok >> c->cin >> c->cout;              // CONV cin cout
            if (tok != "CONV" || c->cin <= 0 || c->cout <= 0)
                throw std::runtime_error("invalid ensemble convolution layer");
            read_floats(c->w, (size_t)c->cout * c->cin * 9);
            read_floats(c->b, c->cout);
        }
        for (FcLayer* fc : {&net.f1, &net.f2}) {
            f >> tok >> fc->in >> fc->out;              // FC in out
            if (tok != "FC" || fc->in <= 0 || fc->out <= 0)
                throw std::runtime_error("invalid ensemble fully-connected layer");
            read_floats(fc->w, (size_t)fc->out * fc->in);
            read_floats(fc->b, fc->out);
        }
        const int pooled = patch_ / 2 / 2;
        const int expected_fc1 = net.c2.cout * pooled * pooled + extra_;
        if (net.c1.cin != 1 || net.c2.cin != net.c1.cout ||
            net.f1.in != expected_fc1 || net.f1.out != hidden ||
            net.f2.in != hidden || net.f2.out != 1)
            throw std::runtime_error("ensemble architecture does not match header");
    }
    if (!f) throw std::runtime_error("truncated ensemble file");
    if (early_exit_members_ == 1 || early_exit_members_ < 0)
        throw std::invalid_argument("early_exit_members must be 0 or at least 2");
    if (!std::isfinite(early_exit_variance_) ||
        early_exit_variance_ < 0.0)
        throw std::invalid_argument(
            "early_exit_variance must be finite and non-negative");
    if (early_exit_members_ >= static_cast<int>(nets_.size()))
        early_exit_members_ = 0;
}

void build_guidance_inputs(const GridMap& m, int node, int goal,
                           int patch_size, int extra_count,
                           std::vector<float>& patch,
                           std::vector<float>& extra) {
    const int R = patch_size / 2;
    const int x = m.x_of(node), y = m.y_of(node);
    patch.assign((size_t)patch_size * patch_size, 1.f);
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const int xx = x + dx, yy = y + dy;
            if (m.in_bounds(xx, yy))
                patch[(size_t)(dy + R) * patch_size + (dx + R)] =
                    m.blocked(xx, yy) ? 1.f : 0.f;
        }
    const double diag = std::hypot((double)m.width(), (double)m.height());
    const double dx = m.x_of(goal) - x, dy = m.y_of(goal) - y;
    const double adx = std::abs(dx), ady = std::abs(dy);
    static const double SQRT2 = 1.4142135623730951;
    extra.clear();
    extra.reserve(static_cast<std::size_t>(extra_count));
    extra.push_back((float)(dx / diag));
    extra.push_back((float)(dy / diag));
    extra.push_back((float)(std::hypot(dx, dy) / diag));
    extra.push_back((float)(((adx + ady) +
        (SQRT2 - 2.0) * std::min(adx, ady)) / diag));
    if (extra_count == 10) {
        extra.push_back((float)line_obstacle_fraction(
            m, x, y, m.x_of(goal), m.y_of(goal)));
        extra.push_back((float)m.obstacle_density(node, 3));
        extra.push_back((float)m.obstacle_density(node, 7));
        extra.push_back((float)m.obstacle_density(node, 15));
        extra.push_back((float)m.degree(node, 8) / 8.0f);
        extra.push_back((float)m.obstacle_density(goal, 3));
    } else if (extra_count != 4) {
        throw std::runtime_error("unsupported ensemble feature count");
    }
}

void EnsembleGuidance::build_inputs(const GridMap& m, int node, int goal,
                                    std::vector<float>& patch,
                                    std::vector<float>& extra) const {
    build_guidance_inputs(
            m, node, goal, patch_, extra_, patch, extra);
}

void EnsembleGuidance::raw_eval(const GridMap& m, int node, int goal,
                                std::vector<float>& outs) const {
    std::vector<float> patch, extra;
    build_inputs(m, node, goal, patch, extra);
    outs.resize(nets_.size());
    for (size_t k = 0; k < nets_.size(); ++k) {
        const float linear = nets_[k].forward(patch, extra.data(), extra_);
        outs[k] = log1p_target_ ? softplus(linear) : linear;
    }
}

void EnsembleGuidance::eval(const GridMap& m, int node, int goal,
                            double& H_L, double& variance) const {
    const GuidanceEvaluation result = eval_detailed(m, node, goal);
    H_L = result.priority;
    variance = result.variance;
}

GuidanceEvaluation EnsembleGuidance::eval_detailed(
        const GridMap& m, int node, int goal) const {
    std::vector<float> patch, extra;
    build_inputs(m, node, goal, patch, extra);

    const std::size_t initial = early_exit_members_ > 0
            ? static_cast<std::size_t>(early_exit_members_)
            : nets_.size();
    std::vector<float> outs;
    outs.reserve(nets_.size());
    for (std::size_t k = 0; k < initial; ++k) {
        const float linear = nets_[k].forward(patch, extra.data(), extra_);
        outs.push_back(log1p_target_ ? softplus(linear) : linear);
    }

    auto summarize = [&]() {
        GuidanceEvaluation summary;
        double mean = 0.0;
        for (float encoded : outs)
            mean += log1p_target_
                    ? log1p_priority(encoded) : static_cast<double>(encoded);
        mean /= static_cast<double>(outs.size());
        double var = 0.0;
        for (float encoded : outs) {
            const double value = log1p_target_
                    ? log1p_priority(encoded) : static_cast<double>(encoded);
            var += (value - mean) * (value - mean);
        }
        var /= static_cast<double>(outs.size());
        summary.priority = std::clamp(mean, 0.0, 1.0);
        summary.variance = std::max(
                0.0, var * variance_scale_ + variance_floor_);
        summary.member_evaluations = static_cast<int>(outs.size());
        return summary;
    };

    GuidanceEvaluation result = summarize();
    if (initial < nets_.size() &&
        result.variance > early_exit_variance_) {
        for (std::size_t k = initial; k < nets_.size(); ++k) {
            const float linear = nets_[k].forward(
                    patch, extra.data(), extra_);
            outs.push_back(log1p_target_ ? softplus(linear) : linear);
        }
        result = summarize();
    }
    return result;
}

} // namespace cadfs
