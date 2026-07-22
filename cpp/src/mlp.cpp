#include "cadfs/mlp.hpp"
#include "cadfs/heuristics.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cadfs {

namespace {
inline float relu(float x) { return x > 0.f ? x : 0.f; }

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

EnsembleGuidance::EnsembleGuidance(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open ensemble file: " + path);
    std::string tok; int ver, K, hidden;
    f >> tok >> ver;
    if (tok != "CADFS_ENSEMBLE") throw std::runtime_error("bad magic");
    f >> tok >> K >> tok >> patch_ >> tok >> extra_ >> tok >> hidden;
    nets_.resize(K);
    auto read_floats = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        for (size_t i = 0; i < n; ++i) f >> v[i];
    };
    for (int k = 0; k < K; ++k) {
        int idx; f >> tok >> idx;                       // MEMBER i
        MemberNet& net = nets_[k];
        for (ConvLayer* c : {&net.c1, &net.c2}) {
            f >> tok >> c->cin >> c->cout;              // CONV cin cout
            read_floats(c->w, (size_t)c->cout * c->cin * 9);
            read_floats(c->b, c->cout);
        }
        for (FcLayer* fc : {&net.f1, &net.f2}) {
            f >> tok >> fc->in >> fc->out;              // FC in out
            read_floats(fc->w, (size_t)fc->out * fc->in);
            read_floats(fc->b, fc->out);
        }
    }
    if (!f) throw std::runtime_error("truncated ensemble file");
}

void EnsembleGuidance::raw_eval(const GridMap& m, int node, int goal,
                                std::vector<float>& outs) const {
    const int R = patch_ / 2;
    const int x = m.x_of(node), y = m.y_of(node);
    std::vector<float> patch((size_t)patch_ * patch_, 1.f);   // OOB = obstacle
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const int xx = x + dx, yy = y + dy;
            if (m.in_bounds(xx, yy))
                patch[(size_t)(dy + R) * patch_ + (dx + R)] =
                    m.blocked(xx, yy) ? 1.f : 0.f;
        }
    const double diag = std::hypot((double)m.width(), (double)m.height());
    const double dx = m.x_of(goal) - x, dy = m.y_of(goal) - y;
    const double adx = std::abs(dx), ady = std::abs(dy);
    static const double SQRT2 = 1.4142135623730951;
    const float extra[4] = {
        (float)(dx / diag), (float)(dy / diag),
        (float)(std::hypot(dx, dy) / diag),
        (float)(((adx + ady) + (SQRT2 - 2.0) * std::min(adx, ady)) / diag)};
    outs.resize(nets_.size());
    for (size_t k = 0; k < nets_.size(); ++k)
        outs[k] = nets_[k].forward(patch, extra, extra_);
}

void EnsembleGuidance::eval(const GridMap& m, int node, int goal,
                            double& H_L, double& variance) const {
    std::vector<float> outs;
    raw_eval(m, node, goal, outs);
    double mu = 0;
    for (float v : outs) mu += v;
    mu /= outs.size();
    double var = 0;
    for (float v : outs) var += (v - mu) * (v - mu);
    var /= outs.size();
    H_L = std::min(1.0, std::max(0.0, mu));   // normalized priority in [0,1]
    variance = var;
}

} // namespace cadfs
