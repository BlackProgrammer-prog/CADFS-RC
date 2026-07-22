#include "cadfs/grid_map.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cadfs {

GridMap::GridMap(int width, int height, std::vector<uint8_t> occ)
    : w_(width), h_(height), occ_(std::move(occ)) {
    if ((int)occ_.size() != w_ * h_) throw std::invalid_argument("occ size mismatch");
}

static uint8_t cell_from_char(char c) {
    switch (c) {
        case '.': case 'G': case 'S': return 0;
        case '@': case 'T': case 'O': case 'W': return 1;
        default: return 1; // unknown -> obstacle (pessimistic)
    }
}

GridMap GridMap::from_ascii(const std::vector<std::string>& rows) {
    if (rows.empty()) throw std::invalid_argument("empty map");
    const int h = (int)rows.size(), w = (int)rows[0].size();
    std::vector<uint8_t> occ((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        if ((int)rows[y].size() != w) throw std::invalid_argument("ragged map rows");
        for (int x = 0; x < w; ++x) occ[(size_t)y * w + x] = cell_from_char(rows[y][x]);
    }
    return GridMap(w, h, std::move(occ));
}

GridMap GridMap::load_movingai(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open map file: " + path);
    std::string line, key;
    int w = 0, h = 0;
    // header: "type octile" / "height H" / "width W" / "map"
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        ss >> key;
        if (key == "height") ss >> h;
        else if (key == "width") ss >> w;
        else if (key == "map") break;
    }
    if (w <= 0 || h <= 0) throw std::runtime_error("bad map header: " + path);
    std::vector<std::string> rows;
    rows.reserve(h);
    while ((int)rows.size() < h && std::getline(f, line)) {
        if ((int)line.size() > w) line.resize(w);
        while ((int)line.size() < w) line.push_back('@');
        rows.push_back(line);
    }
    if ((int)rows.size() != h) throw std::runtime_error("truncated map: " + path);
    return from_ascii(rows);
}

int GridMap::neighbors(int id, int connectivity, int* out_ids, double* out_costs) const {
    static const int DX4[4] = {1, -1, 0, 0}, DY4[4] = {0, 0, 1, -1};
    static const double SQRT2 = 1.4142135623730951;
    const int x = x_of(id), y = y_of(id);
    int n = 0;
    for (int k = 0; k < 4; ++k) {
        const int nx = x + DX4[k], ny = y + DY4[k];
        if (passable(nx, ny)) { out_ids[n] = idx(nx, ny); out_costs[n] = 1.0; ++n; }
    }
    if (connectivity == 8) {
        static const int DXD[4] = {1, 1, -1, -1}, DYD[4] = {1, -1, 1, -1};
        for (int k = 0; k < 4; ++k) {
            const int nx = x + DXD[k], ny = y + DYD[k];
            // no corner cutting: both cardinals must be free
            if (passable(nx, ny) && passable(x + DXD[k], y) && passable(x, y + DYD[k])) {
                out_ids[n] = idx(nx, ny); out_costs[n] = SQRT2; ++n;
            }
        }
    }
    return n;
}

int GridMap::degree(int id, int connectivity) const {
    int ids[8]; double cs[8];
    return neighbors(id, connectivity, ids, cs);
}

double GridMap::obstacle_density(int id, int radius) const {
    const int cx = x_of(id), cy = y_of(id);
    int obs = 0, total = 0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            ++total;
            const int x = cx + dx, y = cy + dy;
            if (!in_bounds(x, y) || blocked(x, y)) ++obs;
        }
    return (double)obs / (double)total;
}

} // namespace cadfs
