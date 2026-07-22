#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cadfs {

// Grid map: row-major, (x, y) with x = column, y = row. 0 = free, 1 = obstacle.
class GridMap {
public:
    GridMap() = default;
    GridMap(int width, int height, std::vector<uint8_t> occ);

    // Parse MovingAI-style ASCII rows: '.' 'G' 'S' free; '@' 'T' 'O' 'W' obstacles.
    static GridMap from_ascii(const std::vector<std::string>& rows);
    // Load a MovingAI .map file (header: type/height/width/map).
    static GridMap load_movingai(const std::string& path);

    int width() const  { return w_; }
    int height() const { return h_; }
    int size() const   { return w_ * h_; }

    bool in_bounds(int x, int y) const { return x >= 0 && x < w_ && y >= 0 && y < h_; }
    bool blocked(int x, int y) const   { return occ_[idx(x, y)] != 0; }
    bool passable(int x, int y) const  { return in_bounds(x, y) && !blocked(x, y); }

    int idx(int x, int y) const { return y * w_ + x; }
    int x_of(int id) const { return id % w_; }
    int y_of(int id) const { return id / w_; }

    // Neighbors under 4- or 8-connectivity. 8-connectivity forbids corner cutting:
    // a diagonal move is valid only if both adjacent cardinal cells are free.
    // Returns number of neighbors written into out_ids/out_costs (max 8).
    int neighbors(int id, int connectivity, int* out_ids, double* out_costs) const;

    // deg_free(n) for R_mob (same corner-cutting rule).
    int degree(int id, int connectivity) const;

    // Obstacle density in the (2r+1)^2 window centered at id; out-of-bounds cells
    // count as obstacles (consistent pessimistic convention for R_obs).
    double obstacle_density(int id, int radius) const;

private:
    int w_ = 0, h_ = 0;
    std::vector<uint8_t> occ_;
};

} // namespace cadfs
