#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cadfs {

    struct IterationLog {
        std::string problem_id;
        std::string city_id;
        std::string map_family;
        std::string distribution_type;

        int64_t iteration = 0;

        std::size_t open_size = 0;
        std::size_t base_focal_size = 0;
        std::size_t focal_size = 0;

        double f_min = 0.0;

        double controller_raw_output = 0.0;
        double controller_safe_output = 0.0;

        std::vector<double> expert_predictions;
        std::vector<double> expert_weights;

        double intra_expert_uncertainty = 0.0;
        double inter_expert_disagreement = 0.0;

        double confidence = 0.0;
        double structural_risk = 0.0;
        double fallback_frequency = 0.0;

        int selected_node = -1;
        bool selected_by_anchor = false;
        bool fallback_triggered = false;

        double g = 0.0;
        double h = 0.0;
        double f = 0.0;
    };

    using IterationLogger = std::function<void(const IterationLog&)>;
}