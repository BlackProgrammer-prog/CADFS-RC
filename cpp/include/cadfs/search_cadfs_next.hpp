#pragma once

#include "confidence.hpp"
#include "controller.hpp"
#include "search.hpp"
#include "expert.hpp"
#include "telemetry.hpp"

namespace cadfs {
    SearchResult cadfs_next_search (
            const GridMap& map,
            const Instance& instance,
            const Config& config,
            const ExpertFusion& fusion,
            const ConfidenceEstimator& confidence,
            const FocalWidthController& controller,
            const IterationLogger* logger = nullptr
            );
}// namespace cadfs
