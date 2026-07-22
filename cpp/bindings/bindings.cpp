#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cadfs/grid_map.hpp"
#include "cadfs/guidance.hpp"
#include "cadfs/mlp.hpp"
#include "cadfs/search_astar.hpp"
#include "cadfs/search_cadfs.hpp"
#include "cadfs/search_focal.hpp"

namespace py = pybind11;
using namespace cadfs;

static Config config_from_dict(const py::dict& d) {
    Config c;
    auto get = [&](const char* k, auto def) {
        using T = decltype(def);
        return d.contains(k) ? d[k].cast<T>() : def;
    };
    c.W = get("W", c.W); c.L = get("L", c.L); c.K = get("K", c.K);
    c.connectivity = get("connectivity", c.connectivity);
    const std::string ct = get("controller", std::string("multiplicative"));
    c.controller = ct == "linear" ? ControllerType::Linear
                 : ct == "fixed" ? ControllerType::Fixed
                 : ct == "tuned_fixed" ? ControllerType::TunedFixed
                 : ControllerType::Multiplicative;
    c.lin_a = get("lin_a", c.lin_a); c.lin_b = get("lin_b", c.lin_b); c.lin_c = get("lin_c", c.lin_c);
    c.tuned_fixed_w = get("tuned_fixed_w", c.tuned_fixed_w);
    c.risk_radius = get("risk_radius", c.risk_radius);
    c.lambda_obs = get("lambda_obs", c.lambda_obs);
    c.lambda_mob = get("lambda_mob", c.lambda_mob);
    c.lambda_dev = get("lambda_dev", c.lambda_dev);
    const std::string rm = get("risk_mode", std::string("normal"));
    c.risk_mode = rm == "off" ? Config::RiskMode::Off
                : rm == "random" ? Config::RiskMode::Random
                : rm == "permuted" ? Config::RiskMode::Permuted : Config::RiskMode::Normal;
    c.risk_seed = get("risk_seed", (uint64_t)c.risk_seed);
    c.alpha = get("alpha", c.alpha); c.beta = get("beta", c.beta);
    c.theta_c = get("theta_c", c.theta_c); c.theta_r = get("theta_r", c.theta_r);
    c.theta_dev = get("theta_dev", c.theta_dev);
    c.fallback_enabled = get("fallback_enabled", c.fallback_enabled);
    c.confidence_enabled = get("confidence_enabled", c.confidence_enabled);
    c.h_min = get("h_min", c.h_min); c.h_max = get("h_max", c.h_max); c.eps = get("eps", c.eps);
    return c;
}

static py::dict result_to_dict(const SearchResult& r) {
    py::dict d;
    d["found"] = r.found; d["cost"] = r.cost;
    d["expansions"] = r.expansions; d["generated"] = r.generated;
    d["runtime_ms"] = r.runtime_ms; d["path"] = r.path;
    d["fallback_rate"] = r.fallback_rate;
    d["mean_w"] = r.mean_w; d["min_w"] = r.min_w; d["max_w"] = r.max_w;
    d["mean_abs_dw"] = r.mean_abs_dw; d["mean_C"] = r.mean_C; d["mean_R"] = r.mean_R;
    return d;
}

PYBIND11_MODULE(cadfs_engine, mod) {
    mod.doc() = "CADFS C++ search core";

    py::class_<GridMap>(mod, "GridMap")
        .def_static("from_ascii", &GridMap::from_ascii)
        .def_static("load_movingai", &GridMap::load_movingai)
        .def_property_readonly("width", &GridMap::width)
        .def_property_readonly("height", &GridMap::height)
        .def("passable", &GridMap::passable);

    py::class_<GuidanceModel, std::shared_ptr<GuidanceModel>>(mod, "GuidanceModel");
    py::class_<MockGuidance, GuidanceModel, std::shared_ptr<MockGuidance>>(mod, "MockGuidance")
        .def(py::init<double, double, uint64_t>(),
             py::arg("h_norm"), py::arg("noise"), py::arg("seed") = 1);
    py::class_<EnsembleGuidance, GuidanceModel, std::shared_ptr<EnsembleGuidance>>(mod, "EnsembleGuidance")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def_property_readonly("members", &EnsembleGuidance::members)
        .def("raw_eval", [](const EnsembleGuidance& e, const GridMap& m,
                            int x, int y, int gx, int gy) {
            std::vector<float> outs;
            e.raw_eval(m, m.idx(x, y), m.idx(gx, gy), outs);
            return outs;
        });

    mod.def("dijkstra_all", [](const GridMap& m, int gx, int gy, int conn) {
        return dijkstra_all(m, m.idx(gx, gy), conn);
    }, "optimal cost-to-go labels d*(cell, goal)");

    mod.def("run_astar", [](const GridMap& m, py::tuple s, py::tuple g,
                            const py::dict& cfg, double weight) {
        Instance ins{s[0].cast<int>(), s[1].cast<int>(), g[0].cast<int>(), g[1].cast<int>()};
        return result_to_dict(astar(m, ins, config_from_dict(cfg), weight));
    }, py::arg("map"), py::arg("start"), py::arg("goal"), py::arg("config"), py::arg("weight") = 1.0);

    mod.def("run_focal", [](const GridMap& m, py::tuple s, py::tuple g,
                            const py::dict& cfg, std::shared_ptr<GuidanceModel> model,
                            double width) {
        Instance ins{s[0].cast<int>(), s[1].cast<int>(), g[0].cast<int>(), g[1].cast<int>()};
        return result_to_dict(focal_fixed(m, ins, config_from_dict(cfg), model.get(), width));
    }, py::arg("map"), py::arg("start"), py::arg("goal"), py::arg("config"),
       py::arg("model") = nullptr, py::arg("width") = 2.0);

    mod.def("run_cadfs", [](const GridMap& m, py::tuple s, py::tuple g,
                            const py::dict& cfg, std::shared_ptr<GuidanceModel> model,
                            double tau_c) {
        Instance ins{s[0].cast<int>(), s[1].cast<int>(), g[0].cast<int>(), g[1].cast<int>()};
        return result_to_dict(cadfs_search(m, ins, config_from_dict(cfg), *model, tau_c));
    }, py::arg("map"), py::arg("start"), py::arg("goal"), py::arg("config"),
       py::arg("model"), py::arg("tau_c") = 0.05);
}
