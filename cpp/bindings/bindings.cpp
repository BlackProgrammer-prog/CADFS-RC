#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cadfs/grid_map.hpp"
#include "cadfs/guidance.hpp"
#include "cadfs/mlp.hpp"
#include "cadfs/search_astar.hpp"
#include "cadfs/search_cadfs.hpp"
#include "cadfs/search_focal.hpp"
#include "cadfs/confidence.hpp"
#include "cadfs/controller.hpp"
#include "cadfs/expert.hpp"
#include "cadfs/search_cadfs_next.hpp"

#include <memory>


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

    mod.def(
            "run_cadfs_next",
            [](const GridMap& map,
               py::tuple start,
               py::tuple goal,
               const py::dict& cfg_dict,
               std::shared_ptr<GuidanceModel> model) {

                if (!model) {
                    throw py::value_error(
                            "run_cadfs_next requires a guidance model");
                }

                Config cfg = config_from_dict(cfg_dict);

                Instance instance{
                        start[0].cast<int>(),
                        start[1].cast<int>(),
                        goal[0].cast<int>(),
                        goal[1].cast<int>()
                };

                auto get_double =
                        [&](const char* key, double default_value) {
                            return cfg_dict.contains(key)
                                   ? cfg_dict[key].cast<double>()
                                   : default_value;
                        };

                auto get_string =
                        [&](const char* key,
                            const std::string& default_value) {
                            return cfg_dict.contains(key)
                                   ? cfg_dict[key].cast<std::string>()
                                   : default_value;
                        };

                std::vector<std::shared_ptr<const Expert>> experts;

                experts.push_back(
                        std::make_shared<GeometricExpert>(*model));

                experts.push_back(
                        std::make_shared<TopologicalExpert>(
                                cfg.connectivity));

                experts.push_back(
                        std::make_shared<GoalDistanceExpert>());

                std::vector<double> expert_weights{
                        1.0, 1.0, 1.0
                };

                if (cfg_dict.contains("expert_weights")) {
                    expert_weights =
                            cfg_dict["expert_weights"]
                                    .cast<std::vector<double>>();
                }

                ExpertFusion fusion(
                        std::move(experts),
                        std::move(expert_weights),
                        DisagreementMetric::Variance);

                CompositeConfidence confidence(
                        get_double("confidence_intra_weight", 1.0),
                        get_double("confidence_inter_weight", 1.0),
                        get_double("confidence_risk_weight", 0.0),
                        get_double("confidence_reference_weight", 0.0),
                        get_double("confidence_ood_weight", 0.0),
                        get_double("confidence_temperature", 0.05));

                const std::string controller_name =
                        get_string(
                                "next_controller",
                                "multiplicative");

                std::unique_ptr<FocalWidthController> controller;

                if (controller_name ==
                    std::string({'f', 'i', 'x', 'e', 'd'})) {
                    controller = std::make_unique<FixedController>(
                            cfg.tuned_fixed_w);
                }

                if (controller_name ==
                    std::string({'m', 'l', 'p'})) {
                    const std::string w1_key(
                            {'m', 'l', 'p', '_', 'w', '1'});
                    const std::string b1_key(
                            {'m', 'l', 'p', '_', 'b', '1'});
                    const std::string w2_key(
                            {'m', 'l', 'p', '_', 'w', '2'});
                    const std::string b2_key(
                            {'m', 'l', 'p', '_', 'b', '2'});
                    const std::string actions_key(
                            {'m', 'l', 'p', '_', 'a', 'c', 't', 'i', 'o', 'n', 's'});

                    auto required_vector = [&](const std::string& key) {
                        const py::str py_key(key);
                        if (!cfg_dict.contains(py_key)) {
                            throw py::key_error(key.c_str());
                        }
                        return cfg_dict[py_key].cast<std::vector<double>>();
                    };

                    std::vector<double> w1 = required_vector(w1_key);
                    std::vector<double> b1 = required_vector(b1_key);
                    std::vector<double> w2 = required_vector(w2_key);
                    std::vector<double> b2 = required_vector(b2_key);
                    std::vector<double> actions;

                    const py::str py_actions_key(actions_key);
                    if (cfg_dict.contains(py_actions_key)) {
                        actions = cfg_dict[py_actions_key]
                                .cast<std::vector<double>>();
                    }

                    const MLPControllerMode mode = actions.empty()
                            ? MLPControllerMode::Regression
                            : MLPControllerMode::Classification;
                    const std::size_t hidden_size = b1.size();

                    controller = std::make_unique<MLPController>(
                            std::move(w1), std::move(b1),
                            std::move(w2), std::move(b2),
                            14, hidden_size, std::move(actions), mode);
                }

                if (!controller) {

                if (controller_name == "linear") {
                    controller =
                            std::make_unique<LinearController>(
                                    get_double("lin_a", 1.0 / 3.0),
                                    get_double("lin_b", 1.0 / 3.0),
                                    get_double("lin_c", 1.0 / 3.0));

                } else if (controller_name == "threshold") {
                    controller =
                            std::make_unique<ThresholdController>(
                                    get_double(
                                            "controller_confidence_threshold",
                                            0.3),
                                    get_double(
                                            "controller_disagreement_threshold",
                                            0.5),
                                    get_double(
                                            "controller_conservative_width",
                                            1.0));

                } else {
                    controller =
                            std::make_unique<
                                    MultiplicativeController>();
                }

                }

                SearchResult result =
                        cadfs_next_search(
                                map,
                                instance,
                                cfg,
                                fusion,
                                confidence,
                                *controller);

                return result_to_dict(result);
            },
            py::arg("map"),
            py::arg("start"),
            py::arg("goal"),
            py::arg("config"),
            py::arg("model"));}
