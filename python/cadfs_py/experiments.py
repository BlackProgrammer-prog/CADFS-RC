"""Shared, reproducible method definitions for CADFS experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

Method = Callable[[Any, tuple[int, int], tuple[int, int], dict], dict]

DEFAULT_NEXT = {
    "expert": {"name": "uniform", "weights": [1 / 3, 1 / 3, 1 / 3]},
    "confidence": {
        "intra_weight": 1.0,
        "inter_weight": 1.0,
        "risk_weight": 0.0,
        "reference_weight": 0.0,
        "ood_weight": 0.0,
        "temperature": 0.05,
    },
    "controller": {"type": "multiplicative"},
}

METHOD_SUITES = {
    "main": [
        "astar", "wastar", "focal_plain", "learn_focal_W",
        "learn_focal_wstar", "cadfs", "cadfs_next",
    ],
    "next": [
        "cadfs", "learn_focal_wstar", "cadfs_next",
        "cadfs_next_geometry", "cadfs_next_uniform",
        "cadfs_next_nointra", "cadfs_next_nointer",
        "cadfs_next_noconf", "cadfs_next_linear",
        "cadfs_next_threshold", "cadfs_next_fixed",
    ],
    "legacy": [
        "astar", "wastar", "focal_plain", "learn_focal_W",
        "learn_focal_wstar", "cadfs", "cadfs_linear", "cadfs_norisk",
        "cadfs_randomrisk", "cadfs_permutedrisk", "cadfs_nofallback",
        "cadfs_noconf",
    ],
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_settings(root: Path, next_path: Path | None = None) -> tuple[dict, dict]:
    legacy = load_json(root / "results/models/tuned.json")
    path = next_path or root / "results/models/tuned_next.json"
    next_settings = load_json(path) if path.exists() else DEFAULT_NEXT
    return legacy, next_settings


def next_config(base: dict, settings: dict, **overrides: Any) -> dict:
    expert = settings.get("expert", DEFAULT_NEXT["expert"])
    confidence = settings.get("confidence", DEFAULT_NEXT["confidence"])
    controller = settings.get("controller", DEFAULT_NEXT["controller"])
    config = dict(
        base,
        expert_weights=list(expert["weights"]),
        confidence_intra_weight=confidence.get("intra_weight", 1.0),
        confidence_inter_weight=confidence.get("inter_weight", 1.0),
        confidence_risk_weight=confidence.get("risk_weight", 0.0),
        confidence_reference_weight=confidence.get("reference_weight", 0.0),
        confidence_ood_weight=confidence.get("ood_weight", 0.0),
        confidence_temperature=confidence.get("temperature", 0.05),
        next_controller=controller.get("type", "multiplicative"),
    )
    config.update({key: value for key, value in controller.items() if key != "type"})
    config.update(overrides)
    return config


def build_methods(engine: Any, ensemble: Any, legacy: dict,
                  next_settings: dict, mlp_path: Path | None = None) -> dict[str, Method]:
    base = dict(legacy["base"])
    base["theta_c"] = legacy["theta_c"]
    tau = legacy["tau_c"]
    width = base["W"]
    tuned_width = legacy["w_star"]

    def legacy_variant(**overrides: Any) -> Method:
        def run(map_, start, goal, config):
            return engine.run_cadfs(
                map_, start, goal, dict(config, **overrides), ensemble, tau)
        return run

    def next_variant(**overrides: Any) -> Method:
        def run(map_, start, goal, config):
            candidate = next_config(config, next_settings, **overrides)
            return engine.run_cadfs_next(map_, start, goal, candidate, ensemble)
        return run

    methods: dict[str, Method] = {
        "astar": lambda m, s, g, c: engine.run_astar(m, s, g, c, 1.0),
        "wastar": lambda m, s, g, c: engine.run_astar(m, s, g, c, width),
        "focal_plain": lambda m, s, g, c: engine.run_focal(m, s, g, c, None, width),
        "learn_focal_W": lambda m, s, g, c: engine.run_focal(m, s, g, c, ensemble, width),
        "learn_focal_wstar": lambda m, s, g, c: engine.run_focal(
            m, s, g, c, ensemble, tuned_width),
        "cadfs": legacy_variant(),
        "cadfs_linear": legacy_variant(
            controller="linear", lin_a=1 / 3, lin_b=1 / 3, lin_c=1 / 3),
        "cadfs_norisk": legacy_variant(risk_mode="off"),
        "cadfs_randomrisk": legacy_variant(risk_mode="random", risk_seed=11),
        "cadfs_permutedrisk": legacy_variant(risk_mode="permuted"),
        "cadfs_nofallback": legacy_variant(fallback_enabled=False),
        "cadfs_noconf": legacy_variant(confidence_enabled=False),
        "cadfs_next": next_variant(),
        "cadfs_next_geometry": next_variant(expert_weights=[1.0, 0.0, 0.0]),
        "cadfs_next_uniform": next_variant(expert_weights=[1 / 3, 1 / 3, 1 / 3]),
        "cadfs_next_nointra": next_variant(confidence_intra_weight=0.0),
        "cadfs_next_nointer": next_variant(confidence_inter_weight=0.0),
        "cadfs_next_noconf": next_variant(confidence_enabled=False),
        "cadfs_next_linear": next_variant(next_controller="linear"),
        "cadfs_next_threshold": next_variant(next_controller="threshold"),
        "cadfs_next_fixed": next_variant(
            next_controller="fixed", tuned_fixed_w=tuned_width),
    }

    if mlp_path and mlp_path.exists():
        model = load_json(mlp_path)
        mlp = {
            "next_controller": "mlp",
            "mlp_w1": model["w1"], "mlp_b1": model["b1"],
            "mlp_w2": model["w2"], "mlp_b2": model["b2"],
        }
        if model.get("actions"):
            mlp["mlp_actions"] = model["actions"]
        methods["cadfs_next_mlp"] = next_variant(**mlp)

    METHOD_SUITES["full"] = list(methods)
    return methods


def select_methods(methods: dict[str, Method], suite: str,
                   requested: list[str] | None = None) -> dict[str, Method]:
    names = requested or METHOD_SUITES[suite]
    unknown = [name for name in names if name not in methods]
    if unknown:
        raise ValueError(f"unknown or unavailable methods: {unknown}")
    return {name: methods[name] for name in names}
