"""Strict, dependency-free versioned JSON training configuration."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from features import FEATURE_SCHEMA_VERSION
from ml.model import MODEL_ARCHITECTURE
from training.root_sampler import ROOT_SAMPLER_VERSION, RootSampler

CONFIG_VERSION = 3


def validate_config(raw: dict) -> dict:
    expected = {"config_version", "seed", "outer_iterations", "3max", "hu", "workers", "trainer_threads",
                "advantage_epochs", "average_epochs", "batch_size", "learning_rate", "strategy_capacity",
                "memory_byte_budget", "sample_byte_budget", "generation_byte_budget", "max_nodes", "max_depth",
                "checkpoint_every", "evaluation_every", "traversal_mode", "root_sampler_version", "feature_schema_version",
                "model_schema_version", "output_dir", "evaluation_max_nodes"}
    if isinstance(raw, dict) and raw.get("traversal_mode") == "outcome_sampling":
        expected.add("outcome_epsilon")
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a JSON object: {type(raw).__name__}")
    if set(raw) != expected:
        raise ValueError(f"Training config fields must match exactly; missing={expected - set(raw)}, extra={set(raw) - expected}")
    if (raw["config_version"] != CONFIG_VERSION or raw["root_sampler_version"] != ROOT_SAMPLER_VERSION
            or raw["feature_schema_version"] != FEATURE_SCHEMA_VERSION or raw["model_schema_version"] != MODEL_ARCHITECTURE):
        raise ValueError("Incompatible training config/schema versions")
    for name in expected - {"3max", "hu", "learning_rate", "traversal_mode", "model_schema_version", "output_dir", "outcome_epsilon"}:
        if type(raw[name]) is not int or raw[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Training config requires positive integer {name}: {raw[name]!r}")
    if raw["traversal_mode"] not in ("external_sampling", "outcome_sampling"):
        raise ValueError(f"Invalid traversal_mode: {raw['traversal_mode']}")
    if raw["traversal_mode"] == "outcome_sampling":
        epsilon = raw["outcome_epsilon"]
        if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError(f"Explicit outcome_epsilon must be in (0, 1]: {epsilon}")
    if type(raw["learning_rate"]) not in (float, int) or not math.isfinite(raw["learning_rate"]) or raw["learning_rate"] <= 0:
        raise ValueError(f"Invalid learning_rate: {raw['learning_rate']}")
    if not isinstance(raw["output_dir"], str) or not raw["output_dir"]:
        raise ValueError(f"Explicit output_dir required: {raw['output_dir']}")
    for name, count in (("3max", 3), ("hu", 2)):
        track = raw[name]
        if not isinstance(track, dict) or set(track) != {"enabled", "traversals_per_player", "root_sampling", "advantage_capacity", "advantage_byte_budget"} or type(track["enabled"]) is not bool or type(track["traversals_per_player"]) is not int or track["traversals_per_player"] < 1:
            raise ValueError(f"Invalid training track {name}: {track}")
        for field in ("advantage_capacity", "advantage_byte_budget"):
            if type(track[field]) is not int or track[field] < 1:
                raise ValueError(f"Track {name} requires positive integer {field}: {track[field]}")
        RootSampler(count, raw["seed"], track["root_sampling"])
    if not any(raw[t]["enabled"] for t in ("3max", "hu")):
        raise ValueError("At least one training track must be enabled")
    if (raw["hu"]["enabled"] and raw["hu"]["root_sampling"]["mixture"]["on_policy"] > 0
            and raw["hu"]["root_sampling"]["tournament_start_players"] == 3 and not raw["3max"]["enabled"]):
        raise ValueError("HU survivor roots require the 3max track; set tournament_start_players=2 for an explicit HU-only tournament")
    return json.loads(json.dumps(raw, allow_nan=False))


def load_config(path: str | Path) -> dict:
    return validate_config(json.loads(Path(path).read_text()))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
