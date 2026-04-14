#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shlex
from pathlib import Path
from typing import Any

import yaml


def to_cli_key(key: str) -> str:
    return f"--{key}"


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def params_to_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        args.extend([to_cli_key(key), normalize_value(value)])
    return args


def build_grid(sweeps: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweeps:
        return []
    keys = list(sweeps.keys())
    values: list[list[Any]] = []
    for key in keys:
        candidates = sweeps[key]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"sweeps.{key} must be non-empty list")
        values.append(candidates)
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_paired_grid(paired_sweeps: dict[str, list[Any]], field_name: str) -> list[dict[str, Any]]:
    if not paired_sweeps:
        return []
    keys = list(paired_sweeps.keys())
    values: list[list[Any]] = []
    lengths: list[int] = []
    for key in keys:
        candidates = paired_sweeps[key]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{field_name}.{key} must be non-empty list")
        values.append(candidates)
        lengths.append(len(candidates))
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise ValueError(
            f"{field_name} list lengths must match for zipped pairing: "
            + ", ".join(f"{k}={len(paired_sweeps[k])}" for k in keys)
        )
    size = unique_lengths[0]
    return [{key: paired_sweeps[key][idx] for key in keys} for idx in range(size)]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML must be mapping: {path}")
    return loaded


def build_experiments(config: dict[str, Any]) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    module = str(config.get("module", "src.ours_v2.train"))
    base_args = config.get("base_args", {})
    if not isinstance(base_args, dict):
        raise ValueError("base_args must be mapping")

    experiments: list[tuple[str, dict[str, Any]]] = []

    sweeps = config.get("sweeps", {})
    if sweeps and not isinstance(sweeps, dict):
        raise ValueError("sweeps must be mapping")

    paired_sweeps = config.get("paired_sweeps", {})
    zip_sweeps = config.get("zip_sweeps", {})
    zipped_sweeps = config.get("zipped_sweeps", {})
    paired_specs = [("paired_sweeps", paired_sweeps), ("zip_sweeps", zip_sweeps), ("zipped_sweeps", zipped_sweeps)]
    active_paired = [(name, spec) for name, spec in paired_specs if spec]
    if len(active_paired) > 1:
        raise ValueError("Use only one of paired_sweeps / zip_sweeps / zipped_sweeps")
    paired_name = ""
    paired_values: dict[str, list[Any]] = {}
    if active_paired:
        paired_name, paired_values = active_paired[0]
        if not isinstance(paired_values, dict):
            raise ValueError(f"{paired_name} must be mapping")

    if sweeps or paired_values:
        sweep_grid = build_grid(sweeps) if sweeps else [{}]
        paired_grid = build_paired_grid(paired_values, paired_name) if paired_values else [{}]
        overlap = set(sweeps.keys()) & set(paired_values.keys())
        if overlap:
            raise ValueError(
                "same key cannot be defined in both sweeps and paired_sweeps-like field: "
                + ", ".join(sorted(overlap))
            )
        idx = 0
        for sweep_combo in sweep_grid:
            for paired_combo in paired_grid:
                idx += 1
                params = dict(base_args)
                params.update(sweep_combo)
                params.update(paired_combo)
                name = f"grid_{idx:03d}"
                experiments.append((name, params))

    explicit = config.get("experiments", [])
    if explicit:
        if not isinstance(explicit, list):
            raise ValueError("experiments must be list")
        for i, row in enumerate(explicit, 1):
            if not isinstance(row, dict):
                raise ValueError(f"experiments[{i}] must be mapping")
            row_copy = dict(row)
            name = str(row_copy.pop("name", f"exp_{i:03d}"))
            params = dict(base_args)
            params.update(row_copy)
            experiments.append((name, params))

    if not experiments:
        experiments = [("single", dict(base_args))]

    return module, experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand experiment YAML/JSON to train commands.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python_bin", type=str, default="python3")
    parser.add_argument("--module", type=str, default="")
    args = parser.parse_args()

    config = load_yaml(args.config)
    module, experiments = build_experiments(config)
    if args.module:
        module = args.module

    for index, (name, params) in enumerate(experiments, 1):
        command = [args.python_bin, "-m", module]
        command.extend(params_to_args(params))
        print(f"{index}\t{name}\t{shlex.join(command)}")


if __name__ == "__main__":
    main()
