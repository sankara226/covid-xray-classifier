"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    seed: int
    image_size: int
    batch_size: int
    max_per_class: int
    validation_fraction: float
    test_fraction: float
    target_sensitivity: float
    data_dir: Path
    artifacts_dir: Path
    figure_dir: Path
    mask: dict[str, tuple[int, int, int, int]]
    training: dict[str, int]


def load_config(
    path: str | Path = "configs/default.yaml",
    *,
    data_dir: str | Path | None = None,
    max_per_class: int | None = None,
) -> Config:
    """Load YAML settings and reject invalid split or image parameters."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    val, test = float(raw["validation_fraction"]), float(raw["test_fraction"])
    if val <= 0 or test <= 0 or val + test >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to less than one")
    if int(raw["image_size"]) < 1 or int(raw["batch_size"]) < 1 or int(raw["max_per_class"]) < 1:
        raise ValueError("image_size, batch_size, and max_per_class must be positive")
    sensitivity = float(raw["target_sensitivity"])
    if not 0 < sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    mask = {k: tuple(map(int, v)) for k, v in raw["mask"].items()}
    for name, (r0, r1, c0, c1) in mask.items():
        if not (0 <= r0 < r1 <= raw["image_size"] and 0 <= c0 < c1 <= raw["image_size"]):
            raise ValueError(f"mask.{name} must be within image bounds")
    selected_max = int(raw["max_per_class"] if max_per_class is None else max_per_class)
    if selected_max < 1:
        raise ValueError("max_per_class must be positive")
    return Config(
        int(raw["seed"]),
        int(raw["image_size"]),
        int(raw["batch_size"]),
        selected_max,
        val,
        test,
        sensitivity,
        Path(data_dir) if data_dir else Path(raw["data_dir"]),
        Path(raw["artifacts_dir"]),
        Path(raw["figure_dir"]),
        mask,
        dict(raw["training"]),
    )
