"""Image discovery, duplicate removal, sampling, and leakage-safe splitting."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
CLASSES = ("Normal", "COVID")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        LOGGER.info("TensorFlow is unavailable; seeded Python and NumPy")


def dataset_root(data_dir: str | Path) -> Path:
    """Locate the folder holding COVID/ and Normal/, which Kaggle nests one level down."""
    root = Path(data_dir)
    if not all((root / cls).exists() for cls in CLASSES):
        candidates = [p for p in root.rglob("COVID") if p.is_dir() and (p.parent / "Normal").is_dir()]
        if candidates:
            root = candidates[0].parent
    return root


def discover_images(data_dir: str | Path) -> list[tuple[Path, int]]:
    root = dataset_root(data_dir)
    rows = [
        (p, label)
        for label, cls in enumerate(CLASSES)
        for p in sorted(
            (root / cls / "images").glob("*") if (root / cls / "images").exists() else (root / cls).glob("*")
        )
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not rows:
        raise FileNotFoundError(f"No COVID/Normal images found under {root}")
    return rows


def file_hash(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deduplicate(rows: list[tuple[Path, int]]) -> list[tuple[Path, int]]:
    seen: dict[str, int] = {}
    kept = []
    for path, label in rows:
        digest = file_hash(path)
        if digest in seen:
            if seen[digest] != label:
                raise ValueError(f"Duplicate image has conflicting labels: {path}")
            continue
        seen[digest] = label
        kept.append((path, label))
    return kept


def split_data(
    rows: list[tuple[Path, int]],
    *,
    seed: int,
    max_per_class: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[tuple[Path, int]]]:
    """Balance, stratify, and assert hash-disjoint partitions."""
    import pandas as pd

    frame = pd.DataFrame(rows, columns=("path", "label"))
    count = min(max_per_class, int(frame.label.value_counts().min()))
    if count < max_per_class:
        LOGGER.warning("At least one class has only %d images; requested %d", count, max_per_class)
    balanced_frame = frame.groupby("label", group_keys=False).sample(n=count, random_state=seed)
    balanced = list(balanced_frame.itertuples(index=False, name=None))
    paths = np.array([str(p) for p, _ in balanced])
    labels = np.array([y for _, y in balanced])
    train_idx, other_idx = train_test_split(
        np.arange(len(labels)),
        test_size=validation_fraction + test_fraction,
        stratify=labels,
        random_state=seed,
    )
    rel_test = test_fraction / (validation_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        other_idx, test_size=rel_test, stratify=labels[other_idx], random_state=seed
    )
    result = {
        name: [(Path(paths[i]), int(labels[i])) for i in idx]
        for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))
    }
    hashes = [{file_hash(p) for p, _ in result[name]} for name in result]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Hash duplicate crosses splits")
    return result


def prepare(
    data_dir: str | Path,
    *,
    seed: int,
    max_per_class: int,
    validation_fraction: float,
    test_fraction: float,
    output_dir: str | Path,
) -> dict[str, list[tuple[Path, int]]]:
    rows = deduplicate(discover_images(data_dir))
    splits = split_data(
        rows,
        seed=seed,
        max_per_class=max_per_class,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "splits.npz", **{k: np.array([[str(p), y] for p, y in v]) for k, v in splits.items()})
    report = source_report(dataset_root(data_dir))
    (output / "source_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return splits


def source_report(data_root: str | Path) -> dict[str, object]:
    """Summarize metadata URL hosts and whether classes share a source."""
    import pandas as pd

    root = Path(data_root)
    sources: dict[str, dict[str, int]] = {}
    for name in CLASSES:
        metadata = root / f"{name}.metadata.xlsx"
        if not metadata.exists():
            continue
        table = pd.read_excel(metadata)
        hosts = table["URL"].map(lambda value: urlparse(str(value)).hostname or "unknown")
        sources[name] = {str(k): int(v) for k, v in hosts.value_counts().items()}
    shared = sorted(set(sources.get("COVID", {})) & set(sources.get("Normal", {})))
    readme = next(iter(sorted(root.glob("README*"))), None)
    normal_subgroups = {}
    if readme:
        content = readme.read_text(errors="ignore")
        for label, pattern in (("RSNA", r"(\d+)\s+RSNA"), ("pediatric_pneumonia", r"(\d+)\s+Kaggle\s*\[9\]")):
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                normal_subgroups[label] = int(match.group(1))
    return {
        "sources_by_class": sources,
        "shared_sources": shared,
        "class_and_source_confounded": bool(sources and not shared),
        "normal_readme_breakdown": normal_subgroups,
    }


def load_splits(path: str | Path) -> dict[str, list[tuple[Path, int]]]:
    data = np.load(path, allow_pickle=False)
    return {key: [(Path(str(row[0])), int(row[1])) for row in data[key]] for key in data.files}


def image_array(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
