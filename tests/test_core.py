from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from covid_xray.audit import mask_region, mask_variants
from covid_xray.config import load_config
from covid_xray.data import deduplicate, split_data
from covid_xray.evaluate import precision_at_prevalence, select_threshold


def test_split_is_disjoint_and_stratified(tmp_path: Path):
    rows = []
    for label in (0, 1):
        for i in range(20):
            path = tmp_path / f"{label}-{i}.png"
            Image.fromarray(np.full((4, 4), label * 100 + i, dtype=np.uint8)).save(path)
            rows.append((path, label))
    split = split_data(rows, seed=42, max_per_class=20, validation_fraction=0.15, test_fraction=0.15)
    assert {key: len(value) for key, value in split.items()} == {"train": 28, "validation": 6, "test": 6}
    hashes = [{row[0].name for row in split[key]} for key in split]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    assert all(sum(y for _, y in values) == len(values) // 2 for values in split.values())


def test_deduplicate_and_conflicting_labels(tmp_path: Path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    payload = np.zeros((3, 3), dtype=np.uint8)
    Image.fromarray(payload).save(first)
    Image.fromarray(payload).save(second)
    assert deduplicate([(first, 0), (second, 0)]) == [(first, 0)]
    with pytest.raises(ValueError, match="conflicting labels"):
        deduplicate([(first, 0), (second, 1)])


def test_threshold_meets_target_on_validation_scores():
    y = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = select_threshold(y, scores, 0.95)
    assert threshold == 0.3
    assert np.mean(scores[y == 1] >= threshold) >= 0.95


def test_mask_changes_only_target_region():
    image = np.ones((8, 8, 3), dtype=np.uint8)
    changed = mask_region(image, (2, 5, 1, 4))
    assert np.all(changed[2:5, 1:4] == 0)
    assert np.all(changed[:2] == 1) and np.all(changed[5:] == 1)
    assert np.all(changed[:, 4:] == 1) and np.all(changed[:, :1] == 1)


def test_config_rejects_invalid_splits(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "seed: 1\nimage_size: 32\nbatch_size: 2\nmax_per_class: 2\n"
        "validation_fraction: 0.6\ntest_fraction: 0.5\ntarget_sensitivity: 0.95\n"
        "data_dir: data\nartifacts_dir: out\nfigure_dir: figs\n"
        "mask: {center_box: [1, 2, 1, 2], corner_box: [0, 1, 0, 1]}\ntraining: {}\n"
    )
    with pytest.raises(ValueError, match="sum to less"):
        load_config(config)


def test_precision_prevalence_closed_form():
    value = precision_at_prevalence(0.9, 0.8, 0.01)
    assert value == pytest.approx(0.009 / (0.009 + 0.2 * 0.99))


def test_mask_variants_match_audit_geometry():
    image = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3) + 1
    variants = mask_variants(image, (2, 6, 2, 6), (0, 4, 0, 4))
    assert np.array_equal(variants["full"], image)
    assert np.all(variants["lungs_hidden"][2:6, 2:6] == 0)
    assert np.array_equal(variants["lungs_only"][2:6, 2:6], image[2:6, 2:6])
    assert variants["lungs_only"].sum() == image[2:6, 2:6].sum()
    assert np.all(variants["corner_hidden"][:4, :4] == 0)
    assert np.array_equal(variants["corner_hidden"][4:], image[4:])
