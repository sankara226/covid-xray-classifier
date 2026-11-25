"""Train baseline and transfer learning models."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

from .config import Config
from .data import load_splits
from .model import build_small_cnn, build_transfer_model, enable_finetuning
from .pipeline import build_dataset

LOGGER = logging.getLogger(__name__)


def _callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3,
                                             restore_best_weights=True)]


def train(config: Config) -> dict[str, float]:
    splits = load_splits(config.artifacts_dir / "splits.npz")
    train_ds = build_dataset(splits["train"], image_size=config.image_size,
                             batch_size=config.batch_size, training=True, seed=config.seed)
    val_ds = build_dataset(splits["validation"], image_size=config.image_size,
                           batch_size=config.batch_size, training=False, seed=config.seed)
    Path(config.artifacts_dir).mkdir(parents=True, exist_ok=True)
    baseline = build_small_cnn(config.image_size, seed=config.seed)
    baseline_hist = baseline.fit(train_ds, validation_data=val_ds,
                                 epochs=config.training["baseline_epochs"], callbacks=_callbacks(), verbose=2)
    model, base = build_transfer_model(config.image_size, seed=config.seed)
    frozen_hist = model.fit(train_ds, validation_data=val_ds,
                            epochs=config.training["frozen_epochs"], callbacks=_callbacks(), verbose=2)
    enable_finetuning(model, base, config.training["finetune_layers"])
    tune_hist = model.fit(train_ds, validation_data=val_ds,
                          epochs=config.training["finetune_epochs"], callbacks=_callbacks(), verbose=2)
    model.save(config.artifacts_dir / "model.keras")
    return {"baseline_best_val_auc": float(max(baseline_hist.history["val_auc"])),
            "frozen_best_val_auc": float(max(frozen_hist.history["val_auc"])),
            "finetuned_best_val_auc": float(max(tune_hist.history["val_auc"]))}
