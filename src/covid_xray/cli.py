"""Command line interface for preparation, training, evaluation, and audits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .audit import mask_variants
from .config import load_config
from .data import image_array, load_splits, prepare, seed_everything
from .evaluate import evaluate, select_threshold
from .pipeline import build_dataset
from .plots import save_error_gallery, save_evaluation_figures, save_masking_figure

LOGGER = logging.getLogger("covid_xray")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Educational COVID/Normal X-ray classifier; not a diagnostic tool.")
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "audit", "gradcam"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--max-per-class", type=int)
    return parser


def _predictions(model, rows, config):
    ds = build_dataset(rows, image_size=config.image_size, batch_size=config.batch_size,
                       training=False, seed=config.seed)
    scores = model.predict(ds, verbose=0).reshape(-1)
    return np.array([label for _, label in rows]), scores


def _load_model(config):
    import tensorflow as tf
    path = config.artifacts_dir / "model.keras"
    if not path.exists():
        raise FileNotFoundError(f"Trained model missing: {path}; run `python -m covid_xray train` first")
    return tf.keras.models.load_model(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args()
    config = load_config(args.config, data_dir=args.data_dir, max_per_class=args.max_per_class)
    seed_everything(config.seed)
    if args.command != "prepare":
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        LOGGER.info("Compute device: %s", devices[0].name if devices else "CPU")
    if args.command == "prepare":
        data_dir = config.data_dir
        if not data_dir.exists():
            import kagglehub
            data_dir = Path(kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database"))
        splits = prepare(data_dir, seed=config.seed, max_per_class=config.max_per_class,
                         validation_fraction=config.validation_fraction, test_fraction=config.test_fraction,
                         output_dir=config.artifacts_dir)
        LOGGER.info("Prepared splits: %s", {k: len(v) for k, v in splits.items()})
    elif args.command == "train":
        from .train import train
        results = train(config)
        (config.artifacts_dir / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("Validation metrics: %s", results)
    elif args.command == "evaluate":
        splits = load_splits(config.artifacts_dir / "splits.npz")
        model = _load_model(config)
        y_val, val_scores = _predictions(model, splits["validation"], config)
        threshold = select_threshold(y_val, val_scores, config.target_sensitivity)
        y_test, test_scores = _predictions(model, splits["test"], config)
        results = evaluate(y_test, test_scores, threshold)
        results["validation_sensitivity_at_threshold"] = float(np.mean(val_scores[y_val == 1] >= threshold))
        results["precision_by_prevalence"] = {str(p): evaluate_precision(results, p) for p in (.5, .1, .05, .01)}
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (config.artifacts_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        source_path = config.artifacts_dir / "source_report.json"
        source_summary = json.loads(source_path.read_text()) if source_path.exists() else {}
        training_path = config.artifacts_dir / "training_metrics.json"
        training_metrics = json.loads(training_path.read_text()) if training_path.exists() else {}
        model_card = {"task": "COVID vs Normal chest X-ray classification; educational only, not a diagnostic tool",
                      "training_metrics": training_metrics, "results": results, "source_audit": source_summary,
                      "limitations": ["COVID and Normal sources are disjoint in the supplied dataset metadata",
                                      "No patient identifiers are available to rule out patient overlap",
                                      "Balanced test data does not represent real-world prevalence",
                                      "This model is not validated for clinical use"]}
        (config.artifacts_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
        np.savez(config.artifacts_dir / "evaluation_predictions.npz", y_test=y_test, scores=test_scores,
                 threshold=np.array(threshold))
        save_evaluation_figures(y_test, test_scores, threshold, config.figure_dir)
        save_error_gallery(splits["test"], y_test, test_scores, threshold, config.figure_dir, config.image_size)
        LOGGER.info("Evaluation: %s", results)
    elif args.command == "audit":
        splits = load_splits(config.artifacts_dir / "splits.npz")
        model = _load_model(config)
        rows = splits["test"]
        labels = np.array([label for _, label in rows])
        values = {name: [] for name in ("full", "lungs_hidden", "lungs_only", "corner_hidden")}
        for path, _ in rows:
            img = image_array(path, config.image_size)
            for name, variant in mask_variants(img, config.mask["center_box"], config.mask["corner_box"]).items():
                values[name].append(variant)
        import tensorflow as tf
        from sklearn.metrics import roc_auc_score
        metrics = {}
        for name, images in values.items():
            ds = tf.data.Dataset.from_tensor_slices(np.asarray(images)).batch(config.batch_size)
            scores = model.predict(ds, verbose=0).reshape(-1)
            metrics[name] = float(roc_auc_score(labels, scores))
        (config.artifacts_dir / "audit.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_masking_figure(metrics, config.figure_dir)
        LOGGER.info("Masking audit: %s", metrics)
    elif args.command == "gradcam":
        from .gradcam import save_gradcam_grid

        splits = load_splits(config.artifacts_dir / "splits.npz")
        model = _load_model(config)
        rows = splits["test"][:6]
        images = [image_array(path, config.image_size) for path, _ in rows]
        labels = ["COVID" if label else "Normal" for _, label in rows]
        save_gradcam_grid(model, images, labels, config.figure_dir / "gradcam_grid.png")
        LOGGER.info("Grad-CAM grid saved to %s", config.figure_dir / "gradcam_grid.png")


def evaluate_precision(results: dict, prevalence: float) -> float:
    from .evaluate import precision_at_prevalence
    return precision_at_prevalence(float(results["sensitivity"]), float(results["specificity"]), prevalence)


if __name__ == "__main__":
    main()
