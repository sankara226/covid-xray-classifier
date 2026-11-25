"""Saved evaluation and audit figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_evaluation_figures(y_true: np.ndarray, scores: np.ndarray, threshold: float,
                           output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = scores >= threshold
    ConfusionMatrixDisplay.from_predictions(y_true, predictions)
    plt.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_true, scores)
    plt.savefig(output / "roc_curve.png", bbox_inches="tight")
    plt.close()
    plt.hist(scores[y_true == 0], bins=20, alpha=.65, label="Normal")
    plt.hist(scores[y_true == 1], bins=20, alpha=.65, label="COVID")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.4f}")
    plt.xlabel("Predicted COVID probability"); plt.ylabel("Images"); plt.legend()
    plt.savefig(output / "confidence_histogram.png", bbox_inches="tight")
    plt.close()


def save_masking_figure(metrics: dict[str, float], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(metrics)
    values = [metrics[key] for key in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Test ROC-AUC")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output / "masking_audit.png", bbox_inches="tight")
    plt.close()


def save_error_gallery(rows, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                       output_dir: str | Path, image_size: int = 224) -> None:
    from .data import image_array

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = scores >= threshold
    groups = (("Missed COVID", [i for i, (y, p) in enumerate(zip(y_true, predictions)) if y == 1 and not p]),
              ("False alarms", [i for i, (y, p) in enumerate(zip(y_true, predictions)) if y == 0 and p]))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    for row, (title, indices) in enumerate(groups):
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(indices):
                i = indices[col]
                ax.imshow(image_array(rows[i][0], image_size).astype(np.uint8))
                ax.set_title(f"{title}\np={scores[i]:.2f}")
    fig.savefig(output / "error_gallery.png", bbox_inches="tight")
    plt.close(fig)
