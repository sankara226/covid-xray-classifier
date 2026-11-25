"""Grad-CAM overlays for the final convolutional layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def gradcam(model, image: np.ndarray) -> np.ndarray:
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    conv = next(layer for layer in reversed(backbone.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    probe = tf.keras.Model(model.inputs, [conv.output, model.output])
    batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, prediction = probe(batch, training=False)
        target = prediction[:, 0]
    grads = tape.gradient(target, activations)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heat = tf.reduce_sum(weights * activations, axis=-1)[0]
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return tf.image.resize(heat[..., None], image.shape[:2]).numpy()[..., 0]


def save_gradcam_grid(model, images: list[np.ndarray], labels: list[str], output: str | Path) -> None:
    count = len(images)
    fig, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[0, i].imshow(np.clip(image / 255, 0, 1))
        axes[0, i].set_title(label)
        axes[1, i].imshow(np.clip(image / 255, 0, 1))
        axes[1, i].imshow(gradcam(model, image), cmap="jet", alpha=.4)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
