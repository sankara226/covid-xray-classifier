"""TensorFlow image pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_dataset(rows: list[tuple[Path, int]], *, image_size: int, batch_size: int,
                 training: bool, seed: int):
    import tensorflow as tf

    paths = [str(path) for path, _ in rows]
    labels = np.array([label for _, label in rows], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (image_size, image_size))
        return image, label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
