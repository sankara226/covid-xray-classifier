"""CNN baselines and MobileNetV2 transfer model."""

from __future__ import annotations


def build_small_cnn(image_size: int = 224, *, seed: int = 42):
    import tensorflow as tf
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05, seed=seed),
        tf.keras.layers.RandomZoom(0.10, seed=seed + 1),
        tf.keras.layers.RandomContrast(0.10, seed=seed + 2),
    ], name="augment")
    inputs = tf.keras.Input((image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1 / 255.0)(augmentation(inputs))
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def build_transfer_model(image_size: int = 224, *, pretrained: bool = True, seed: int = 42):
    import tensorflow as tf
    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights,
                                             input_shape=(image_size, image_size, 3))
    base.trainable = False
    inputs = tf.keras.Input((image_size, image_size, 3))
    x = tf.keras.layers.RandomRotation(0.05, seed=seed)(inputs)
    x = tf.keras.layers.RandomZoom(0.1, seed=seed + 1)(x)
    x = tf.keras.layers.RandomContrast(0.1, seed=seed + 2)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model, base


def enable_finetuning(model, base, layers: int = 30):
    import tensorflow as tf
    base.trainable = True
    for layer in base.layers[:-layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
