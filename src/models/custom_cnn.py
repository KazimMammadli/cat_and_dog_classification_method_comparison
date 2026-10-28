from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    l2_reg: float = 1e-4,
    name: str = "custom_cnn",
) -> keras.Model:
    """
    Builds a 3-block convolutional neural network trained from scratch.
    Includes BatchNormalization, L2 regularization, and progressive Dropout.
    """
    reg = regularizers.l2(l2_reg)

    return keras.Sequential([
        keras.Input(shape=input_shape),

        # Block 1 — 32 filters
        layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        # Block 2 — 64 filters
        layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        # Block 3 — 128 filters
        layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        # Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name=name)
