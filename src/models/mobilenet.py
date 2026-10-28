from typing import Tuple, Optional
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    weights: Optional[str] = "imagenet",
    frozen: bool = True,
    name: str = "mobilenetv2_classifier",
) -> keras.Model:
    """
    Builds a transfer learning classifier using pretrained MobileNetV2 feature extractor.
    """
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    base.trainable = not frozen

    inp = keras.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name=name)
