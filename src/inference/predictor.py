import os
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
from PIL import Image

class CatDogPredictor:
    """
    Inference wrapper for trained Cat vs. Dog classification models.
    """

    def __init__(self, model_path: str, model_type: str = "mobilenetv2", img_size=(160, 160)):
        import tensorflow as tf
        self.model_path = str(model_path)
        self.model_type = model_type.lower()
        self.img_size = img_size
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype("uint8")).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        img = img.resize(self.img_size)
        arr = np.array(img, dtype=np.float32)

        if "mobilenet" in self.model_type:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            arr = preprocess_input(arr)
        else:
            arr = arr / 255.0

        return np.expand_dims(arr, axis=0)

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        x = self.preprocess_image(image)
        raw_prob = float(self.model.predict(x, verbose=0).flatten()[0])

        # 0 = Cat, 1 = Dog
        label = "Dog" if raw_prob >= 0.5 else "Cat"
        dog_conf = raw_prob
        cat_conf = 1.0 - raw_prob

        return {
            "label": label,
            "probability": raw_prob,
            "cat_confidence": cat_conf,
            "dog_confidence": dog_conf,
            "confidence_percent": (dog_conf if label == "Dog" else cat_conf) * 100.0,
            "model_type": self.model_type,
        }
