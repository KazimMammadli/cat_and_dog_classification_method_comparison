import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Optional

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_dataset_splits(
    data_dir: str,
    output_dir: str = "data_splits",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """
    Splits raw cat and dog image folders into train, val, and test splits.
    """
    random.seed(seed)
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    cats = [p for p in data_path.glob("**/*") if p.suffix.lower() in VALID_EXTENSIONS and "cat" in p.parent.name.lower()]
    dogs = [p for p in data_path.glob("**/*") if p.suffix.lower() in VALID_EXTENSIONS and "dog" in p.parent.name.lower()]

    if not cats or not dogs:
        # Fallback to direct subdirectories if naming differs
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            cats = [p for p in subdirs[0].glob("*") if p.suffix.lower() in VALID_EXTENSIONS]
            dogs = [p for p in subdirs[1].glob("*") if p.suffix.lower() in VALID_EXTENSIONS]

    random.shuffle(cats)
    random.shuffle(dogs)

    splits = ["train", "val", "test"]
    counts = {}

    for split in splits:
        (out_path / split / "cats").mkdir(parents=True, exist_ok=True)
        (out_path / split / "dogs").mkdir(parents=True, exist_ok=True)

    n_cats, n_dogs = len(cats), len(dogs)
    train_c_end = int(ratios[0] * n_cats)
    val_c_end = train_c_end + int(ratios[1] * n_cats)

    train_d_end = int(ratios[0] * n_dogs)
    val_d_end = train_d_end + int(ratios[1] * n_dogs)

    split_map = {
        "train": (cats[:train_c_end], dogs[:train_d_end]),
        "val": (cats[train_c_end:val_c_end], dogs[train_d_end:val_d_end]),
        "test": (cats[val_c_end:], dogs[val_d_end:]),
    }

    for split, (c_files, d_files) in split_map.items():
        for f in c_files:
            shutil.copy2(f, out_path / split / "cats" / f.name)
        for f in d_files:
            shutil.copy2(f, out_path / split / "dogs" / f.name)
        counts[split] = {"cats": len(c_files), "dogs": len(d_files)}

    return counts


def get_augmentation_layer(heavy: bool = False):
    """Constructs Keras Sequential image augmentation layer."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    if heavy:
        return keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.25),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ], name="heavy_augmentation")
    else:
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ], name="light_augmentation")


def load_datasets(
    split_dir: str,
    img_size: Tuple[int, int] = (160, 160),
    batch_size: int = 32,
    model_type: str = "mobilenetv2",
    seed: int = 42,
):
    """
    Builds tf.data.Dataset pipelines for train, val, test with appropriate preprocessing.
    """
    import tensorflow as tf
    from tensorflow import keras

    autotune = tf.data.AUTOTUNE

    def load_ds(folder, shuffle=False):
        return keras.utils.image_dataset_from_directory(
            folder,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            label_mode="binary",
        )

    train_ds = load_ds(os.path.join(split_dir, "train"), shuffle=True)
    val_ds = load_ds(os.path.join(split_dir, "val"), shuffle=False)
    test_ds = load_ds(os.path.join(split_dir, "test"), shuffle=False)
    class_names = train_ds.class_names

    if model_type.lower() == "scratch_cnn" or model_type.lower() == "custom_cnn":
        augment = get_augmentation_layer(heavy=True)
        def pre_train(x, y):
            return augment(tf.cast(x, tf.float32) / 255.0, training=True), y
        def pre_eval(x, y):
            return tf.cast(x, tf.float32) / 255.0, y
    else:
        # MobileNetV2 preprocessing
        augment = get_augmentation_layer(heavy=False)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        def pre_train(x, y):
            return preprocess_input(augment(x, training=True)), y
        def pre_eval(x, y):
            return preprocess_input(x), y

    train_ds = train_ds.map(pre_train, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(pre_eval, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(pre_eval, num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names
