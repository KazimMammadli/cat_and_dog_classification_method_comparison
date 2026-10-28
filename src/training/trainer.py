import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

def train_model(
    model,
    train_ds,
    val_ds,
    epochs: int = 15,
    lr: float = 1e-4,
    model_type: str = "mobilenetv2",
    output_dir: str = "outputs",
    model_save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standardized training routine with ReduceLROnPlateau, EarlyStopping, and ModelCheckpoint.
    """
    from tensorflow import keras

    os.makedirs(output_dir, exist_ok=True)
    if model_save_path is None:
        model_save_path = os.path.join(output_dir, f"{model_type}_best.keras")

    if model_type in ("custom_cnn", "scratch_cnn"):
        loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        patience_stop = 15
        patience_lr = 6
    else:
        loss = "binary_crossentropy"
        patience_stop = 5
        patience_lr = 3

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_json = os.path.join(output_dir, f"{model_type}_history.json")
    with open(hist_json, "w") as f:
        json.dump(hist_dict, f, indent=2)

    return {"model": model, "history": hist_dict, "save_path": model_save_path}
