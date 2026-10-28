import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from .visualize import plot_confusion_matrix

def evaluate_model(model, test_ds, class_names=None, output_dir: str = "outputs", model_name: str = "Model"):
    if class_names is None:
        class_names = ["Cat", "Dog"]

    y_true, y_prob = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    border = "=" * 40
    print(f"\n{border}")
    print(f"  {model_name} Test Results")
    print(f"{border}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"{border}\n")

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=class_names, title=f"Confusion Matrix — {model_name}", save_path=cm_path)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
