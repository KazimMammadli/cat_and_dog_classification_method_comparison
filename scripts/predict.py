#!/usr/bin/env python3
"""
predict.py — Classify an image as Cat or Dog using trained models.

Usage:
  python scripts/predict.py path/to/image.jpg --model mobilenetv2
  python scripts/predict.py path/to/image.jpg --model custom_cnn --model-path models/custom_cnn_best.keras
"""

import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Predict Cat vs. Dog from an Image")
    parser.add_argument("image_path", type=str, help="Path to input image (.jpg, .png, etc.)")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "custom_cnn"], help="Model architecture type")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .keras model checkpoint")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join("models", f"{args.model}_best.keras")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at: {model_path}")
        sys.exit(1)

    try:
        from src.inference.predictor import CatDogPredictor
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Loading {args.model} from {model_path}...")
    predictor = CatDogPredictor(model_path, model_type=args.model)

    print(f"Classifying image: {args.image_path}...")
    res = predictor.predict(args.image_path)

    border = "=" * 40
    print(f"\n{border}")
    print("         CLASSIFICATION RESULT")
    print(f"{border}")
    print(f"  Prediction : {res[label].upper()}")
    print(f"  Confidence : {res[confidence_percent]:.2f}%")
    print(f"  Cat Score  : {res[cat_confidence]*100:.2f}%")
    print(f"  Dog Score  : {res[dog_confidence]*100:.2f}%")
    print(f"{border}\n")


if __name__ == "__main__":
    main()
