#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained Cat vs Dog model on test dataset.

Usage:
  python scripts/evaluate.py --model mobilenetv2 --model-path models/mobilenetv2_best.keras --split-dir data_splits
"""

import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cats vs Dogs Model")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "custom_cnn"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model file")
    parser.add_argument("--split-dir", type=str, default="data_splits", help="Directory containing test/ folder")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save confusion matrix")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.split_dir):
        print(f"Error: Split directory does not exist: {args.split_dir}")
        sys.exit(1)

    try:
        import tensorflow as tf
        from src.data.dataset import load_datasets
        from src.evaluation.metrics import evaluate_model
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    print(f"Loading test dataset from {args.split_dir}...")
    _, _, test_ds, class_names = load_datasets(
        args.split_dir,
        batch_size=args.batch_size,
        model_type=args.model,
    )

    evaluate_model(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
