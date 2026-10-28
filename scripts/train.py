#!/usr/bin/env python3
"""
train.py — Train Cats vs Dogs classification models (Custom CNN or MobileNetV2).

Usage:
  python scripts/train.py --model mobilenetv2 --data-dir data/train --epochs 15
  python scripts/train.py --model custom_cnn --data-dir data/train --epochs 60
"""

import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs. Dogs Image Classifiers")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["custom_cnn", "mobilenetv2"], help="Model architecture to train")
    parser.add_argument("--split-dir", type=str, default="data_splits", help="Directory containing train/val/test splits")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save training curves and history")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save .keras model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    try:
        from src.data.dataset import load_datasets
        from src.models import get_model
        from src.training.trainer import train_model
        from src.evaluation.visualize import plot_history
    except ImportError as e:
        print(f"Error importing ML dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.exists(args.split_dir):
        print(f"Error: Split directory does not exist: {args.split_dir}")
        print("Please prepare dataset splits first.")
        sys.exit(1)

    print(f"Loading dataset from {args.split_dir}...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.split_dir,
        batch_size=args.batch_size,
        model_type=args.model,
    )

    print(f"Building {args.model} model...")
    model = get_model(args.model)

    os.makedirs(args.models_dir, exist_ok=True)
    save_path = os.path.join(args.models_dir, f"{args.model}_best.keras")

    print("Starting training...")
    result = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        output_dir=args.output_dir,
        model_save_path=save_path,
    )

    curve_path = os.path.join(args.output_dir, f"{args.model}_history.png")
    plot_history(result["history"], title=f"{args.model.upper()} — Training History", save_path=curve_path)
    print(f"✓ Training complete. Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
