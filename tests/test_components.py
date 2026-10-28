import unittest
from pathlib import Path

class TestComponents(unittest.TestCase):
    def test_required_structure(self):
        req_files = [
            "app/app.py",
            "scripts/train.py",
            "scripts/predict.py",
            "scripts/evaluate.py",
            "src/__init__.py",
            "src/data/__init__.py",
            "src/data/dataset.py",
            "src/models/__init__.py",
            "src/models/custom_cnn.py",
            "src/models/mobilenet.py",
            "src/training/__init__.py",
            "src/training/trainer.py",
            "src/evaluation/__init__.py",
            "src/evaluation/metrics.py",
            "src/evaluation/visualize.py",
            "src/inference/__init__.py",
            "src/inference/predictor.py",
            "docs/COMPARATIVE_ANALYSIS.md",
        ]
        for f in req_files:
            self.assertTrue(Path(f).exists(), f"Missing required file: {f}")

    def test_models_exist(self):
        self.assertTrue(Path("models").exists())
        self.assertTrue(Path("models/custom_cnn_best.keras").exists())

if __name__ == "__main__":
    unittest.main()
