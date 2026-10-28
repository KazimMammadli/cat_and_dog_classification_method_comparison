import unittest
import numpy as np
from PIL import Image

class TestInferencePipeline(unittest.TestCase):
    def test_dummy_image_preprocessing(self):
        try:
            import tensorflow
            from src.inference.predictor import CatDogPredictor
        except ImportError:
            self.skipTest("TensorFlow not installed")

        # Create dummy image
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        # Test PIL image resize shape
        resized = img.resize((160, 160))
        self.assertEqual(resized.size, (160, 160))

if __name__ == "__main__":
    unittest.main()
