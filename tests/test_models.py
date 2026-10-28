import unittest

class TestModelArchitectures(unittest.TestCase):
    def test_custom_cnn_build(self):
        try:
            from src.models.custom_cnn import build_custom_cnn
            model = build_custom_cnn(input_shape=(160, 160, 3))
            self.assertEqual(model.input_shape, (None, 160, 160, 3))
            self.assertEqual(model.output_shape, (None, 1))
        except ImportError:
            self.skipTest("TensorFlow not installed in current environment")

    def test_mobilenet_build(self):
        try:
            from src.models.mobilenet import build_mobilenetv2_classifier
            model = build_mobilenetv2_classifier(input_shape=(160, 160, 3), weights=None, frozen=True)
            self.assertEqual(model.input_shape, (None, 160, 160, 3))
            self.assertEqual(model.output_shape, (None, 1))
        except ImportError:
            self.skipTest("TensorFlow not installed in current environment")

if __name__ == "__main__":
    unittest.main()
