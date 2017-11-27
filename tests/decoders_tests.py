import unittest
import numpy as np

from mynn import decoders


class BinaryDecoderTestCase(unittest.TestCase):
    def test_default_constructor(self):
        d = decoders.ThresholdBinaryDecoder()
        self.assertEqual(d.probability_threshold, 0.5)
        self.assertEqual(d.class_a, 1)
        self.assertEqual(d.class_b, 0)

    def test_override_constructor(self):
        d = decoders.ThresholdBinaryDecoder(
            probability_threshold=0.7,
            class_a=15,
            class_b=2
        )
        self.assertEqual(d.probability_threshold, 0.7)
        self.assertEqual(d.class_a, 15)
        self.assertEqual(d.class_b, 2)

    def test_error_constructor(self):
        with self.assertRaises(ValueError):
            decoders.ThresholdBinaryDecoder(
                class_a=1,
                class_b=1
            )

    def test_predict_simple(self):
        d = decoders.ThresholdBinaryDecoder()
        probas = np.array([0, 0.2, 0.4999, 0.5, 0.50000001, 0.6, 1])

        out = d.predict(probas)
        self.assertListEqual(
            out.tolist(),
            [[0], [0], [0], [0], [1], [1], [1]]
        )

    def test_predict_non_default(self):
        d = decoders.ThresholdBinaryDecoder(
            probability_threshold=0.7,
            class_a=15,
            class_b=2
        )
        probas = np.array([0, 0.2, 0.4999, 0.5, 0.50000001, 0.6, 1])

        out = d.predict(probas)
        self.assertListEqual(
            out.tolist(),
            [[2], [2], [2], [2], [2], [2], [15]]
        )


class OneHotDecoderTestCase(unittest.TestCase):
    def test_default_constructor(self):
        d = decoders.OneHotDecoder()
        self.assertIsNone(d.classes)

    def test_constructor(self):
        d = decoders.OneHotDecoder(classes=[1, 2, 3])
        self.assertListEqual(d.classes.tolist(), [1, 2, 3])

    def test_error_constructor(self):
        with self.assertRaises(TypeError):
            decoders.OneHotDecoder(classes=1)

    def test_prediction_indices(self):
        d = decoders.OneHotDecoder()

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        results = d.predict(probas)
        self.assertListEqual(
            results.tolist(),
            [[0], [2], [2], [0]]
        )

    def test_prediction_classes(self):
        d = decoders.OneHotDecoder(classes=['A', 'B', 'C'])

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        results = d.predict(probas)
        self.assertListEqual(
            results.tolist(),
            [['A'], ['C'], ['C'], ['A']]
        )

    def test_wrong_sized_prediction_classes(self):
        d = decoders.OneHotDecoder(classes=['A', 'B'])

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        with self.assertRaises(ValueError):
            d.predict(probas)


if __name__ == '__main__':
    unittest.main()
