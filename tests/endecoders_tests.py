import unittest
import numpy as np

from mynn import endecoders


class BinaryEncoderTestCase(unittest.TestCase):
    def test_default_constructor(self):
        d = endecoders.ThresholdBinaryEncoderDecoder()
        self.assertEqual(d.probability_threshold, 0.5)
        self.assertEqual(d.class_a, 1)
        self.assertEqual(d.class_b, 0)

    def test_override_constructor(self):
        d = endecoders.ThresholdBinaryEncoderDecoder(
            probability_threshold=0.7,
            class_a=15,
            class_b=2
        )
        self.assertEqual(d.probability_threshold, 0.7)
        self.assertEqual(d.class_a, 15)
        self.assertEqual(d.class_b, 2)

    def test_error_constructor(self):
        with self.assertRaises(ValueError):
            endecoders.ThresholdBinaryEncoderDecoder(
                class_a=1,
                class_b=1
            )

    def test_encode_default(self):
        d = endecoders.ThresholdBinaryEncoderDecoder()
        probas = np.array([
            [1, 0, 1, 0, 1]
        ]).T

        out = d.decode(probas)
        self.assertListEqual(
            out.tolist(),
            [[1, 0, 1, 0, 1]]
        )

    def test_decode_simple(self):
        d = endecoders.ThresholdBinaryEncoderDecoder()
        probas = np.array([0, 0.2, 0.4999, 0.5, 0.50000001, 0.6, 1])

        out = d.decode(probas)
        self.assertListEqual(
            out.tolist(),
            [[0, 0, 0, 0, 1, 1, 1]]
        )

    def test_decode_non_default(self):
        d = endecoders.ThresholdBinaryEncoderDecoder(
            probability_threshold=0.7,
            class_a=15,
            class_b=2
        )
        probas = np.array([0, 0.2, 0.4999, 0.5, 0.50000001, 0.6, 1])

        out = d.decode(probas)
        self.assertListEqual(
            out.tolist(),
            [[2, 2, 2, 2, 2, 2, 15]]
        )

    def test_repr(self):
        d = endecoders.ThresholdBinaryEncoderDecoder(
            probability_threshold=0.7,
            class_a=15,
            class_b=2
        )
        self.assertIsInstance(repr(d), str)


class OneHotDecoderTestCase(unittest.TestCase):
    def test_default_constructor(self):
        d = endecoders.OneHotEncoderDecoder(classes=[1, 2, 3])
        self.assertListEqual(d.classes.tolist(), [1, 2, 3])

    def test_error_constructor(self):
        with self.assertRaises(TypeError):
            endecoders.OneHotEncoderDecoder(classes=1)

    def test_decoder_indices(self):
        d = endecoders.OneHotEncoderDecoder(classes=[0, 1, 2])

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        results = d.decode(probas)
        self.assertListEqual(
            results.tolist(),
            [[0, 2, 2, 0]]
        )

    def test_decode_strings(self):
        d = endecoders.OneHotEncoderDecoder(classes=['A', 'B', 'C'])

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        results = d.decode(probas)
        self.assertListEqual(
            results.tolist(),
            [['A', 'C', 'C', 'A']]
        )

    def test_wrong_sized_decode_classes(self):
        d = endecoders.OneHotEncoderDecoder(classes=['A', 'B'])

        probas = np.array([
            [0.8, 0.2, 0.3, 0.333],
            [0.1, 0.3, 0.3, 0.333],
            [0.1, 0.5, 0.4, 0.333]
        ])

        with self.assertRaises(ValueError):
            d.decode(probas)

    def test_encode_extra_classes(self):
        d = endecoders.OneHotEncoderDecoder(classes=['A', 'B'])
        input_y = np.array([
            ['A'],
            ['B'],
            ['C']  # extra class
        ])

        with self.assertRaises(ValueError):
            d.encode(input_y)

    def test_encode(self):
        d = endecoders.OneHotEncoderDecoder(classes=['A', 'B'])
        input_y = np.array(['A', 'B', 'A', 'A']).reshape(1, -1)

        expected_output = np.array([
            [1., 0, 1, 1],
            [0, 1, 0, 0.]
        ])
        classes = d.encode(input_y)
        self.assertListEqual(
            classes.tolist(),
            expected_output.tolist()
        )

    def test_repr(self):
        d = endecoders.OneHotEncoderDecoder(classes=['A', 'B'])
        self.assertIsInstance(repr(d), str)

if __name__ == '__main__':
    unittest.main()
