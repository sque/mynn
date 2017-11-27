import unittest
import numpy as np
from mynn.loss import BinaryCrossEntropyLoss
from mynn._const import SMALL_FLOAT
from ._utils import approximated_derivative


class BinaryCrossEntropyLossTestCase(unittest.TestCase):
    def setUp(self):
        self.l = BinaryCrossEntropyLoss()

    def test_forward(self):
        # Test boundaries
        self.assertAlmostEqual(self.l(0, 0), 0, 5)
        self.assertAlmostEqual(self.l(1, 1), 0, 5)

        # Test a known value
        self.assertAlmostEqual(self.l(0.5, 0.5), 0.693147, 3)

    def test_clip_activations(self):
        # Check that it clips on boundaries but is almost the same
        self.assertNotEqual(0, BinaryCrossEntropyLoss._clip_activations(0))
        self.assertNotEqual(1, BinaryCrossEntropyLoss._clip_activations(1))

        self.assertAlmostEqual(0, BinaryCrossEntropyLoss._clip_activations(0), 10)
        self.assertAlmostEqual(1, BinaryCrossEntropyLoss._clip_activations(1), 10)

        # Check that it actual clips in further distances
        self.assertAlmostEqual(0, BinaryCrossEntropyLoss._clip_activations(-10), 10)
        self.assertAlmostEqual(1, BinaryCrossEntropyLoss._clip_activations(+15), 10)

    def test_derivative(self):
        l = BinaryCrossEntropyLoss()
        # Test boundaries
        self.assertAlmostEqual(l.derivative(0, 0), 1, 5)
        self.assertAlmostEqual(l.derivative(1, 1), -1, 5)

        # Test a known value
        self.assertAlmostEqual(l.derivative(0.5, 0.5), 0, 5)

    @unittest.skip(reason="math should be reconsidered. this function is not used anyway")
    def test_gradient_checking(self):
        points = [SMALL_FLOAT, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1 - SMALL_FLOAT]

        for p in points:
            print(self.l.derivative(p, p), approximated_derivative(self.l, p, p))
            self.assertTrue(np.isclose(self.l.derivative(p, p), approximated_derivative(self.l, p, p)))


if __name__ == '__main__':
    unittest.main()
