import unittest

from mynn.loss import CrossEntropyLoss



class CrossEntropyLossTestCase(unittest.TestCase):

    def test_forward(self):
        l = CrossEntropyLoss()

        # Test boundaries
        self.assertAlmostEqual(l(0, 0), 0, 5)
        self.assertAlmostEqual(l(1, 1), 0, 5)

        # Test a known value
        self.assertAlmostEqual(l(0.5, 0.5), 0.693147, 3)


    def test_clip_activations(self):

        # Check that it clips on boundaries but is almost the same
        self.assertNotEqual(0, CrossEntropyLoss._clip_activations(0))
        self.assertNotEqual(1, CrossEntropyLoss._clip_activations(1))

        self.assertAlmostEqual(0, CrossEntropyLoss._clip_activations(0), 10)
        self.assertAlmostEqual(1, CrossEntropyLoss._clip_activations(1), 10)

        # Check that it actual clips in further distances
        self.assertAlmostEqual(0, CrossEntropyLoss._clip_activations(-10), 10)
        self.assertAlmostEqual(1, CrossEntropyLoss._clip_activations(+15), 10)

    def test_derivative(self):
        l = CrossEntropyLoss()
        # Test boundaries
        self.assertAlmostEqual(l.derivative(0, 0), 1, 5)
        self.assertAlmostEqual(l.derivative(1, 1), -1, 5)

        # Test a known value
        self.assertAlmostEqual(l.derivative(0.5, 0.5), 0, 5)




if __name__ == '__main__':
    unittest.main()
