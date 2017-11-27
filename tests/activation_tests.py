import unittest
import numpy as np

from mynn.activation import TanhActivation, SigmoidActivation, ReLUActivation, SoftmaxActivation

from tests._utils import approximated_derivative


class SigmoidTestCase(unittest.TestCase):
    def setUp(self):
        self.s = SigmoidActivation()

    def test_forward(self):
        self.assertAlmostEqual(self.s(-1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s(-1), 0.26894, 3)
        self.assertEqual(self.s(0), 0.5)
        self.assertAlmostEqual(self.s(1), 1 - 0.26894, 3)
        self.assertAlmostEqual(self.s(1e+10), 1.000, 2)

    def test_derivative(self):
        self.assertAlmostEqual(self.s.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(+1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(-2), 0.1049, 2)
        self.assertAlmostEqual(self.s.derivative(0), 0.25)
        self.assertAlmostEqual(self.s.derivative(2), 0.1049, 2)

    def test_gradient_checking(self):
        points = [-1e+10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, +1e+10]

        for p in points:
            self.assertTrue(np.isclose(self.s.derivative(p), approximated_derivative(self.s, p)))


class TanhTestCase(unittest.TestCase):
    def setUp(self):
        self.s = TanhActivation()

    def test_forward(self):
        self.assertAlmostEqual(self.s(-1e+10), -1.0, 2)
        self.assertAlmostEqual(self.s(-1), -0.7615, 3)
        self.assertEqual(self.s(0), 0)
        self.assertAlmostEqual(self.s(1), 0.7615, 3)
        self.assertAlmostEqual(self.s(1e+10), 1.0, 2)

    def test_derivative(self):
        self.assertAlmostEqual(self.s.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(-2), 0.0706, 2)
        self.assertAlmostEqual(self.s.derivative(-1), 0.41997, 2)
        self.assertAlmostEqual(self.s.derivative(0), 1)
        self.assertAlmostEqual(self.s.derivative(1), 0.41997, 2)
        self.assertAlmostEqual(self.s.derivative(2), 0.0706, 2)
        self.assertAlmostEqual(self.s.derivative(+1e+10), 0.0, 2)

    def test_gradient_checking(self):
        points = [-1e+10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, +1e+10]

        for p in points:
            self.assertTrue(np.isclose(self.s.derivative(p), approximated_derivative(self.s, p)))


class ReluTestCase(unittest.TestCase):
    def setUp(self):
        self.s = ReLUActivation()

    def test_forward(self):
        self.assertAlmostEqual(self.s(-1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s(-1), 0.0, 3)
        self.assertEqual(self.s(0), 0)
        self.assertAlmostEqual(self.s(1), 1, 3)
        self.assertAlmostEqual(self.s(1e+10), 1e+10, 2)

    def test_derivative(self):
        self.assertAlmostEqual(self.s.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(-2), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(-1), 0.0, 2)
        self.assertAlmostEqual(self.s.derivative(0), 0)
        self.assertAlmostEqual(self.s.derivative(1), 1, 2)
        self.assertAlmostEqual(self.s.derivative(2), 1, 2)
        self.assertAlmostEqual(self.s.derivative(+1e+10), 1, 2)

    def test_gradient_checking(self):
        points = [-1e+10, -5, -4, -3, -2, -1, 0.1, 1, 2, 3, 4, 5, +1e+10]
        epsilons = [0.1e-8] * (len(points) - 1) + [1]

        for p, e in zip(points, epsilons):
            self.assertTrue(np.isclose(self.s.derivative(p), approximated_derivative(self.s, p, e=e)))


class SoftmaxTestCase(unittest.TestCase):
    def setUp(self):
        self.s = SoftmaxActivation()
        self.examples_in_ = np.array([
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],

        ]).T

        self.examples_out_ = np.array([
            [0.02364, 0.06426, 0.17468, 0.47483, 0.02364, 0.06426, 0.17468]
        ]
        ).T

    def test_forward(self):
        self.assertTrue(np.all(np.isclose(self.s(self.examples_in_), self.examples_out_, rtol=1.e-04)))


if __name__ == '__main__':
    unittest.main()
