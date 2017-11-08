import unittest
from mynn.activation import TanhActivation, SigmoidActivation, ReLUActivation


class SigmoidTestCase(unittest.TestCase):

    def test_forward(self):
        s = SigmoidActivation()

        self.assertAlmostEqual(s(-1e+10), 0.0, 2)
        self.assertAlmostEqual(s(-1), 0.26894, 3)
        self.assertEqual(s(0), 0.5)
        self.assertAlmostEqual(s(1), 1 - 0.26894, 3)
        self.assertAlmostEqual(s(1e+10), 1.000, 2)

    def test_derivative(self):

        s = SigmoidActivation()
        self.assertAlmostEqual(s.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(s.derivative(+1e+10), 0.0, 2)
        self.assertAlmostEqual(s.derivative(-2), 0.1049, 2)
        self.assertAlmostEqual(s.derivative(0), 0.25)
        self.assertAlmostEqual(s.derivative(2), 0.1049, 2)


class TanhTestCase(unittest.TestCase):

    def test_forward(self):
        t = TanhActivation()

        self.assertAlmostEqual(t(-1e+10), -1.0, 2)
        self.assertAlmostEqual(t(-1), -0.7615, 3)
        self.assertEqual(t(0), 0)
        self.assertAlmostEqual(t(1), 0.7615, 3)
        self.assertAlmostEqual(t(1e+10), 1.0, 2)

    def test_derivative(self):

        t = TanhActivation()
        self.assertAlmostEqual(t.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(t.derivative(-2), 0.0706, 2)
        self.assertAlmostEqual(t.derivative(-1), 0.41997, 2)
        self.assertAlmostEqual(t.derivative(0), 1)
        self.assertAlmostEqual(t.derivative(1), 0.41997, 2)
        self.assertAlmostEqual(t.derivative(2), 0.0706, 2)
        self.assertAlmostEqual(t.derivative(+1e+10), 0.0, 2)


class ReluTestCase(unittest.TestCase):

    def test_forward(self):
        rl = ReLUActivation()

        self.assertAlmostEqual(rl(-1e+10), 0.0, 2)
        self.assertAlmostEqual(rl(-1), 0.0, 3)
        self.assertEqual(rl(0), 0)
        self.assertAlmostEqual(rl(1), 1, 3)
        self.assertAlmostEqual(rl(1e+10), 1e+10, 2)

    def test_derivative(self):

        t = ReLUActivation()
        self.assertAlmostEqual(t.derivative(-1e+10), 0.0, 2)
        self.assertAlmostEqual(t.derivative(-2), 0.0, 2)
        self.assertAlmostEqual(t.derivative(-1), 0.0, 2)
        self.assertAlmostEqual(t.derivative(0), 0)
        self.assertAlmostEqual(t.derivative(1), 1, 2)
        self.assertAlmostEqual(t.derivative(2), 1, 2)
        self.assertAlmostEqual(t.derivative(+1e+10), 1, 2)


if __name__ == '__main__':
    unittest.main()
