import unittest
import numpy as np
from mynn.hyperopt import uniform, log_uniform, uniform_choice


class DistributionsTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_uniform_delta(self):

        p = uniform(1, 10)
        self.assertEqual(p.delta, 9)

        p = uniform(-5.5, 10)
        self.assertEqual(15.5, p.delta)

        p = uniform(0.001, 1)
        self.assertEqual(0.999, p.delta)

    def test_log_uniform_delta(self):
        p = log_uniform(1, 10)
        self.assertEqual(9, p.delta)
        self.assertEqual(1, p.delta_log)

        p = log_uniform(1.5, 32)
        self.assertEqual(30.5, p.delta)
        self.assertAlmostEqual(1.3291, p.delta_log, 3)

        p = log_uniform(0.001, 1)
        self.assertEqual(0.999, p.delta)
        self.assertEqual(3, p.delta_log)

    def test_uniform_positive(self):
        par = uniform(4, 15)
        samples = par.sample(500)
        self.assertAlmostEqual(4, samples.min(), 1)
        self.assertAlmostEqual(15, samples.max(), 1)
        self.assertAlmostEqual(9.5, samples.mean(), 0)

    def test_uniform_both_axis(self):
        par = uniform(-4, 15)
        samples = par.sample(1000)
        self.assertAlmostEqual(-4, samples.min(), 1)
        self.assertAlmostEqual(15, samples.max(), 0)
        self.assertAlmostEqual(9.5, samples.mean(), 0)

    def test_log_uniform_positive(self):
        par = log_uniform(4, 400)
        samples = par.sample(10000)
        self.assertAlmostEqual(4, samples.min(), 0)
        self.assertAlmostEqual(400, samples.max(), 0)
        self.assertAlmostEqual(85, samples.mean(), 0)

    def test_2(self):
        pass
        # print(np.random.rand(1))


if __name__ == '__main__':
    unittest.main()
