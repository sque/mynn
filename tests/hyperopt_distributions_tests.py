import unittest
import numpy as np
from mynn.hyperopt.distributions import uniform, log_uniform, uniform_choice, discrete


class UniformTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_delta(self):
        p = uniform(1, 10)
        self.assertEqual(p.delta, 9)

        p = uniform(-5.5, 10)
        self.assertEqual(15.5, p.delta)

        p = uniform(0.001, 1)
        self.assertEqual(0.999, p.delta)

    def test_sample_positive(self):
        par = uniform(4, 15)
        samples = par.sample(500)
        self.assertAlmostEqual(4, samples.min(), 1)
        self.assertAlmostEqual(15, samples.max(), 1)
        self.assertAlmostEqual(9.5, samples.mean(), 0)

    def test_sample_both_axis(self):
        par = uniform(-4, 15)
        samples = par.sample(1000)
        self.assertAlmostEqual(-4, samples.min(), 1)
        self.assertAlmostEqual(15, samples.max(), 0)
        self.assertAlmostEqual(5.5, samples.mean(), 0)

    def test_sample_one(self):
        par = uniform(-4, 15)
        sample = par.sample()
        self.assertAlmostEqual(3.9234, sample, 3)
        self.assertIsInstance(sample, float)


class LogUniformTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

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

    def test_log_uniform_positive(self):
        par = log_uniform(4, 400)
        samples = par.sample(10000)
        self.assertAlmostEqual(4, samples.min(), 0)
        self.assertAlmostEqual(400, samples.max(), 0)
        self.assertAlmostEqual(85, samples.mean(), 0)

    def test_sample_one(self):
        par = log_uniform(4, 150)
        sample = par.sample()
        self.assertAlmostEqual(18.1328, sample, 3)
        self.assertIsInstance(sample, float)


class UniformChoiceTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_samples_numbers(self):
        p = uniform_choice([5, 6, 7, 8, 9])
        samples = p.sample(10000)
        self.assertAlmostEqual(5, samples.min(), 0)
        self.assertAlmostEqual(9, samples.max(), 0)
        self.assertAlmostEqual(7, samples.mean(), 0)

    def test_samples_strings(self):
        p = uniform_choice(['a', 'b', 'c', 'd'])
        samples = p.sample(10000)
        values, counts = np.unique(samples, return_counts=True)
        self.assertEqual(
            ['a', 'b', 'c', 'd'],
            values.tolist()
        )
        self.assertAlmostEqual(
            0.25,
            counts[0] / 10000,
            1
        )
        self.assertAlmostEqual(
            0.25,
            counts[1] / 10000,
            1
        )
        self.assertAlmostEqual(
            0.25,
            counts[2] / 10000,
            1
        )

    def test_sample_one(self):
        par = uniform_choice(['a', 'b', 'c', 'd'])
        sample = par.sample()
        self.assertEqual('b', sample)
        self.assertIsInstance(sample, str)


class DiscreteTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_samples_positive(self):
        p = discrete(uniform(1, 15))
        samples = p.sample(100)
        self.assertEqual(1, samples.min())
        self.assertEqual(15, samples.max())
        self.assertAlmostEqual(8, samples.mean(), 0)

    def test_samples_all_range(self):
        p = discrete(uniform(-30, 15))
        samples = p.sample(500)
        self.assertEqual(-30, samples.min())
        self.assertEqual(15, samples.max())
        self.assertAlmostEqual(-7.5, samples.mean(), 0)

    def test_sample_one(self):
        par = discrete(uniform(-30, 15))
        sample = par.sample()
        self.assertEqual(-11, sample, 3)
        self.assertIsInstance(sample, np.int64)


if __name__ == '__main__':
    unittest.main()
