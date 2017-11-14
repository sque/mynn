import unittest
import numpy as np
from mynn.preprocess import StandardScaler

class StandardScalerTestCase(unittest.TestCase):

    def test_default_construction(self):

        ss = StandardScaler()
        self.assertIsNone(ss._mean)
        self.assertIsNone(ss._variance)

    def test_construction_with_arguments(self):

        ss = StandardScaler(mean=15, variance=5.1)
        self.assertEqual(ss._mean, 15)
        self.assertEqual(ss._variance, 5.1)

    def test_train(self):

        ss = StandardScaler()

        data = np.array([
            [1, 2, 1, 2, 1, 2]
        ])

        ss.train(data)
        self.assertEqual(ss._mean, [1.5])
        self.assertEqual(ss._variance, [0.25])


if __name__ == '__main__':
    unittest.main()
