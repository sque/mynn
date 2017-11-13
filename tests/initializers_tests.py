import numpy as np
import unittest

from mynn.initializers import ConstantWeightInitializer, NormalWeightInitializer, VarianceScalingWeightInitializer


class ConstantWeightInitializerTestCase(unittest.TestCase):

    def test_out_of_index(self):
        i = ConstantWeightInitializer([1, 2, 3])
        with self.assertRaises(IndexError):
            i.get_initial_weight(4, 4, 5)

    def test_broadcasting(self):
        i = ConstantWeightInitializer([1, 2, 3])

        W, b = i.get_initial_weight(1, 4, 5)

        self.assertTrue(np.all(
            W == np.ones((4,5))
        ))

        self.assertTrue(np.all(
            b == np.zeros((4, 1))
        ))

    def test_array_assignment(self):
        W1 = np.array([[1, 2, 3], [3, 4, 5]])
        W2 = np.array([[7, 5, 4], [0., 2, 1.], [5, 2, 5]])
        i = ConstantWeightInitializer([W1, W2])

        W, b = i.get_initial_weight(1, 2, 3)
        self.assertTrue(np.all(W == W1))
        self.assertTrue(np.all(b == np.zeros((2, 1))))

        W, b = i.get_initial_weight(2, 3, 3)
        self.assertTrue(np.all(W == W2))
        self.assertTrue(np.all(b == np.zeros((3, 1))))

    def test_array_wrong_shape(self):
        W1 = np.array([[1, 2, 3], [3, 4, 5]])
        W2 = np.array([[7, 5, 4], [0., 2, 1.], [5, 2, 5]])
        i = ConstantWeightInitializer([W1, W2])

        with self.assertRaises(ValueError):
            i.get_initial_weight(1, 10, 3)

        with self.assertRaises(ValueError):
            i.get_initial_weight(1, 2, 10)


class NormalWeightInitializerTestCase(unittest.TestCase):

    def test_get_initial_weights(self):
        np.random.seed(1)
        i = NormalWeightInitializer()

        W, b = i.get_initial_weight(1, 4, 5)
        self.assertEqual(W.shape, (4, 5))
        random_normal = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                               [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                               [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                               [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521]])

        self.assertTrue(np.allclose(W, expected_w))

        self.assertTrue(np.all(
            b == np.zeros((4, 1))
        ))

class VarianceScalingWeightInitializerTestCase(unittest.TestCase):

    def setUp(self):
        self.normal_random = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                                       [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                                       [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                                       [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521]])

    def test_get_initial_weights_with_scale2(self):
        np.random.seed(1)
        i = VarianceScalingWeightInitializer(scale=2)

        W, b = i.get_initial_weight(1, 4, 5)
        self.assertEqual(W.shape, (4, 5))

        self.assertTrue(np.allclose(W, self.normal_random * np.sqrt(2 / 5)))

        self.assertTrue(np.all(
            b == np.zeros((4, 1))
        ))

    def test_get_initial_weights_with_scale3(self):
        np.random.seed(1)
        i = VarianceScalingWeightInitializer(scale=3)

        W, b = i.get_initial_weight(1, 4, 5)
        self.assertEqual(W.shape, (4, 5))

        self.assertTrue(np.allclose(W, self.normal_random * np.sqrt(3 / 5)))

        self.assertTrue(np.all(
            b == np.zeros((4, 1))
        ))


if __name__ == '__main__':
    unittest.main()
