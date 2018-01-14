import numpy as np
import unittest

from mynn.initializers import ConstantWeightInitializer, NormalWeightInitializer, XavierWeightInitializer
from mynn.layers import FullyConnected, Input


class ConstantWeightInitializerTestCase(unittest.TestCase):

    def setUp(self):
        i = Input((2, None))
        self.fc = FullyConnected(3, name='fc')(i)
        self.fc2 = FullyConnected(4, name='fc2')(self.fc)

    def test_broadcasting(self):
        i = ConstantWeightInitializer({
            'fc': {'W': 0.5, 'b': 15}})

        i(self.fc)

        self.assertTrue(np.all(
            self.fc.parameters['b'] == np.ones((3, 1)) * 15
        ))

        self.assertTrue(np.all(
            self.fc.parameters['W'] == np.ones((3, 2)) * 0.5
        ))

    def test_exact(self):
        W1 = np.array([[1, 2], [2, 3], [4, 5]])
        W2 = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]])

        i = ConstantWeightInitializer({
            'fc': {'W': W1, 'b': 15},
            'fc2': {'W': W2, 'b': 0.5}})

        i(self.fc)
        i(self.fc2)

        self.assertTrue(np.all(
            self.fc.parameters['b'] == np.ones((3, 1)) * 15
        ))

        self.assertTrue(np.all(
            self.fc.parameters['W'] == W1
        ))

        self.assertTrue(np.all(
            self.fc2.parameters['b'] == np.ones((4, 1)) * 0.5
        ))

        self.assertTrue(np.all(
            self.fc2.parameters['W'] == W2
        ))

    def test_array_wrong_shape(self):
        W = np.array([[1, 2], [2, 3], [4, 5]])

        i = ConstantWeightInitializer({

            'fc2': {'W': W, 'b': 0.5}})

        with self.assertRaises(TypeError):
            i(self.fc2)


class NormalWeightInitializerTestCase(unittest.TestCase):

    def setUp(self):
        i = Input((2, None))
        self.fc = FullyConnected(3, name='fc')(i)
        self.fc2 = FullyConnected(4, name='fc2')(self.fc)

    def test_initialize(self):
        np.random.seed(1)
        i = NormalWeightInitializer(0.5)

        i(self.fc)

        # Check not random equal to zero
        self.assertTrue(np.all(np.zeros((3, 1)) == self.fc.parameters['b']))

        expected_W = np.array([[2.08511002e-01, 3.60162247e-01],
                               [5.71874087e-05, 1.51166286e-01],
                               [7.33779454e-02, 4.61692974e-02]])

        self.assertTrue(np.allclose(
            expected_W,
            self.fc.parameters['W']))

    def test_initialize_with_scale(self):
        np.random.seed(1)
        i = NormalWeightInitializer(100)

        i(self.fc)

        # Check not random equal to zero
        self.assertTrue(np.all(np.zeros((3, 1)) == self.fc.parameters['b']))

        expected_W = np.array([[4.17022005e+01, 7.20324493e+01],
                               [1.14374817e-02, 3.02332573e+01],
                               [1.46755891e+01, 9.23385948e+00]])

        self.assertTrue(np.allclose(
            expected_W,
            self.fc.parameters['W']))




class XavierWeightInitializerTestCase(unittest.TestCase):

    def setUp(self):
        self.normal_random = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                                       [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                                       [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                                       [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521]])

        i = Input((2, None))
        self.fc = FullyConnected(3, name='fc')(i)
        self.fc2 = FullyConnected(4, name='fc2')(self.fc)

    def test_get_initial_weights_with_scale2(self):
        np.random.seed(1)
        i = XavierWeightInitializer(scale=2)

        i(self.fc)

        # Check not random equal to zero
        self.assertTrue(np.all(np.zeros((3, 1)) == self.fc.parameters['b']))

        expected_W = (self.normal_random.ravel() * np.sqrt(2 / 5))[:6].reshape(3, 2)
        self.assertTrue(np.allclose(self.fc.parameters['W'], expected_W))

    def test_get_initial_weights_with_scale3(self):
        np.random.seed(1)
        i = XavierWeightInitializer(scale=3)

        i(self.fc)

        # Check not random equal to zero
        self.assertTrue(np.all(np.zeros((3, 1)) == self.fc.parameters['b']))

        expected_W = (self.normal_random.ravel() * np.sqrt(3 / 5))[:6].reshape(3, 2)
        self.assertTrue(np.allclose(self.fc.parameters['W'], expected_W))


if __name__ == '__main__':
    unittest.main()
