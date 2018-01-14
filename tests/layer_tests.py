import unittest
import numpy as np
from mynn.loss import BinaryCrossEntropyLoss, CrossEntropyLoss
from mynn import activation
from mynn.layers import ShapeDescription, Input, FullyConnected
from ._utils import approximated_derivative, approximated_derivative_parameters


class ShapeDescriptionTestCase(unittest.TestCase):

    def test_complies_same(self):
        sd = ShapeDescription((None, 2))

        self.assertTrue((10, 2) in sd)
        self.assertTrue((1, 2) in sd)
        self.assertTrue((30, 2) in sd)

    def test_partial_shape_incomplies(self):
        sd = ShapeDescription((None, 2))

        with self.assertRaises(ValueError):
            self.assertTrue((None, 2) in sd)

        with self.assertRaises(ValueError):
            self.assertTrue((None,) in sd)

        with self.assertRaises(ValueError):
            self.assertTrue((None, None) in sd)

    def test_different_shape(self):
        sd = ShapeDescription((None, 2))

        self.assertFalse((2,) in sd)
        self.assertFalse((1, 1, 2) in sd)
        self.assertFalse(tuple() in sd)

    def test_equality(self):
        sd = ShapeDescription((None, 2))

        sd2 = ShapeDescription((None, 2))

        sd3 = ShapeDescription((4, 2))

        self.assertEqual(sd, sd2)
        self.assertEqual(sd, (None, 2))
        self.assertNotEqual(sd, sd3)
        self.assertNotEqual(sd2, sd3)


class InputTestCase(unittest.TestCase):

    def test_shapes(self):
        i = Input((None, 10))
        self.assertEqual((None, 10), i.output_shape)
        self.assertEqual((None, 10), i.input_shape)

    def test_forward(self):
        i = Input((None, 10))
        x = np.random.rand(10, 10)
        self.assertTrue(
            np.all(x == i.forward(x))
        )

    def test_shape_validation(self):
        i = Input((None, 2))

        with self.assertRaises(TypeError):
            x = np.random.rand(10, 10)
            i.forward(x)

        with self.assertRaises(TypeError):
            x = np.random.rand(2, 10)
            i.forward(x)


class FullyConnectedTestCase(unittest.TestCase):

    def test_parameter_info(self):
        i = Input((2, None))
        fc = FullyConnected(10, activation=activation.ReLUActivation)(i)

        self.assertEqual(fc.parameters_info['W'].shape, (10, 2))
        self.assertEqual(fc.parameters_info['b'].shape, (10, 1))

    def test_parameter_set(self):
        i = Input((2, None))
        fc = FullyConnected(10, activation=activation.ReLUActivation)(i)

        w = np.ones((10, 2))
        b = np.ones((10, 1)) * 2
        fc.parameters = {'W': w, 'b': b}

        self.assertTrue(np.all(w == fc.parameters['W']))
        self.assertTrue(np.all(b == fc.parameters['b']))

    def test_input_output_shape(self):
        i = Input((2, None))
        fc = FullyConnected(10, activation=activation.ReLUActivation)(i)

        self.assertEqual(fc.input_shape, (2, None))
        self.assertEqual(fc.output_shape, (10, None))

    def test_forward(self):
        i = Input((2, None))
        fc = FullyConnected(10, activation=activation.ReLUActivation)(i)
        fc.parameters = {'W': np.ones((10, 2)), 'b': np.ones((10, 1)) * 2}

        x = np.random.rand(2, 5)
        y_expected = np.dot(np.ones((10, 2)), x) + (np.ones((10, 1)) * 2)
        y = fc.forward(x)

        self.assertTrue(np.all(y_expected == y))

        # Test negative input
        x = np.random.rand(2, 5) * -3
        y_expected = np.maximum(0, np.dot(np.ones((10, 2)), x) + (np.ones((10, 1)) * 2))
        y = fc.forward(x)

        self.assertTrue(np.all(y_expected == y))

        # Test cache
        z_expected = np.dot(np.ones((10, 2)), x) + (np.ones((10, 1)) * 2)
        self.assertTrue(np.all(z_expected == fc._cache.Z))

        self.assertTrue(np.all(x == fc._cache.In))
        self.assertTrue(np.all(y_expected == fc._cache.Out))

    def test_forward_without_activation(self):
        i = Input((2, None))
        fc = FullyConnected(10, activation=None)(i)
        fc.parameters = {'W': np.ones((10, 2)), 'b': np.ones((10, 1)) * 2}

        x = np.random.rand(2, 5) * -3
        y_expected = np.dot(np.ones((10, 2)), x) + (np.ones((10, 1)) * 2)
        y = fc.forward(x)

        self.assertTrue(np.all(y_expected == y))

        # Test cache
        self.assertTrue(np.all(y_expected == fc._cache.Z))

        self.assertTrue(np.all(x == fc._cache.In))
        self.assertTrue(np.all(y_expected == fc._cache.Out))

    def test_backwards_grad_check_with_loss(self):
        loss = BinaryCrossEntropyLoss()
        i = Input((2, None))
        fc = FullyConnected(1, activation=activation.SigmoidActivation)(i)

        np.random.seed(1)
        w = np.random.randn(*(1, 2))
        b = np.zeros((1, 1)) * 2
        fc.parameters = {'W': w.copy(), 'b': b.copy()}

        x = np.random.rand(2, 15)
        y_expected = np.random.rand(1, 15)
        y = fc.forward(x)

        (dx,), (dW, db) = fc.backward(loss.derivative(y, y_expected))

        def execute_network(x, W, b):
            i = Input((2, None))
            fc = FullyConnected(1, activation=activation.SigmoidActivation)(i)

            fc.parameters = {'W': W, 'b': b}
            return loss(fc.forward(x), y_expected)

        dx_aprox, dW_aprox, db_aprox = approximated_derivative_parameters(execute_network, x, w, b, e=0.1e-5)

        # print(dx.shape)
        # print(dx)
        # print(dx_aprox)
        # print(dx/dx_aprox)

        # print(dW, dW_aprox)
        # print(dW - dW_aprox)
        # print(db - db_aprox)
        self.assertTrue(np.all(dW - dW_aprox < 0.1e-7))
        self.assertTrue(np.all(db - db_aprox < 0.1e-7))
        # self.assertTrue(np.all(dx - dx_aprox < 0.1e-7))

    def test_approx_gradient(self):
        def eq(x1, x2, x3):
            return x3 * (x1 ** 2 + np.exp(x2))

        def diff_eq(x1, x2, x3):
            dx1 = 2 * x1 * x3
            dx2 = x3 * np.exp(x2)
            dx3 = x1 ** 2 + np.exp(x2)
            return dx1, dx2, dx3

        x1 = np.array([35.3])
        x2 = np.array([1.5])
        x3 = np.array([-0.3])

        dx1, dx2, dx3 = diff_eq(x1, x2, x3)
        dx1_aprox, dx2_aprox, dx3_aprox = approximated_derivative_parameters(eq, x1, x2, x3, e=.1e-3)

        self.assertTrue(np.all((dx1 - dx1_aprox) < .1e-07))
        self.assertTrue(np.all((dx2 - dx2_aprox) < .1e-07))
        self.assertTrue(np.all((dx3 - dx3_aprox) < .1e-07))

    def test_backwards_multi_layer(self):
        loss = BinaryCrossEntropyLoss()
        i = Input((2, None))
        fc1 = FullyConnected(5, activation=activation.SigmoidActivation, name='fc1')(i)
        fc2 = FullyConnected(1, activation=activation.SigmoidActivation, name='fc2')(fc1)

        np.random.seed(1)
        w1 = np.random.randn(*fc1.parameters_info['W'].shape)
        b1 = np.zeros((fc1.parameters_info['b'].shape)) * 2
        fc1.parameters = {'W': w1.copy(), 'b': b1.copy()}

        w2 = np.random.randn(*fc2.parameters_info['W'].shape)
        b2 = np.zeros((fc2.parameters_info['b'].shape)) * 2
        fc2.parameters = {'W': w2.copy(), 'b': b2.copy()}

        x = np.random.rand(2, 6)
        y_expected = np.random.rand(1, 6)
        y = fc2.forward(fc1.forward(x))

        (din2,), (dW2, db2) = fc2.backward(loss.derivative(y, y_expected))
        (din1,), (dW1, db1) = fc1.backward(din2)

        def execute_network(x, w1, b1, w2, b2):
            i = Input((2, None))
            fc1 = FullyConnected(5, activation=activation.SigmoidActivation)(i)
            fc2 = FullyConnected(1, activation=activation.SigmoidActivation)(fc1)

            fc1.parameters = {'W': w1.copy(), 'b': b1.copy()}

            fc2.parameters = {'W': w2.copy(), 'b': b2.copy()}
            return loss(fc2.forward(fc1.forward(x)), y_expected)

        dX_approx, dW1_aprox, db1_aprox, dw2_aprox, db2_aprox \
            = approximated_derivative_parameters(execute_network, x, w1, b1, w2, b2, e=.1e-3)

        # print('Original dW1:', dW1)
        # print('Approx dW1  :', dW1_aprox)
        # print(dW1/dW1_aprox)
        #
        # print('Original db1:', db1.shape, '\n', db1)
        # print('Approx db1  :',db1_aprox.shape, '\n', db1_aprox)

        # print(din1 / dX_approx)
        self.assertTrue(np.all(dW1 - dW1_aprox < 0.1e-7))
        self.assertTrue(np.all(db1 - db1_aprox < 0.1e-7))
        self.assertTrue(np.all(dW2 - dw2_aprox < 0.1e-7))
        self.assertTrue(np.all(db2 - db2_aprox < 0.1e-7))
        # self.assertTrue(np.all(din1 - dX_approx < 0.1e-7))


if __name__ == '__main__':
    unittest.main()
