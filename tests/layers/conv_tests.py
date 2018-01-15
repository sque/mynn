import unittest
from mynn.layers.input import Input
from mynn.layers.conv import Conv2D
from mynn.activation import ReLUActivation

import numpy as np


class Conv2DTestCase(unittest.TestCase):

    def test_parameter_info(self):
        i = Input((13, 14, 3, None))
        c = Conv2D(10, (3, 3), activation=ReLUActivation)(i)

        self.assertEqual(c.parameters_info['KW'].shape, (3, 3, 3, 10))
        self.assertTrue(c.parameters_info['KW'].init_random)
        self.assertEqual(c.parameters_info['Kb'].shape, (1, 1, 1, 10))
        self.assertFalse(c.parameters_info['Kb'].init_random)

    def test_parameter_set(self):
        i = Input((16, 16, 1, None))
        c = Conv2D(10, (3, 3), stride=2, activation=ReLUActivation)(i)

        np.random.seed(0)
        w = np.random.rand(3, 3, 1, 10)
        b = np.random.rand(1, 1, 1, 10)
        c.parameters = {'KW': w, 'Kb': b}

        self.assertTrue(np.all(w == c.parameters['KW']))
        self.assertTrue(np.all(b == c.parameters['Kb']))

        # Set wrong shape
        with self.assertRaises(TypeError):
            c.parameters = {'KW': np.random.rand(10, 2, 2, 1)}

    def test_input_output_shape_1(self):
        i = Input((13, 13, 3, None))
        c = Conv2D(10, (3, 3), stride=2, activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 13, 3, None))
        self.assertEqual(c.output_shape, (7, 7, 10, None))

    def test_input_output_shape_2(self):
        i = Input((13, 14, 3, None))
        c = Conv2D(10, (3, 3), stride=3, activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 14, 3, None))
        self.assertEqual(c.output_shape, (5, 5, 10, None))

    def test_input_output_shape_same_padding_1(self):
        i = Input((13, 13, 3, None))
        c = Conv2D(10, (3, 3), stride=1, padding='same', activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 13, 3, None))
        self.assertEqual(c.output_shape, (13, 13, 10, None))

    def test_input_output_shape_same_padding_2(self):
        i = Input((13, 13, 3, None))
        c = Conv2D(10, (5, 5), stride=1, padding='same', activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 13, 3, None))
        self.assertEqual(c.output_shape, (13, 13, 10, None))

    def test_input_output_shape_same_padding_3(self):
        i = Input((13, 14, 3, None))
        c = Conv2D(10, (5, 5), stride=2, padding='same', activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 14, 3, None))
        self.assertEqual(c.output_shape, (7, 7, 10, None))

    def test_input_output_shape_same_padding_4(self):
        i = Input((13, 14, 3, None))
        c = Conv2D(10, (3, 3), stride=2, padding='same', activation=ReLUActivation)(i)

        self.assertEqual(c.input_shape, (13, 14, 3, None))
        self.assertEqual(c.output_shape, (7, 7, 10, None))

    def test_padding_type(self):
        # default padding
        c = Conv2D(3, (3, 3))
        self.assertEqual('same', c.padding_type)

        # correct paddings
        c = Conv2D(3, (3, 3), padding='valid')
        self.assertEqual('valid', c.padding_type)

        # correct paddings
        c = Conv2D(3, (3, 3), padding='same')
        self.assertEqual('same', c.padding_type)

        # incorrect paddings
        with self.assertRaises(ValueError):
            Conv2D(3, (3, 3), padding='broken')

    def test_forward_3_3_without_activation(self):
        i = Input((4, 4, 2, None))
        c = Conv2D(3, (3, 3), padding='valid')(i)
        c_padded = Conv2D(3, (3, 3), padding='same')(i)

        f1 = np.array([
            [3, 0, -3],
            [2, 0, -2],
            [3, 0, -3]
        ])
        f2 = f1.T
        f3 = f2 + f1
        f1 = f1.reshape(3, 3, 1, 1)
        f2 = f2.reshape(3, 3, 1, 1)
        f3 = f3.reshape(3, 3, 1, 1)

        KW = np.concatenate((f1, f2, f3), axis=3)
        KW = np.concatenate((KW, KW * 2), axis=2)

        input_image_C1 = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]).reshape(4, 4, 1)
        input_image_C2 = np.array([
            [0.5, 0.2, 0.2, 0.2],
            [0.2, 0.5, 0.2, 0.2],
            [0.2, 0.2, 0.5, 0.2],
            [0.2, 0.2, 0.2, 0.5],
        ]).reshape(4, 4, 1)
        input_image = np.concatenate((input_image_C1, input_image_C2), axis=2).reshape(4, 4, 2, 1)

        # print(input_image[1:4, 1:4, 0, 0])
        # print(KW[:, :, 0, 0])
        # print(input_image[1:4, 1:4, 0, 0] * KW[:, :, 0, 0])
        # print(np.sum(input_image[1:4, 1:4, 1, 0] * KW[:, :, 1, 0]))
        c.parameters = {'KW': KW, 'Kb': np.zeros((1, 1, 1, 3))}
        c_padded.parameters = {'KW': KW.copy(), 'Kb': np.zeros((1, 1, 1, 3))}

        # TEST VALID
        convoluted = c.forward(input_image)
        self.assertTupleEqual(convoluted.shape, (2, 2, 3, 1))

        # Test some samples on the convoluted picture that works as it should
        self.assertEqual(convoluted[0, 0, 0, 0], -16)
        self.assertEqual(convoluted[0, 1, 0, 0], -14.8)
        self.assertEqual(convoluted[1, 1, 0, 0], -16)

        # TEST PADDED
        convoluted_padded = c_padded.forward(input_image)
        self.assertTupleEqual(convoluted_padded.shape, (4, 4, 3, 1))
        print(convoluted_padded[3, 3, 0, 0])
        self.assertEqual(convoluted_padded[0, 0, 0, 0], -25.8)
        self.assertEqual(convoluted_padded[3, 3, 0, 0], 66.8)
        self.assertEqual(convoluted_padded[3, 3, 1, 0], 60.8)
        self.assertEqual(convoluted_padded[3, 3, 2, 0], 127.6)

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
