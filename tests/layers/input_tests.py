import unittest
import numpy as np
from mynn.layers.input import Input


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


if __name__ == '__main__':
    unittest.main()
