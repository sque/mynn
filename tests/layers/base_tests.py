import unittest

from mynn.layers.base import ShapeDescription


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
