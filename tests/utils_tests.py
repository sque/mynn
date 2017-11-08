import unittest
from mynn._utils import grouped, nested_chain_iterable


class UtilsTestCase(unittest.TestCase):

    def test_grouped_simple_usage(self):
        s = [1, 2, 3, 4, 5, 6]

        self.assertListEqual(list(grouped(s, 2)), [(1,2), (3, 4), (5,6)])

        self.assertListEqual(list(grouped(s, 3)), [(1, 2, 3), (4, 5, 6)])

    def test_grouped_empty(self):

        self.assertListEqual(list(grouped([], 2)), [])

    def test_grouped_with_remainder(self):
        s = [1, 2, 3, 4, 5]
        self.assertListEqual(list(grouped(s, 2)), [(1,2), (3, 4), (5, None)])

    def test_nested_chain_simple_usage(self):
        s = [[[1], [2, 3]], [[4], [5, 6]]]

        self.assertListEqual(list(nested_chain_iterable(s, 1)),
                             [[1], [2, 3], [4], [5, 6]])

        self.assertListEqual(list(nested_chain_iterable(s, 2)),
                             [1, 2, 3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
