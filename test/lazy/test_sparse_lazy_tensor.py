import unittest

from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestSparseLazyTensor(LazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        pass

    def evaluate_lazy_tensor(self, lazy_tensor):
        pass


if __name__ == "__main__":
    unittest.main()
