import unittest

import torch

from gpytorch.lazy.sparse_lazy_tensor import SparseLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestSparseLazyTensor(LazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        i = [[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 2, 3]]
        v = [3, 1.5, 1.5, 2, 5, 6]
        return SparseLazyTensor(indices=torch.Tensor(i), values=torch.Tensor(v), sparse_size=torch.Tensor([4, 4]))

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.to_dense()


if __name__ == "__main__":
    unittest.main()
