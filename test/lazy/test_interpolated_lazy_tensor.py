from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import unittest
import os
import random
from gpytorch.lazy import NonLazyTensor, InterpolatedLazyTensor
from gpytorch.utils import approx_equal


class TestInterpolatedLazyTensor(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(3, 1)
        left_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(3, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(3, 1)
        right_interp_values = torch.tensor([[1, 2], [2, 0.5], [1, 3]], dtype=torch.float).repeat(3, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_tensor_mat = torch.randn(6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.t().matmul(base_lazy_tensor_mat)
        base_tensor = base_lazy_tensor_mat
        base_tensor.requires_grad = True
        base_tensor_copy = base_lazy_tensor_mat
        base_lazy_tensor = NonLazyTensor(base_tensor)

        test_matrix = torch.randn(9, 4)

        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_tensor.matmul(test_matrix)

        left_matrix = torch.zeros(9, 6)
        right_matrix = torch.zeros(9, 6)
        left_matrix.scatter_(1, left_interp_indices, left_interp_values_copy)
        right_matrix.scatter_(1, right_interp_indices, right_interp_values_copy)

        actual = left_matrix.matmul(base_tensor_copy).matmul(right_matrix.t()).matmul(test_matrix)
        self.assertTrue(approx_equal(res, actual))

        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_tensor.grad, base_tensor_copy.grad))
        self.assertTrue(approx_equal(left_interp_values.grad, left_interp_values_copy.grad))

    def test_batch_matmul(self):
        left_interp_indices = torch.tensor([[2, 3], [3, 4], [4, 5]], dtype=torch.long).repeat(5, 3, 1)
        left_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 3, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True
        right_interp_indices = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).repeat(5, 3, 1)
        right_interp_values = torch.tensor([[1, 2], [2, 0.5], [1, 3]], dtype=torch.float).repeat(5, 3, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(-1, -2).matmul(base_lazy_tensor_mat)
        base_tensor = base_lazy_tensor_mat
        base_tensor_copy = base_tensor.clone()
        base_tensor.requires_grad = True
        base_tensor_copy.requires_grad = True
        base_lazy_tensor = NonLazyTensor(base_tensor)

        test_matrix = torch.randn(5, 9, 4)

        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_tensor.matmul(test_matrix)

        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(9, 6)
            right_matrix_comp = torch.zeros(9, 6)
            left_matrix_comp.scatter_(1, left_interp_indices[i], left_interp_values_copy[i])
            right_matrix_comp.scatter_(1, right_interp_indices[i], right_interp_values_copy[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)

        actual = left_matrix.matmul(base_tensor_copy).matmul(right_matrix.transpose(-1, -2))
        actual = actual.matmul(test_matrix)
        self.assertTrue(approx_equal(res, actual))

        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_tensor.grad, base_tensor_copy.grad))
        self.assertTrue(approx_equal(left_interp_values.grad, left_interp_values_copy.grad))

    def test_inv_matmul(self):
        base_lazy_tensor_mat = torch.randn(6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.t().matmul(base_lazy_tensor_mat)
        test_matrix = torch.randn(3, 4)

        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True

        right_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        right_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_tensor = base_lazy_tensor_mat
        base_lazy_tensor.requires_grad = True
        base_lazy_tensor_copy = base_lazy_tensor_mat
        test_matrix_tensor = test_matrix
        test_matrix_tensor.requires_grad = True
        test_matrix_tensor_copy = test_matrix

        interp_lazy_tensor = InterpolatedLazyTensor(
            NonLazyTensor(base_lazy_tensor),
            left_interp_indices,
            left_interp_values,
            right_interp_indices,
            right_interp_values,
        )
        res = interp_lazy_tensor.inv_matmul(test_matrix_tensor)

        left_matrix = torch.zeros(3, 6)
        right_matrix = torch.zeros(3, 6)
        left_matrix.scatter_(1, left_interp_indices, left_interp_values_copy)
        right_matrix.scatter_(1, right_interp_indices, right_interp_values_copy)
        actual_mat = left_matrix.matmul(base_lazy_tensor_copy).matmul(right_matrix.transpose(-1, -2))
        actual = gpytorch.inv_matmul(actual_mat, test_matrix_tensor_copy)

        self.assertTrue(approx_equal(res, actual))

        # Backward pass
        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_lazy_tensor.grad, base_lazy_tensor_copy.grad))
        self.assertTrue(approx_equal(left_interp_values.grad, left_interp_values_copy.grad))

    def test_inv_matmul_batch(self):
        base_lazy_tensor = torch.randn(6, 6)
        base_lazy_tensor = (base_lazy_tensor.t().matmul(base_lazy_tensor)).unsqueeze(0).repeat(5, 1, 1)
        base_lazy_tensor_copy = base_lazy_tensor.clone()
        base_lazy_tensor.requires_grad = True
        base_lazy_tensor_copy.requires_grad = True

        test_matrix_tensor = torch.randn(5, 3, 4)
        test_matrix_tensor_copy = test_matrix_tensor.clone()
        test_matrix_tensor.requires_grad = True
        test_matrix_tensor_copy.requires_grad = True

        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).unsqueeze(0).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).unsqueeze(0).repeat(5, 1, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True

        right_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).unsqueeze(0).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).unsqueeze(0).repeat(5, 1, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        interp_lazy_tensor = InterpolatedLazyTensor(
            NonLazyTensor(base_lazy_tensor),
            left_interp_indices,
            left_interp_values,
            right_interp_indices,
            right_interp_values,
        )
        res = interp_lazy_tensor.inv_matmul(test_matrix_tensor)

        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(3, 6)
            right_matrix_comp = torch.zeros(3, 6)
            left_matrix_comp.scatter_(1, left_interp_indices[i], left_interp_values_copy[i])
            right_matrix_comp.scatter_(1, right_interp_indices[i], right_interp_values_copy[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)
        actual_mat = left_matrix.matmul(base_lazy_tensor_copy).matmul(right_matrix.transpose(-1, -2))
        actual = gpytorch.inv_matmul(actual_mat, test_matrix_tensor_copy)

        self.assertTrue(approx_equal(res, actual))

        # Backward pass
        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_lazy_tensor.grad, base_lazy_tensor_copy.grad))
        self.assertTrue(approx_equal(left_interp_values.grad, left_interp_values_copy.grad))

    def test_matmul_batch(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 3, 1)
        left_interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 3, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 3, 1)
        right_interp_values = torch.tensor([[1, 2], [2, 0.5], [1, 3]], dtype=torch.float).repeat(5, 3, 1)

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(1, 2).matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True
        test_matrix = torch.randn(1, 9, 4)

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_tensor.matmul(test_matrix)

        left_matrix = torch.tensor(
            [
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
            ],
            dtype=torch.float,
        ).repeat(5, 1, 1)

        right_matrix = torch.tensor(
            [
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
            ],
            dtype=torch.float,
        ).repeat(5, 1, 1)
        actual = left_matrix.matmul(base_lazy_tensor_mat).matmul(right_matrix.transpose(-1, -2)).matmul(test_matrix)

        self.assertTrue(approx_equal(res, actual))

    def test_getitem_batch(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(1, 2).matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = (
            base_lazy_tensor[:, 2:5, 0:3]
            + base_lazy_tensor[:, 2:5, 1:4]
            + base_lazy_tensor[:, 3:6, 0:3]
            + base_lazy_tensor[:, 3:6, 1:4]
        ).evaluate()

        self.assertTrue(approx_equal(interp_lazy_tensor[2].evaluate(), actual[2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[0:2].evaluate(), actual[0:2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[:, 2:3].evaluate(), actual[:, 2:3]))
        self.assertTrue(approx_equal(interp_lazy_tensor[:, 0:2].evaluate(), actual[:, 0:2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[1, :1, :2].evaluate(), actual[1, :1, :2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[1, 1, :2], actual[1, 1, :2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[1, :1, 2], actual[1, :1, 2]))

    def test_get_item_tensor_index(self):
        # Tests the default LV.__getitem__ behavior
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]])
        right_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)

        base_lazy_tensor_mat = torch.randn(6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(-1, -2).matmul(base_lazy_tensor_mat)

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        evaluated = interp_lazy_tensor.evaluate()

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))

    def test_get_item_tensor_index_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(1, 2).matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        evaluated = interp_lazy_tensor.evaluate()

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(approx_equal(interp_lazy_tensor[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(interp_lazy_tensor[index], evaluated[index]))

    def test_diag(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]])
        right_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)

        base_lazy_tensor_mat = torch.randn(6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.t().matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = interp_lazy_tensor.evaluate()
        self.assertTrue(approx_equal(actual.diag(), interp_lazy_tensor.diag()))

    def test_batch_diag(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(1, 2).matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = interp_lazy_tensor.evaluate()
        actual_diag = torch.stack(
            [actual[0].diag(), actual[1].diag(), actual[2].diag(), actual[3].diag(), actual[4].diag()]
        )

        self.assertTrue(approx_equal(actual_diag, interp_lazy_tensor.diag()))

    def test_sample(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float)

        base_lazy_tensor_mat = torch.randn(6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.t().matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, left_interp_indices, left_interp_values
        )

        actual = interp_lazy_tensor.evaluate()

        samples = interp_lazy_tensor.zero_mean_mvn_samples(10000)
        sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1e-5, 1e5)).max().item(), 2e-1)

    def test_batch_sample(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float).repeat(5, 1, 1)

        base_lazy_tensor_mat = torch.randn(5, 6, 6)
        base_lazy_tensor_mat = base_lazy_tensor_mat.transpose(1, 2).matmul(base_lazy_tensor_mat)
        base_lazy_tensor_mat.requires_grad = True

        base_lazy_tensor = NonLazyTensor(base_lazy_tensor_mat)
        interp_lazy_tensor = InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, left_interp_indices, left_interp_values
        )

        actual = interp_lazy_tensor.evaluate()

        samples = interp_lazy_tensor.zero_mean_mvn_samples(10000)
        sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1e-5, 1e5)).max().item(), 2e-1)


if __name__ == "__main__":
    unittest.main()
