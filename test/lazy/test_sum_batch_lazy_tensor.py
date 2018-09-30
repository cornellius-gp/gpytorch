from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import torch
import unittest
import gpytorch
from gpytorch.lazy import SumBatchLazyTensor, NonLazyTensor
from test._utils import approx_equal


class TestSumBatchLazyTensor(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

        blocks = torch.randn(12, 4, 4)
        self.blocks = blocks.transpose(-1, -2).matmul(blocks)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul(self):
        rhs_tensor = torch.randn(4, 8, requires_grad=True)
        rhs_tensor_copy = rhs_tensor.clone().detach().requires_grad_(True)
        block_tensor = self.blocks.clone().requires_grad_(True)
        block_tensor_copy = self.blocks.clone().requires_grad_(True)

        actual_mat = block_tensor_copy.sum(0)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor)).matmul(rhs_tensor)
        actual = actual_mat.matmul(rhs_tensor_copy)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()

        self.assertTrue(approx_equal(rhs_tensor.grad, rhs_tensor_copy.grad))
        self.assertTrue(approx_equal(block_tensor.grad, block_tensor_copy.grad))

    def test_batch_matmul(self):
        rhs_tensor = torch.randn(3, 4, 8, requires_grad=True)
        rhs_tensor_copy = rhs_tensor.clone().detach().requires_grad_(True)
        block_tensor = self.blocks.clone().requires_grad_(True)
        block_tensor_copy = self.blocks.clone().requires_grad_(True)

        actual_mat = block_tensor_copy.view(3, 4, 4, 4).sum(1)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4).matmul(rhs_tensor)
        actual = actual_mat.matmul(rhs_tensor_copy)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()

        self.assertTrue(approx_equal(rhs_tensor.grad, rhs_tensor_copy.grad))
        self.assertTrue(approx_equal(block_tensor.grad, block_tensor_copy.grad))

    def test_diag(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        actual_mat = block_tensor.sum(0)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor)).diag()
        actual = actual_mat.diag()
        self.assertTrue(approx_equal(actual, res))

    def test_batch_diag(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        actual_mat = block_tensor.view(3, 4, 4, 4).sum(1)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4).diag()
        actual = torch.cat([
            actual_mat[0].diag().unsqueeze(0),
            actual_mat[1].diag().unsqueeze(0),
            actual_mat[2].diag().unsqueeze(0),
        ])
        self.assertTrue(approx_equal(actual, res))

    def test_getitem(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        actual_mat = block_tensor.sum(0)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor))[:5, 2]
        actual = actual_mat[:5, 2]
        self.assertTrue(approx_equal(actual, res))

    def test_getitem_batch(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        actual_mat = block_tensor.view(3, 4, 4, 4).sum(1)

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4)[0].evaluate()
        actual = actual_mat[0]
        self.assertTrue(approx_equal(actual, res))

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4)[0, :5].evaluate()
        actual = actual_mat[0, :5]
        self.assertTrue(approx_equal(actual, res))

        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4)[1:, :5, 2]
        actual = actual_mat[1:, :5, 2]
        self.assertTrue(approx_equal(actual, res))

    def test_get_item_tensor_index(self):
        # Tests the default LV.__getitem__ behavior
        block_tensor = self.blocks.clone().requires_grad_(True)
        lazy_tensor = SumBatchLazyTensor(NonLazyTensor(block_tensor))
        evaluated = lazy_tensor.evaluate()

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))

    def test_get_item_tensor_index_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        block_tensor = self.blocks.clone().requires_grad_(True)
        lazy_tensor = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4)
        evaluated = lazy_tensor.evaluate()

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))

    def test_sample(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        res = SumBatchLazyTensor(NonLazyTensor(block_tensor))
        actual = res.evaluate()

        with gpytorch.settings.max_root_decomposition_size(1000):
            samples = res.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 4e-1)

    def test_batch_sample(self):
        block_tensor = self.blocks.clone().requires_grad_(True)
        res = SumBatchLazyTensor(NonLazyTensor(block_tensor), num_blocks=4)
        actual = res.evaluate()

        with gpytorch.settings.max_root_decomposition_size(1000):
            samples = res.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 4e-1)


if __name__ == "__main__":
    unittest.main()
