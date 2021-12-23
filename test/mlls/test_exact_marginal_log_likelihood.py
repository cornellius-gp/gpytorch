import unittest

import torch

import gpytorch

from .test_leave_one_out_pseudo_likelihood import ExactGPModel


class TestExactMarginalLogLikelihood(unittest.TestCase):
    def get_data(self, shapes, combine_terms, dtype=None, device=None):
        train_x = torch.rand(*shapes, dtype=dtype, device=device, requires_grad=True)
        train_y = torch.sin(train_x[..., 0]) + torch.cos(train_x[..., 1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)
        model = ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype, device=device)
        exact_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=likelihood,
            model=model,
            combine_terms=combine_terms
        )
        return train_x, train_y, exact_mll

    def test_smoke(self):
        """Make sure the exact_mll works without batching."""
        train_x, train_y, exact_mll = self.get_data([5, 2], combine_terms=True)
        output = exact_mll.model(train_x)
        loss = -exact_mll(output, train_y)
        loss.backward()
        self.assertTrue(train_x.grad is not None)

        train_x, train_y, exact_mll = self.get_data([5, 2], combine_terms=False)
        output = exact_mll.model(train_x)
        mll_out = exact_mll(output, train_y)
        loss = -1 * sum(mll_out)
        loss.backward()
        assert len(mll_out) == 4
        self.assertTrue(train_x.grad is not None)

    def test_smoke_batch(self):
        """Make sure the exact_mll works without batching."""
        train_x, train_y, exact_mll = self.get_data([3, 3, 3, 5, 2], combine_terms=True)
        output = exact_mll.model(train_x)
        loss = -exact_mll(output, train_y)
        assert loss.shape == (3, 3, 3)
        loss.sum().backward()
        self.assertTrue(train_x.grad is not None)

        train_x, train_y, exact_mll = self.get_data([3, 3, 3, 5, 2], combine_terms=False)
        output = exact_mll.model(train_x)
        mll_out = exact_mll(output, train_y)
        loss = -1 * sum(mll_out)
        assert len(mll_out) == 4
        assert loss.shape == (3, 3, 3)
        loss.sum().backward()
        self.assertTrue(train_x.grad is not None)


if __name__ == "__main__":
    unittest.main()
