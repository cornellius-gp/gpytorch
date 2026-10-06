#!/usr/bin/env python3

import gc
import unittest
import weakref

import torch

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestExactGPStateDictPreHookLeak(unittest.TestCase):
    """Regression tests for #2649 (ExactGP retaining load_state_dict pre-hooks)."""

    def _make_model(self, train_x, train_y):
        likelihood = GaussianLikelihood()
        return ExactGPModel(train_x, train_y, likelihood)

    def test_init_clears_load_state_dict_pre_hooks(self):
        train_x = torch.randn(20, 2)
        train_y = torch.randn(20)
        model = self._make_model(train_x, train_y)
        self.assertEqual(len(model._load_state_dict_pre_hooks), 0)

    def test_repeated_construction_does_not_retain_inputs(self):
        # CPU-friendly proxy: construct/destroy models and ensure train inputs
        # are not retained via pre-hook reference cycles.
        input_refs = []
        for _ in range(30):
            train_x = torch.randn(20, 2)
            train_y = torch.randn(20)
            input_refs.append(weakref.ref(train_x))
            model = self._make_model(train_x, train_y)
            self.assertEqual(len(model._load_state_dict_pre_hooks), 0)
            del model, train_x, train_y
            gc.collect()

        alive = sum(ref() is not None for ref in input_refs)
        self.assertEqual(
            alive,
            0,
            f"Expected train inputs to be collectable; {alive}/30 still alive",
        )

    def test_fantasy_model_clears_load_state_dict_pre_hooks(self):
        train_x = torch.randn(20, 2)
        train_y = torch.randn(20)
        model = self._make_model(train_x, train_y)
        model.eval()
        model(torch.randn(5, 2))

        fantasy = model.get_fantasy_model(torch.randn(3, 2), torch.randn(3))
        self.assertEqual(len(model._load_state_dict_pre_hooks), 0)
        self.assertEqual(len(fantasy._load_state_dict_pre_hooks), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for allocated-bytes check")
    def test_repeated_construction_cuda_memory_does_not_grow_unbounded(self):
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        for _ in range(20):
            train_x = torch.randn(20, 2, device=device)
            train_y = torch.randn(20, device=device)
            likelihood = GaussianLikelihood().to(device)
            model = ExactGPModel(train_x, train_y, likelihood)
            self.assertEqual(len(model._load_state_dict_pre_hooks), 0)
            del model, likelihood, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()

        growth = torch.cuda.memory_allocated() - baseline
        self.assertLess(
            growth,
            1_000_000,
            f"CUDA memory grew by {growth} bytes after constructing/destroying ExactGP models",
        )


if __name__ == "__main__":
    unittest.main()
