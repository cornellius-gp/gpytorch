from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import SpectralMixtureKernel


class TestSpectralMixtureKernel(unittest.TestCase):
    def test_standard(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        means = [1, 2]
        scales = [0.5, 0.25]
        weights = [4, 2]
        kernel = SpectralMixtureKernel(num_mixtures=2)
        kernel.initialize(
            log_mixture_weights=torch.tensor([[4, 2]], dtype=torch.float).log(),
            log_mixture_means=torch.tensor([[[[1]], [[2]]]], dtype=torch.float).log(),
            log_mixture_scales=torch.tensor([[[[0.5]], [[0.25]]]], dtype=torch.float).log(),
        )
        kernel.eval()

        actual = torch.zeros(3, 2)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    new_term = torch.cos(2 * math.pi * (a[i] - b[j]) * means[k])
                    new_term *= torch.exp(-2 * (math.pi * (a[i] - b[j])) ** 2 * scales[k] ** 2)
                    new_term *= weights[k]
                    actual[i, j] += new_term.item()

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch_separate(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2], [-1, 2]], dtype=torch.float).view(2, 2, 1)
        means = torch.tensor([[1, 2], [2, 3]], dtype=torch.float).view(2, 2, 1, 1)
        scales = torch.tensor([[0.5, 0.25], [0.25, 1]], dtype=torch.float).view(2, 2, 1, 1)
        weights = torch.tensor([[4, 2], [1, 2]], dtype=torch.float).view(2, 2)
        kernel = SpectralMixtureKernel(batch_size=2, num_mixtures=2)
        kernel.initialize(
            log_mixture_weights=weights.log(), log_mixture_means=means.log(), log_mixture_scales=scales.log()
        )
        kernel.eval()

        actual = torch.zeros(2, 3, 2)
        for l in range(2):
            for k in range(2):
                for i in range(3):
                    for j in range(2):
                        new_term = torch.cos(2 * math.pi * (a[l, i] - b[l, j]) * means[l, k])
                        new_term *= torch.exp(-2 * (math.pi * (a[l, i] - b[l, j])) ** 2 * scales[l, k] ** 2)
                        new_term *= weights[l, k]
                        actual[l, i, j] += new_term.item()

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
