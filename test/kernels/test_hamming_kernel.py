#!/usr/bin/env python3

import pickle
import unittest

import torch

from gpytorch.kernels import HammingIMQKernel
from gpytorch.priors import GammaPrior


class TestHammingIMQKernel(unittest.TestCase):
    def create_seq(self, batch_size, seq_len, vocab_size):
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    def create_seq_pairs(self, batch_size, seq_len, vocab_size):
        a = self.create_seq(batch_size, seq_len, vocab_size)
        b = self.create_seq(batch_size, seq_len, vocab_size)
        set_to_a = torch.rand(batch_size, seq_len) < 0.5
        b = torch.where(set_to_a, a, b)
        return a, b

    def test_computes_hamming_imq_function(self):
        """
        Create one-hot encoded discrete sequences and flatten them.
        Compute the pairwise Hamming distance $d$ between the sequences.
        Check the result of the kernel evaluation is
        $((1 + \alpha) / (\alpha + d))^{\beta}$.
        """
        vocab_size = 8
        seq_len = 4
        alpha = 2.0
        beta = 0.5
        kernel = HammingIMQKernel(vocab_size=vocab_size)
        kernel.initialize(alpha=alpha, beta=beta)
        kernel.eval()

        # Create two discrete sequences with some matches.
        a = torch.tensor([[7, 7, 7, 7], [5, 7, 3, 4]])
        b = torch.tensor(
            [
                [7, 5, 7, 4],
                [6, 7, 3, 7],
                [5, 7, 3, 4],
            ]
        )

        # Convert to one-hot representation.
        a_one_hot = torch.zeros(*a.shape, vocab_size)
        a_one_hot.scatter_(index=a.unsqueeze(-1), dim=-1, value=1)
        b_one_hot = torch.zeros(*b.shape, vocab_size)
        b_one_hot.scatter_(index=b.unsqueeze(-1), dim=-1, value=1)

        # Flatten the one-hot representations.
        a_one_hot_flat = a_one_hot.view(a.size(0), -1)
        b_one_hot_flat = b_one_hot.view(b.size(0), -1)

        # Compute the Hamming distance.
        d = seq_len - (a_one_hot.unsqueeze(-3) * b_one_hot.unsqueeze(-4)).sum(dim=(-1, -2))

        # Compute the kernel evaluation.
        actual = ((1 + alpha) / (alpha + d)) ** beta
        res = kernel(a_one_hot_flat, b_one_hot_flat).to_dense()

        # Check the result.
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_alpha(self):
        """
        Check that the kernel can be initialized with alpha.
        """
        alpha = 2.0
        kernel = HammingIMQKernel(vocab_size=8)
        kernel.initialize(alpha=alpha)
        actual_value = torch.tensor(alpha).view_as(kernel.alpha)
        self.assertLess(torch.norm(kernel.alpha - actual_value), 1e-5)

    def test_initialize_alpha_batch(self):
        batch_size = 2
        alpha = torch.rand(batch_size)
        kernel = HammingIMQKernel(vocab_size=8, batch_shape=torch.Size([batch_size]))
        kernel.initialize(alpha=alpha)
        actual_value = alpha.view_as(kernel.alpha)
        self.assertLess(torch.norm(kernel.alpha - actual_value), 1e-5)

    def test_initialize_beta(self):
        """
        Check that the kernel can be initialized with beta.
        """
        beta = 0.5
        kernel = HammingIMQKernel(vocab_size=8)
        kernel.initialize(beta=beta)
        actual_value = torch.tensor(beta).view_as(kernel.beta)
        self.assertLess(torch.norm(kernel.beta - actual_value), 1e-5)

    def test_initialize_beta_batch(self):
        batch_size = 2
        beta = torch.rand(batch_size)
        kernel = HammingIMQKernel(vocab_size=8, batch_shape=torch.Size([batch_size]))
        kernel.initialize(beta=beta)
        actual_value = beta.view_as(kernel.beta)
        self.assertLess(torch.norm(kernel.beta - actual_value), 1e-5)

    def create_kernel_with_prior(self, alpha_prior=None, beta_prior=None):
        return HammingIMQKernel(
            vocab_size=8,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )

    def test_prior_type(self):
        self.create_kernel_with_prior()
        self.create_kernel_with_prior(
            alpha_prior=GammaPrior(1.0, 1.0),
            beta_prior=GammaPrior(1.0, 1.0),
        )
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)

    def test_pickle_with_prior(self):
        kernel = self.create_kernel_with_prior(
            alpha_prior=GammaPrior(1.0, 1.0),
            beta_prior=GammaPrior(1.0, 1.0),
        )
        pickle.loads(pickle.dumps(kernel))  # Should be able to pickle and unpickle with a prior.


if __name__ == "__main__":
    unittest.main()
