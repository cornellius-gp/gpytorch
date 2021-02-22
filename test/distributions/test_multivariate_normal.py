#!/usr/bin/env python3

import math
import unittest

import torch
from torch.distributions import MultivariateNormal as TMultivariateNormal

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, LazyTensor, NonLazyTensor, RootLazyTensor, lazify
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device


class TestMultivariateNormal(BaseTestCase, unittest.TestCase):
    seed = 1

    def test_multivariate_normal_non_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype))
            mvn = MultivariateNormal(mean=mean, covariance_matrix=covmat, validate_args=True)
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
            self.assertAllClose(mvn.variance, torch.diag(covmat))
            self.assertAllClose(mvn.scale_tril, covmat.sqrt())
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
            self.assertAlmostEqual(mvn.log_prob(torch.zeros(3, device=device, dtype=dtype)).item(), -4.8157, places=4)
            logprob = mvn.log_prob(torch.zeros(2, 3, device=device, dtype=dtype))
            logprob_expected = torch.tensor([-4.8157, -4.8157], device=device, dtype=dtype)
            self.assertAllClose(logprob, logprob_expected)
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_non_lazy_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_non_lazy(cuda=True)

    def test_multivariate_normal_batch_non_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype))
            mvn = MultivariateNormal(
                mean=mean.repeat(2, 1), covariance_matrix=covmat.repeat(2, 1, 1), validate_args=True
            )
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
            self.assertAllClose(mvn.variance, covmat.diag().repeat(2, 1))
            self.assertAllClose(mvn.scale_tril, torch.diag(covmat.diag().sqrt()).repeat(2, 1, 1))
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAllClose(mvn.entropy(), 4.3157 * torch.ones(2, device=device, dtype=dtype))
            logprob = mvn.log_prob(torch.zeros(2, 3, device=device, dtype=dtype))
            logprob_expected = -4.8157 * torch.ones(2, device=device, dtype=dtype)
            self.assertAllClose(logprob, logprob_expected)
            logprob = mvn.log_prob(torch.zeros(2, 2, 3, device=device, dtype=dtype))
            logprob_expected = -4.8157 * torch.ones(2, 2, device=device, dtype=dtype)
            self.assertAllClose(logprob, logprob_expected)
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_batch_non_lazy_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_batch_non_lazy(cuda=True)

    def test_multivariate_normal_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype))
            covmat_chol = torch.cholesky(covmat)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
            self.assertAllClose(mvn.variance, torch.diag(covmat))
            self.assertAllClose(mvn.covariance_matrix, covmat)
            self.assertAllClose(mvn._unbroadcasted_scale_tril, covmat_chol)
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            self.assertAllClose(mvn_plus1._unbroadcasted_scale_tril, covmat_chol)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            self.assertAllClose(mvn_times2._unbroadcasted_scale_tril, covmat_chol * 2)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAllClose(mvn_divby2._unbroadcasted_scale_tril, covmat_chol / 2)
            # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
            # uses using root_decomposition which is not very reliable
            # self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
            # self.assertAlmostEqual(mvn.log_prob(torch.zeros(3)).item(), -4.8157, places=4)
            # self.assertTrue(
            #     torch.allclose(
            #         mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
            #     )
            # )
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_lazy_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_lazy(cuda=True)

    def test_multivariate_normal_batch_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype).repeat(2, 1)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype)).repeat(2, 1, 1)
            covmat_chol = torch.cholesky(covmat)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
            self.assertAllClose(mvn.variance, torch.diagonal(covmat, dim1=-2, dim2=-1))
            self.assertAllClose(mvn._unbroadcasted_scale_tril, covmat_chol)
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            self.assertAllClose(mvn_plus1._unbroadcasted_scale_tril, covmat_chol)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            self.assertAllClose(mvn_times2._unbroadcasted_scale_tril, covmat_chol * 2)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAllClose(mvn_divby2._unbroadcasted_scale_tril, covmat_chol / 2)
            # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
            # uses using root_decomposition which is not very reliable
            # self.assertTrue(torch.allclose(mvn.entropy(), 4.3157 * torch.ones(2)))
            # self.assertTrue(
            #     torch.allclose(mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
            # )
            # self.assertTrue(
            #     torch.allclose(mvn.log_prob(torch.zeros(2, 2, 3)), -4.8157 * torch.ones(2, 2))
            # )
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_batch_lazy_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_batch_lazy(cuda=True)

    def test_multivariate_normal_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype))
            mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))
            base_samples = mvn.get_base_samples(torch.Size([3, 4]))
            self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 3]))
            base_samples = mvn.get_base_samples()
            self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3]))

    def test_multivariate_normal_correlated_samples_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_correlated_samples(cuda=True)

    def test_multivariate_normal_batch_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype))
            mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=NonLazyTensor(covmat).repeat(2, 1, 1))
            base_samples = mvn.get_base_samples(torch.Size((3, 4)))
            self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 3]))
            base_samples = mvn.get_base_samples()
            self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([2, 3]))

    def test_multivariate_normal_batch_correlated_samples_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_batch_correlated_samples(cuda=True)

    def test_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.randn(4, device=device, dtype=dtype)
            var = torch.randn(4, device=device, dtype=dtype).abs_()
            values = torch.randn(4, device=device, dtype=dtype)

            res = MultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
            actual = TMultivariateNormal(mean, torch.eye(4, device=device, dtype=dtype) * var).log_prob(values)
            self.assertLess((res - actual).div(res).abs().item(), 1e-2)

            mean = torch.randn(3, 4, device=device, dtype=dtype)
            var = torch.randn(3, 4, device=device, dtype=dtype).abs_()
            values = torch.randn(3, 4, device=device, dtype=dtype)

            res = MultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
            actual = TMultivariateNormal(
                mean, var.unsqueeze(-1) * torch.eye(4, device=device, dtype=dtype).repeat(3, 1, 1)
            ).log_prob(values)
            self.assertLess((res - actual).div(res).abs().norm(), 1e-2)

    def test_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_log_prob(cuda=True)

    def test_kl_divergence(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean0 = torch.randn(4, device=device, dtype=dtype)
            mean1 = mean0 + 1
            var0 = torch.randn(4, device=device, dtype=dtype).abs_()
            var1 = var0 * math.exp(2)

            dist_a = MultivariateNormal(mean0, DiagLazyTensor(var0))
            dist_b = MultivariateNormal(mean1, DiagLazyTensor(var0))
            dist_c = MultivariateNormal(mean0, DiagLazyTensor(var1))

            res = torch.distributions.kl.kl_divergence(dist_a, dist_a)
            actual = 0.0
            self.assertLess((res - actual).abs().item(), 1e-2)

            res = torch.distributions.kl.kl_divergence(dist_b, dist_a)
            actual = var0.reciprocal().sum().div(2.0)
            self.assertLess((res - actual).div(res).abs().item(), 1e-2)

            res = torch.distributions.kl.kl_divergence(dist_a, dist_c)
            actual = 0.5 * (8 - 4 + 4 * math.exp(-2))
            self.assertLess((res - actual).div(res).abs().item(), 1e-2)

    def test_kl_divergence_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_kl_divergence(cuda=True)

    def test_getitem(self):
        shape = (2, 4, 3, 2)
        cov = torch.randn(*shape, shape[1])
        cov = cov @ cov.transpose(-1, -2)
        mean = torch.randn(*shape)
        dist = MultivariateNormal(mean, cov)
        dist_cov = dist.covariance_matrix

        d = dist[1]
        assert torch.equal(d.mean, dist.mean[1])
        self.assertAllClose(d.covariance_matrix, dist_cov[1])

        d = dist[..., 1]
        assert torch.equal(d.mean, dist.mean[..., 1])
        cov = dist_cov[..., 1, 1]
        self.assertAllClose(d.covariance_matrix, cov.unsqueeze(-1) * torch.eye(shape[-2]))

        d = dist[:, [2, 3], :, 1:]
        assert torch.equal(d.mean, dist.mean[:, [2, 3], :, 1:])
        self.assertAllClose(d.covariance_matrix, dist_cov[:, [2, 3], :, 1:, 1:])

        d = dist[:, :, ..., [0, 1, 1, 0]]
        assert torch.equal(d.mean, dist.mean[..., [0, 1, 1, 0]])
        self.assertAllClose(d.covariance_matrix, dist_cov[..., [0, 1, 1, 0], :][..., [0, 1, 1, 0]])

        d = dist[1, 2, 2, ...]
        assert torch.equal(d.mean, dist.mean[1, 2, 2, :])
        self.assertAllClose(d.covariance_matrix, dist_cov[1, 2, 2, :, :])

    def test_base_sample_shape(self):
        a = torch.randn(5, 10)
        lazy_square_a = RootLazyTensor(lazify(a))
        dist = MultivariateNormal(torch.zeros(5), lazy_square_a)

        # check that providing the base samples is okay
        samples = dist.rsample(torch.Size((16,)), base_samples=torch.randn(16, 10))
        self.assertEqual(samples.shape, torch.Size((16, 5)))

        # check that an event shape of base samples fails
        self.assertRaises(RuntimeError, dist.rsample, torch.Size((16,)), base_samples=torch.randn(16, 5))

        # check that the proper event shape of base samples is okay for
        # a non root lt
        nonlazy_square_a = lazify(lazy_square_a.evaluate())
        dist = MultivariateNormal(torch.zeros(5), nonlazy_square_a)

        samples = dist.rsample(torch.Size((16,)), base_samples=torch.randn(16, 5))
        self.assertEqual(samples.shape, torch.Size((16, 5)))


if __name__ == "__main__":
    unittest.main()
