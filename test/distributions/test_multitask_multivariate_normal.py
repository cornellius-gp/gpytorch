#!/usr/bin/env python3

import math
import os
import random
import unittest

import torch
from linear_operator.operators import DiagLinearOperator, KroneckerProductLinearOperator

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device


class TestMultiTaskMultivariateNormal(BaseTestCase, unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_multitask_multivariate_normal_exceptions(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1], device=device, dtype=dtype)
            covmat = torch.eye(2, device=device, dtype=dtype)
            with self.assertRaises(RuntimeError):
                MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

    def test_multitask_multivariate_normal_exceptions_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multitask_multivariate_normal_exceptions(cuda=True)

    def test_multitask_multivariate_normal(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=dtype, device=device)
            variance = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype, device=device)

            # interleaved
            covmat = variance.view(-1).diag_embed()
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
            self.assertTrue(torch.equal(mtmvn.mean, mean))
            self.assertTrue(torch.allclose(mtmvn.variance, variance))
            self.assertTrue(torch.allclose(mtmvn.scale_tril, covmat.sqrt()))
            self.assertTrue(mtmvn.event_shape == torch.Size([3, 2]))
            self.assertTrue(mtmvn.batch_shape == torch.Size())
            mvn_plus1 = mtmvn + 1
            self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
            self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
            mvn_times2 = mtmvn * 2
            self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
            self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
            mvn_divby2 = mtmvn / 2
            self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
            self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
            self.assertAlmostEqual(mtmvn.entropy().item(), 11.80326, places=4)
            self.assertAlmostEqual(
                mtmvn.log_prob(torch.zeros(3, 2, device=device, dtype=dtype)).item(), -14.52826, places=4
            )
            logprob = mtmvn.log_prob(torch.zeros(2, 3, 2, device=device, dtype=dtype))
            logprob_expected = -14.52826 * torch.ones(2, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(logprob, logprob_expected))
            conf_lower, conf_upper = mtmvn.confidence_region()
            self.assertTrue(torch.allclose(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
            self.assertTrue(torch.allclose(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
            self.assertTrue(mtmvn.sample().shape == torch.Size([3, 2]))
            self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 3, 2]))
            self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 3, 2]))

            # non-interleaved
            covmat = variance.transpose(-1, -2).reshape(-1).diag_embed()
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat, interleaved=False)
            self.assertTrue(torch.equal(mtmvn.mean, mean))
            self.assertTrue(torch.allclose(mtmvn.variance, variance))
            self.assertTrue(torch.allclose(mtmvn.scale_tril, covmat.sqrt()))
            self.assertTrue(mtmvn.event_shape == torch.Size([3, 2]))
            self.assertTrue(mtmvn.batch_shape == torch.Size())

    def test_multitask_multivariate_normal_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multitask_multivariate_normal(cuda=True)

    def test_multitask_multivariate_normal_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=dtype, device=device).repeat(2, 1, 1)
            variance = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype, device=device).repeat(2, 1, 1)

            # interleaved
            covmat = variance.view(2, 1, -1) * torch.eye(6, device=device, dtype=dtype)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
            self.assertTrue(torch.equal(mtmvn.mean, mean))
            self.assertTrue(torch.allclose(mtmvn.variance, variance))
            self.assertTrue(torch.allclose(mtmvn.scale_tril, covmat.sqrt()))
            self.assertTrue(mtmvn.event_shape == torch.Size([3, 2]))
            self.assertTrue(mtmvn.batch_shape == torch.Size([2]))
            mvn_plus1 = mtmvn + 1
            self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
            self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
            mvn_times2 = mtmvn * 2
            self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
            self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
            mvn_divby2 = mtmvn / 2
            self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
            self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
            self.assertTrue(torch.allclose(mtmvn.entropy(), 11.80326 * torch.ones(2, device=device, dtype=dtype)))
            logprob = mtmvn.log_prob(torch.zeros(2, 3, 2, device=device, dtype=dtype))
            logprob_expected = -14.52826 * torch.ones(2, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(logprob, logprob_expected))
            logprob = mtmvn.log_prob(torch.zeros(3, 2, 3, 2, device=device, dtype=dtype))
            logprob_expected = -14.52826 * torch.ones(3, 2, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(logprob, logprob_expected))
            conf_lower, conf_upper = mtmvn.confidence_region()
            self.assertTrue(torch.allclose(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
            self.assertTrue(torch.allclose(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
            self.assertTrue(mtmvn.sample().shape == torch.Size([2, 3, 2]))
            self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 2, 3, 2]))
            self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 2, 3, 2]))

            # non-interleaved
            covmat = variance.transpose(-1, -2).reshape(2, 1, -1) * torch.eye(6, device=device, dtype=dtype)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat, interleaved=False)
            self.assertTrue(torch.equal(mtmvn.mean, mean))
            self.assertTrue(torch.allclose(mtmvn.variance, variance))
            self.assertTrue(torch.allclose(mtmvn.scale_tril, covmat.sqrt()))
            self.assertTrue(mtmvn.event_shape == torch.Size([3, 2]))
            self.assertTrue(mtmvn.batch_shape == torch.Size([2]))

    def test_multitask_multivariate_normal_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multitask_multivariate_normal_batch(cuda=True)

    def test_multivariate_normal_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=dtype, device=device)
            variance = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype, device=device)
            covmat = variance.view(-1).diag_embed()
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
            base_samples = mtmvn.get_base_samples(torch.Size([3, 4]))
            self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 3, 2]))
            base_samples = mtmvn.get_base_samples()
            self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 2]))

    def test_multivariate_normal_correlated_samples_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_correlated_samples(cuda=True)

    def test_multivariate_normal_batch_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=dtype, device=device).repeat(2, 1, 1)
            variance = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype, device=device).repeat(2, 1, 1)
            covmat = variance.view(2, 1, -1) * torch.eye(6, device=device, dtype=dtype)
            mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
            base_samples = mtmvn.get_base_samples(torch.Size((3, 4)))
            self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 3, 2]))
            base_samples = mtmvn.get_base_samples()
            self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([2, 3, 2]))

    def test_multivariate_normal_batch_correlated_samples_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multivariate_normal_batch_correlated_samples(cuda=True)

    def test_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.randn(4, 3, device=device, dtype=dtype)
            var = torch.randn(12, device=device, dtype=dtype).abs_()
            values = mean + 0.5
            diffs = (values - mean).view(-1)

            res = MultitaskMultivariateNormal(mean, DiagLinearOperator(var)).log_prob(values)
            actual = -0.5 * (math.log(math.pi * 2) * 12 + var.log().sum() + (diffs / var * diffs).sum())
            self.assertLess((res - actual).div(res).abs().item(), 1e-2)

            mean = torch.randn(3, 4, 3, device=device, dtype=dtype)
            var = torch.randn(3, 12, device=device, dtype=dtype).abs_()
            values = mean + 0.5
            diffs = (values - mean).view(3, -1)

            res = MultitaskMultivariateNormal(mean, DiagLinearOperator(var)).log_prob(values)
            actual = -0.5 * (math.log(math.pi * 2) * 12 + var.log().sum(-1) + (diffs / var * diffs).sum(-1))
            self.assertLess((res - actual).div(res).abs().norm(), 1e-2)

    def test_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_log_prob(cuda=True)

    def test_to_data_independent_dist(self, dtype=torch.float, device="cpu", interleaved=True):
        # Create a fake covariance
        factor = torch.randn(4, 4, device=device, dtype=dtype)
        data_covar = factor.mT @ factor
        task_covar = torch.tensor([[1.0, 0.3, 0.1], [0.3, 1.0, 0.3], [0.1, 0.3, 1.0]], device=device, dtype=dtype)
        if interleaved:
            covar = KroneckerProductLinearOperator(data_covar, task_covar)
        else:
            covar = KroneckerProductLinearOperator(task_covar, data_covar)

        mean = torch.randn(4, 3, device=device, dtype=dtype)
        dist = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)

        res = dist.to_data_independent_dist(jitter_val=1e-4)
        self.assertEqual(res.mean, mean)
        data_var = data_covar.diagonal(dim1=-1, dim2=-2)
        jitter = torch.eye(3, dtype=dtype, device=device) * 1e-4
        self.assertAllClose(res.covariance_matrix, data_var.view(-1, 1, 1) * task_covar + jitter)

    def test_to_data_independent_dist_no_interleave(self, dtype=torch.float, device="cpu"):
        return self.test_to_data_independent_dist(dtype=dtype, device=device, interleaved=False)

    def test_multitask_from_batch(self):
        mean = torch.randn(2, 3)
        variance = torch.randn(2, 3).clamp_min(1e-6)
        mvn = MultivariateNormal(mean, DiagLinearOperator(variance))
        mmvn = MultitaskMultivariateNormal.from_batch_mvn(mvn, task_dim=-1)
        self.assertTrue(isinstance(mmvn, MultitaskMultivariateNormal))
        self.assertEqual(mmvn.batch_shape, torch.Size([]))
        self.assertEqual(mmvn.event_shape, torch.Size([3, 2]))
        self.assertEqual(mmvn.covariance_matrix.shape, torch.Size([6, 6]))
        self.assertEqual(mmvn.mean, mean.transpose(-1, -2))
        self.assertEqual(mmvn.variance, variance.transpose(-1, -2))

        mean = torch.randn(2, 4, 3)
        variance = torch.randn(2, 4, 3).clamp_min(1e-6)
        mvn = MultivariateNormal(mean, DiagLinearOperator(variance))
        mmvn = MultitaskMultivariateNormal.from_batch_mvn(mvn, task_dim=0)
        self.assertTrue(isinstance(mmvn, MultitaskMultivariateNormal))
        self.assertEqual(mmvn.batch_shape, torch.Size([4]))
        self.assertEqual(mmvn.event_shape, torch.Size([3, 2]))
        self.assertEqual(mmvn.covariance_matrix.shape, torch.Size([4, 6, 6]))
        self.assertEqual(mmvn.mean, mean.permute(1, 2, 0))
        self.assertEqual(mmvn.variance, variance.permute(1, 2, 0))

    def test_multitask_from_repeat(self):
        mean = torch.randn(2, 3)
        variance = torch.randn(2, 3).clamp_min(1e-6)
        mvn = MultivariateNormal(mean, DiagLinearOperator(variance))
        mmvn = MultitaskMultivariateNormal.from_repeated_mvn(mvn, num_tasks=4)
        self.assertTrue(isinstance(mmvn, MultitaskMultivariateNormal))
        self.assertEqual(mmvn.batch_shape, torch.Size([2]))
        self.assertEqual(mmvn.event_shape, torch.Size([3, 4]))
        self.assertEqual(mmvn.covariance_matrix.shape, torch.Size([2, 12, 12]))
        for i in range(4):
            self.assertEqual(mmvn.mean[..., i], mean)
            self.assertEqual(mmvn.variance[..., i], variance)

    def test_from_independent_mvns(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # Test non-batch mode mvns
            n_tasks = 2
            n = 4
            mvns = [
                MultivariateNormal(
                    mean=torch.randn(4, device=device, dtype=dtype),
                    covariance_matrix=DiagLinearOperator(torch.randn(n, device=device, dtype=dtype).abs_()),
                )
                for i in range(n_tasks)
            ]
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
            expected_mean_shape = [n, n_tasks]
            expected_covar_shape = [n * n_tasks] * 2
            self.assertEqual(list(mvn.mean.shape), expected_mean_shape)
            self.assertEqual(list(mvn.covariance_matrix.shape), expected_covar_shape)

            # Test mixed batch mode mvns
            # Second MVN is batched, so the first one will be expanded to match.
            mvns[1] = mvns[1].expand(torch.Size([3]))
            expected_mvn = mvn.expand(torch.Size([3]))
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
            self.assertTrue(torch.equal(mvn.mean, expected_mvn.mean))
            self.assertTrue(torch.equal(mvn.covariance_matrix, expected_mvn.covariance_matrix))

            # Test batch mode mvns
            b = 3
            mvns = [
                MultivariateNormal(
                    mean=torch.randn(b, n, device=device, dtype=dtype),
                    covariance_matrix=DiagLinearOperator(torch.randn(b, n, device=device, dtype=dtype).abs_()),
                )
                for i in range(n_tasks)
            ]
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
            self.assertEqual(list(mvn.mean.shape), [b] + expected_mean_shape)
            self.assertEqual(list(mvn.covariance_matrix.shape), [b] + expected_covar_shape)

    def test_from_independent_mvns_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_from_independent_mvns(cuda=True)

    def test_multitask_multivariate_normal_broadcasting(self):
        mean = torch.randn(5, 1, 3)
        _covar = torch.randn(6, 6)
        covar = _covar @ _covar.transpose(-1, -2)
        sample = MultitaskMultivariateNormal(mean, covar).rsample()
        self.assertEqual(sample.shape, torch.Size([5, 2, 3]))

        mean = torch.randn(5, 1)
        _covar = torch.randn(3, 10, 10)
        covar = _covar @ _covar.transpose(-1, -2)
        sample = MultitaskMultivariateNormal(mean, covar).rsample()
        self.assertEqual(sample.shape, torch.Size([3, 5, 2]))

        with self.assertRaises(RuntimeError):
            mean = torch.randn(5, 1)
            _covar = torch.randn(12, 12)
            covar = _covar @ _covar.transpose(-1, -2)
            MultitaskMultivariateNormal(mean, covar)

    def test_getitem_interleaved(self):
        mean_shape = (2, 4, 3, 2)
        covar_shape = (2, 4, 6, 6)
        mean = torch.randn(mean_shape)
        _covar = torch.randn(covar_shape)
        covar = _covar @ _covar.transpose(-1, -2)
        distribution = MultitaskMultivariateNormal(mean, covar, validate_args=True)

        def flat(observation: int, task: int) -> int:
            return observation * 2 + task

        part = distribution[1, -1]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size(()))
        self.assertEqual(part.event_shape, torch.Size((3, 2)))
        self.assertAllClose(part.mean, mean[1, -1])
        self.assertAllClose(part.covariance_matrix, covar[1, -1])

        part = distribution[1, 0, ...]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size(()))
        self.assertEqual(part.event_shape, torch.Size((3, 2)))
        self.assertAllClose(part.mean, mean[1, 0])
        self.assertAllClose(part.covariance_matrix, covar[1, 0])

        part = distribution[..., 2, 1]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2,)))
        self.assertEqual(part.event_shape, (4,))
        self.assertAllClose(part.mean, mean[..., 2, 1])
        self.assertAllClose(part.covariance_matrix, torch.diag_embed(covar[:, :, flat(2, 1), flat(2, 1)]))

        part = distribution[1, ..., -2]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((4,)))
        self.assertEqual(part.event_shape, torch.Size((3,)))
        self.assertAllClose(part.mean, mean[1, :, :, 0])
        self.assertAllClose(part.covariance_matrix, covar[1, :, ::2, ::2])

        part = distribution[..., 2, :]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((2,)))
        self.assertAllClose(part.mean, mean[:, :, 2, :])
        self.assertAllClose(part.covariance_matrix, covar[:, :, 2 * 2 : 3 * 2, 2 * 2 : 3 * 2])

        part = distribution[0, :, :, torch.tensor([1, 0])]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((4,)))
        self.assertEqual(part.event_shape, torch.Size((3, 2)))
        self.assertAllClose(part.mean, mean[0, ..., torch.tensor([1, 0])])
        indices = torch.tensor([1, 0, 3, 2, 5, 4])
        self.assertAllClose(part.covariance_matrix, covar[0, :, indices][..., indices])

        part = distribution[:, 1, torch.tensor([2, 0])]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2,)))
        self.assertEqual(part.event_shape, torch.Size((2, 2)))
        self.assertAllClose(part.mean, mean[:, 1, torch.tensor([2, 0])])
        indices = torch.tensor([4, 5, 0, 1])
        self.assertAllClose(part.covariance_matrix, covar[:, 1, indices][..., indices])

        part = distribution[..., 1:, :-1]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((2, 1)))
        self.assertAllClose(part.mean, mean[..., 1:, :-1])
        indices = torch.tensor([flat(1, 0), flat(2, 0)])
        self.assertAllClose(part.covariance_matrix, covar[..., indices, :][..., indices])

        part = distribution[..., torch.tensor([2, 0, 2]), torch.tensor([1, 0, 0])]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((3,)))
        self.assertAllClose(part.mean, mean[..., torch.tensor([2, 0, 2]), torch.tensor([1, 0, 0])])
        indices = torch.tensor([flat(2, 1), flat(0, 0), flat(2, 0)])
        self.assertAllClose(part.covariance_matrix, covar[..., indices, :][..., indices])

    def test_getitem_non_interleaved(self):
        mean_shape = (2, 4, 3, 2)
        covar_shape = (2, 4, 6, 6)
        mean = torch.randn(mean_shape)
        _covar = torch.randn(covar_shape)
        covar = _covar @ _covar.transpose(-1, -2)
        distribution = MultitaskMultivariateNormal(mean, covar, validate_args=True, interleaved=False)

        def flat(observation: int, task: int) -> int:
            return task * 3 + observation

        part = distribution[1, -1]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size(()))
        self.assertEqual(part.event_shape, torch.Size((3, 2)))
        self.assertAllClose(part.mean, mean[1, -1])
        self.assertAllClose(part.covariance_matrix, covar[1, -1])

        part = distribution[..., 2, 1]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2,)))
        self.assertEqual(part.event_shape, (4,))
        self.assertAllClose(part.mean, mean[..., 2, 1])
        self.assertAllClose(part.covariance_matrix, torch.diag_embed(covar[:, :, flat(2, 1), flat(2, 1)]))

        part = distribution[1, ..., -2]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((4,)))
        self.assertEqual(part.event_shape, torch.Size((3,)))
        self.assertAllClose(part.mean, mean[1, :, :, 0])
        self.assertAllClose(part.covariance_matrix, covar[1, :, :3, :3])

        part = distribution[..., 2, :]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((2,)))
        self.assertAllClose(part.mean, mean[:, :, 2, :])
        self.assertAllClose(part.covariance_matrix, covar[:, :, 2::3, 2::3])

        part = distribution[0, :, :, torch.tensor([1, 0])]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((4,)))
        self.assertEqual(part.event_shape, torch.Size((3, 2)))
        self.assertAllClose(part.mean, mean[0, ..., torch.tensor([1, 0])])
        indices = torch.tensor([3, 4, 5, 0, 1, 2])
        self.assertAllClose(part.covariance_matrix, covar[0, :, indices][..., indices])

        part = distribution[:, 1, torch.tensor([2, 0])]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2,)))
        self.assertEqual(part.event_shape, torch.Size((2, 2)))
        self.assertAllClose(part.mean, mean[:, 1, torch.tensor([2, 0])])
        indices = torch.tensor([2, 0, 5, 3])
        self.assertAllClose(part.covariance_matrix, covar[:, 1, indices][..., indices])

        part = distribution[..., 1:, :-1]
        self.assertIsInstance(part, MultitaskMultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((2, 1)))
        self.assertAllClose(part.mean, mean[..., 1:, :-1])
        indices = torch.tensor([flat(1, 0), flat(2, 0)])
        self.assertAllClose(part.covariance_matrix, covar[..., indices, :][..., indices])

        part = distribution[..., torch.tensor([2, 0, 2]), torch.tensor([1, 0, 0])]
        self.assertFalse(isinstance(part, MultitaskMultivariateNormal))
        self.assertIsInstance(part, MultivariateNormal)
        self.assertEqual(part.batch_shape, torch.Size((2, 4)))
        self.assertEqual(part.event_shape, torch.Size((3,)))
        self.assertAllClose(part.mean, mean[..., torch.tensor([2, 0, 2]), torch.tensor([1, 0, 0])])
        indices = torch.tensor([flat(2, 1), flat(0, 0), flat(2, 0)])
        self.assertAllClose(part.covariance_matrix, covar[..., indices, :][..., indices])

    def test_repr(self):
        mean = torch.randn(5, 1, 3)
        covar = torch.eye(6)
        dist = MultitaskMultivariateNormal(mean, covar)
        dist_repr = str(dist)
        self.assertEqual(dist_repr, "MultitaskMultivariateNormal(mean shape: torch.Size([5, 2, 3]))")


if __name__ == "__main__":
    unittest.main()
