from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.utils import approx_equal


class TestMultiTaskMultivariateNormal(unittest.TestCase):
    def test_multitask_multivariate_normal_exceptions(self):
        mean = torch.tensor([0, 1], dtype=torch.float)
        covmat = torch.eye(2)
        with self.assertRaises(RuntimeError):
            MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

    def test_multitask_multivariate_normal(self):
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float)
        variance = 1 + torch.arange(4, dtype=torch.float)
        covmat = torch.diag(variance)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
        self.assertTrue(torch.equal(mtmvn.mean, mean))
        self.assertTrue(approx_equal(mtmvn.variance, variance.view(2, 2)))
        self.assertTrue(torch.equal(mtmvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mtmvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
        mvn_times2 = mtmvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
        mvn_divby2 = mtmvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
        self.assertAlmostEqual(mtmvn.entropy().item(), 7.2648, places=4)
        self.assertAlmostEqual(mtmvn.log_prob(torch.zeros(2, 2)).item(), -7.3064, places=4)
        self.assertTrue(approx_equal(mtmvn.log_prob(torch.zeros(3, 2, 2)), -7.3064 * torch.ones(3)))
        conf_lower, conf_upper = mtmvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
        self.assertTrue(mtmvn.sample().shape == torch.Size([2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 2, 2]))

    def test_multitask_multivariate_normal_batch(self):
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float).repeat(2, 1, 1)
        variance = 1 + torch.arange(4, dtype=torch.float)
        covmat = torch.diag(variance).repeat(2, 1, 1)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
        self.assertTrue(torch.equal(mtmvn.mean, mean))
        self.assertTrue(approx_equal(mtmvn.variance, variance.repeat(2, 1).view(2, 2, 2)))
        self.assertTrue(torch.equal(mtmvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mtmvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
        mvn_times2 = mtmvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
        mvn_divby2 = mtmvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
        self.assertTrue(approx_equal(mtmvn.entropy(), 7.2648 * torch.ones(2)))
        self.assertTrue(approx_equal(mtmvn.log_prob(torch.zeros(2, 2, 2)), -7.3064 * torch.ones(2)))
        self.assertTrue(approx_equal(mtmvn.log_prob(torch.zeros(3, 2, 2, 2)), -7.3064 * torch.ones(3, 2)))
        conf_lower, conf_upper = mtmvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
        self.assertTrue(mtmvn.sample().shape == torch.Size([2, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 2, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 2, 2, 2]))

    def test_multivariate_normal_correlated_sampels(self):
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float)
        variance = 1 + torch.arange(4, dtype=torch.float)
        covmat = torch.diag(variance)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

        base_samples = mtmvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 2]))

        base_samples = mtmvn.get_base_samples()
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([2, 2]))

    def test_multivariate_normal_batch_correlated_sampels(self):
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float).repeat(2, 1, 1)
        variance = 1 + torch.arange(4, dtype=torch.float)
        covmat = torch.diag(variance).repeat(2, 1, 1)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

        base_samples = mtmvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 2, 2]))

        base_samples = mtmvn.get_base_samples()
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([2, 2, 2]))


if __name__ == "__main__":
    unittest.main()
