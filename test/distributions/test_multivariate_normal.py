from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import LazyTensor, NonLazyTensor
from gpytorch.utils import approx_equal


class TestMultivariateNormal(unittest.TestCase):
    def test_multivariate_normal_non_lazy(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=covmat, validate_args=True)
        self.assertTrue(approx_equal(mvn.variance, torch.diag(covmat)))
        self.assertTrue(approx_equal(mvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
        self.assertAlmostEqual(mvn.log_prob(torch.zeros(3)).item(), -4.8157, places=4)
        self.assertTrue(approx_equal(mvn.log_prob(torch.zeros(2, 3)), torch.tensor([-4.8157, -4.8157])))
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_batch_non_lazy(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=covmat.repeat(2, 1, 1), validate_args=True)
        self.assertTrue(approx_equal(mvn.variance, covmat.diag().repeat(2, 1)))
        self.assertTrue(approx_equal(mvn.scale_tril, torch.diag(covmat.diag().sqrt()).repeat(2, 1, 1)))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        self.assertTrue(approx_equal(mvn.entropy(), 4.3157 * torch.ones(2)))
        self.assertTrue(approx_equal(mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2)))
        self.assertTrue(approx_equal(mvn.log_prob(torch.zeros(2, 2, 3)), -4.8157 * torch.ones(2, 2)))
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_lazy(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))
        self.assertIsInstance(mvn.covariance_matrix, LazyTensor)
        self.assertTrue(torch.equal(mvn.variance, torch.diag(covmat)))
        self.assertTrue(torch.equal(mvn.covariance_matrix.evaluate(), covmat))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate()))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate() * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate() / 4))
        # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
        # uses using root_decomposition which is not very reliable
        # self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
        # self.assertAlmostEqual(mvn.log_prob(torch.zeros(3)).item(), -4.8157, places=4)
        # self.assertTrue(
        #     approx_equal(
        #         mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
        #     )
        # )
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_batch_lazy(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=NonLazyTensor(covmat).repeat(2, 1, 1))
        self.assertIsInstance(mvn.covariance_matrix, LazyTensor)
        self.assertTrue(approx_equal(mvn.variance, covmat.diag().repeat(2, 1)))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate()))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate() * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix.evaluate(), mvn.covariance_matrix.evaluate() / 4))
        # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
        # uses using root_decomposition which is not very reliable
        # self.assertTrue(approx_equal(mvn.entropy(), 4.3157 * torch.ones(2)))
        # self.assertTrue(
        #     approx_equal(mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
        # )
        # self.assertTrue(
        #     approx_equal(mvn.log_prob(torch.zeros(2, 2, 3)), -4.8157 * torch.ones(2, 2))
        # )
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_correlated_sampels(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))

        base_samples = mvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 3]))

        base_samples = mvn.get_base_samples()
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3]))

    def test_multivariate_normal_batch_correlated_sampels(self):
        mean = torch.tensor([0, 1, 2], dtype=torch.float)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5]))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=NonLazyTensor(covmat).repeat(2, 1, 1))

        base_samples = mvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 3]))

        base_samples = mvn.get_base_samples()
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([2, 3]))


if __name__ == "__main__":
    unittest.main()
