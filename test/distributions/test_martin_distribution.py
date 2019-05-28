#!/usr/bin/env python3

import math
import unittest
from test._utils import least_used_cuda_device
from test._base_test_case import BaseTestCase

import torch
from gpytorch.distributions import MartinDistribution, MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, LazyTensor, NonLazyTensor, CholLazyTensor


class TestMultivariateNormal(BaseTestCase, unittest.TestCase):
    seed = 1

    def test_kl_divergence(self):
        for _ in range(5):
            L = torch.diag(0.5 * torch.randn(2).exp())
            L[1, 0] = 0.1 * torch.randn(1).item()
            mu = torch.randn(2)
            nu = torch.rand(2) + 2.3

            Lp = torch.diag(0.5 * torch.randn(2).exp())
            Lp[1, 0] = 0.1 * torch.randn(1).item()
            mup = torch.randn(2)

            full_sample_shape = (2,)

            for B in [3, 4, None]:
                d = MartinDistribution(mu, CholLazyTensor(L), nu)
                d_large_nu = MartinDistribution(mu, CholLazyTensor(L), 2000.0 * nu)
                mvn = MultivariateNormal(mu, CholLazyTensor(L))
                mvnp = MultivariateNormal(mup, CholLazyTensor(Lp))

            #z = d.rsample()
            #assert z.shape == full_sample_shape

            #entropy = d.entropy()
            #entropy_large_nu = d_large_nu.entropy().item()
            #entropy_mvn = mvn.entropy().sum().item()
            #assert math.fabs(entropy_large_nu - entropy_mvn) < 0.03

            kl_mvn = torch.distributions.kl.kl_divergence(d, mvn).item()
            assert kl_mvn > 0.10

            kl_mvn_large_nu = torch.distributions.kl.kl_divergence(d_large_nu, mvn).item()
            assert kl_mvn_large_nu < 0.03 and kl_mvn_large_nu > -1.0e-2

            kl_mvnp = torch.distributions.kl.kl_divergence(d, mvnp).item()
            assert kl_mvnp > 0.2

            if B is not None:
                L = L.unsqueeze(0).expand((B,) + L.shape)
                mu = mu.unsqueeze(0).expand((B,) + mu.shape)
                nu = nu.unsqueeze(0).expand((B,) + nu.shape)
                full_sample_shape = (B,) + full_sample_shape

                Lp = Lp.unsqueeze(0).expand((B,) + Lp.shape)
                mup = mup.unsqueeze(0).expand((B,) + mup.shape)


if __name__ == "__main__":
    unittest.main()
