import math
import torch
import torch.distributions as dist
from torch.distributions.multivariate_normal import _batch_mahalanobis


def _batch_mv(bmat, bvec):
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


class MartinDist(object):
    def __init__(self, mu, L, nu):
        self.dim = mu.size(-1)
        assert L.size(-1) == self.dim
        assert L.size(-2) == self.dim
        assert nu.size(-1) == self.dim
        self.mu = mu
        self.L = L
        self.nu = nu
        self.half_nu = 0.5 * nu
        self.gamma_dist = dist.Gamma(self.half_nu, self.half_nu)

    def rsample(self, sample_shape=()):
        tau = self.gamma_dist.rsample(sample_shape=sample_shape)
        z = torch.randn(sample_shape + (self.dim,)) / tau.sqrt()
        return _batch_mv(self.L, z) + self.mu

    def entropy(self):
        logdet = torch.diagonal(self.L, dim1=-1, dim2=-2).log().sum()
        half_nu_one, half_nu = self.half_nu + 0.5, self.half_nu
        entropy1 = half_nu_one * (torch.digamma(half_nu_one) - torch.digamma(half_nu))
        entropy2 = torch.lgamma(half_nu) - torch.lgamma(half_nu_one)
        entropy3 = 0.5 * torch.log(math.pi * self.nu)
        return logdet + (entropy1 + entropy2 + entropy3).sum()

    def KL(self, mvn):
        assert type(mvn) == dist.MultivariateNormal
        mean_term = 0.5 * _batch_mahalanobis(mvn.scale_tril, mvn.loc - self.mu).sum()
        log_det = torch.diagonal(mvn.scale_tril, dim1=-1, dim2=-2).log().sum()
        constant = 0.5 * math.log(2.0 * math.pi) * self.mu.numel()
        nu_diag = (self.nu / (self.nu - 2.0)).sqrt().unsqueeze(-2)
        LinvL = torch.triangular_solve(self.L, mvn.scale_tril, upper=False)[0] * nu_diag
        trace_term = 0.5 * LinvL.pow(2.0).sum()
        return -self.entropy() + mean_term + log_det + constant + trace_term

"""
some basic tests
"""
if __name__ == "__main__":

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
            d = MartinDist(mu, L, nu)
            d_large_nu = MartinDist(mu, L, 2000.0 * nu)
            mvn = dist.MultivariateNormal(mu, scale_tril=L)
            mvnp = dist.MultivariateNormal(mup, Lp)

            z = d.rsample()
            assert z.shape == full_sample_shape

            entropy = d.entropy()
            entropy_large_nu = d_large_nu.entropy().item()
            entropy_mvn = mvn.entropy().sum().item()
            assert math.fabs(entropy_large_nu - entropy_mvn) < 0.03

            kl_mvn = d.KL(mvn).item()
            assert kl_mvn > 0.10

            kl_mvn_large_nu = d_large_nu.KL(mvn).item()
            assert kl_mvn_large_nu < 0.03 and kl_mvn_large_nu > -1.0e-2

            kl_mvnp = d.KL(mvnp).item()
            assert kl_mvnp > 0.2

            if B is not None:
                L = L.unsqueeze(0).expand((B,) + L.shape)
                mu = mu.unsqueeze(0).expand((B,) + mu.shape)
                nu = nu.unsqueeze(0).expand((B,) + nu.shape)
                full_sample_shape = (B,) + full_sample_shape

                Lp = Lp.unsqueeze(0).expand((B,) + Lp.shape)
                mup = mup.unsqueeze(0).expand((B,) + mup.shape)

    print("Yay! Tests passed!")
