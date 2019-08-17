import unittest
import torch
import numpy as np

from spectralgp.samplers import MeanEllipticalSlice

class TestMeanEllipticalSlice(unittest.TestCase):
    def test_m_ess(self, nsamples=10000):
        pmean = torch.zeros(2)
        pmean[0] = -2.
        prior_dist = torch.distributions.MultivariateNormal(pmean, covariance_matrix=torch.eye(2))
        
        lmean = torch.zeros(2)
        lmean[0] = 2.
        likelihood = torch.distributions.MultivariateNormal(lmean, covariance_matrix=torch.eye(2))

        prior_inv = torch.inverse(prior_dist.covariance_matrix)
        lik_inv = torch.inverse(likelihood.covariance_matrix)

        true_postsigma = torch.inverse(prior_inv + lik_inv)
        true_postmu = true_postsigma.matmul(prior_inv.matmul(pmean) + lik_inv.matmul(lmean))

        def lfn(x):
            lmean = torch.zeros(2)
            lmean[0] = 2.
            likelihood = torch.distributions.MultivariateNormal(lmean, covariance_matrix=torch.eye(2))
            return likelihood.log_prob(x)

        #lfn = lambda x: likelihood.log_prob(x)

        init = torch.zeros(2)

        m_ess_runner = MeanEllipticalSlice(init, prior_dist, lfn, nsamples)
        samples, _ = m_ess_runner.run()
        samples = samples.numpy()
        samples = samples[:, int(nsamples/2):]

        est_mean = np.mean(samples,1)
        print(est_mean)
        est_cov = np.cov(samples)

        print(np.linalg.norm(est_mean - true_postmu.numpy()))
        print(np.linalg.norm(est_cov - true_postsigma.numpy()))

        # import matplotlib.pyplot as plt
        # N = 60
        # X = np.linspace(-3, 3, N)
        # Y = np.linspace(-3, 4, N)
        # X, Y = np.meshgrid(X, Y)
        # # Pack X and Y into a single 3-dimensional array
        # pos = np.empty(X.shape + (2,))
        # pos[:, :, 0] = X
        # pos[:, :, 1] = Y
        # pos = torch.tensor(pos).float()
        # posterior_dist = torch.distributions.MultivariateNormal(true_postmu, true_postsigma)
        # Z = posterior_dist.log_prob(pos).numpy()

        # plt.contourf(X, Y, Z)
        # plt.scatter(samples[0,:], samples[1,:], color='black', alpha = 0.3)
        # plt.show()



if __name__ == "__main__":
    unittest.main()

