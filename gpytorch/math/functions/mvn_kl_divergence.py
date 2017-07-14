import math
from torch.autograd import Function
from .invmm import Invmm
from gpytorch.math.functions import ExactGPMarginalLogLikelihood


class MVNKLDivergence(Function):
    """
    PyTorch function for computing the KL-Divergence between two multivariate
    Normal distributions.

    For this function, the first Gaussian distribution is parameterized by the
    mean vector \mu_1 and the Cholesky decomposition of the covariance matrix U_1:
    N(\mu_1, U_1^{\top}U_1).

    The second Gaussian distribution is parameterized by the mean vector \mu_2
    and the full covariance matrix \Sigma_2: N(\mu_2, \Sigma_2)

    The KL divergence between two multivariate Gaussians is given by:

        KL(N_1||N_2) = 0.5 * (Tr(\Sigma_2^{-1}\Sigma_{1}) + (\mu_2 -
            \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) -
            logdet(\Sigma_{1}) - D)

    Where D is the dimensionality of the distributions.
    """
    def __call__(self, mu1_var, chol_covar1_var, mu2_var, covar2_var):
        mu_diffs = mu2_var - mu1_var

        # ExactGPMarginalLogLikelihood gives us -0.5 * [\mu_2 -
        # \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) + const]
        # Multiplying that by -2 gives us two of the terms in the KL divergence
        # (plus an unwanted constant that we can subtract out).
        K_part = ExactGPMarginalLogLikelihood()(covar2_var, mu_diffs)

        # Get logdet(\Sigma_{1})
        log_det_covar1 = chol_covar1_var.diag().log().sum() * 2

        # Get Tr(\Sigma_2^{-1}\Sigma_{1})
        trace = Invmm()(covar2_var, chol_covar1_var.t().mm(chol_covar1_var)).trace()

        # get D
        D = len(mu_diffs)

        # Compute the KL Divergence. We subtract out D * log(2 * pi) to get rid
        # of the extra unwanted constant term that ExactGPMarginalLogLikelihood gives us.
        res = 0.5 * (trace - log_det_covar1 - 2 * K_part - (1 + math.log(2 * math.pi)) * D)

        return res
