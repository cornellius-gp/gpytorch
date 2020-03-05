#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor, delazify
from ..settings import record_ciq_stats, trace_mode
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class VariationalStrategy(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param bool learn_inducing_points: (optional, default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros_like(self.variational_distribution.mean)
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        if variational_inducing_covar is not None:
            raise RuntimeError("VariationalStrategy currently does not handle the variational_inducing_covar")

        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = delazify(full_covar[..., :num_induc, :num_induc].add_jitter())
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:].add_jitter(1e-4)

        # Error out if we encounter NaNs
        if not torch.equal(induc_induc_covar, induc_induc_covar):
            raise RuntimeError("NaN encountered in K_ZZ matrix")

        # Compute interpolation terms
        # K_XZ K_ZZ^{-1} \mu_z
        # K_XZ K_ZZ^{-1/2} \mu_Z
        L = induc_induc_covar.double().cholesky()
        interp_term = torch.triangular_solve(induc_data_covar.double(), L, upper=False)[0]
        interp_term = interp_term.to(induc_data_covar.dtype)
        interp_mean = torch.matmul(
            interp_term.transpose(-1, -2), (inducing_values - self.prior_distribution.mean).unsqueeze(-1)
        ).squeeze(-1)
        interp_var = interp_term.pow(2).sum(dim=-2)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = interp_mean.squeeze(-1) + test_mean

        if trace_mode.on():
            predictive_covar = data_data_covar.evaluate() - interp_var.diag_embed(dim1=-1, dim2=-2)
        else:
            predictive_var = (data_data_covar.diag() - interp_var).clamp_min(1e-10)
            if record_ciq_stats.on():
                record_ciq_stats.min_var = predictive_var.min().item()
            predictive_covar = DiagLazyTensor(predictive_var)

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
