#!/usr/bin/env python3

import warnings

import torch

from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, delazify
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class OldVersionWarning(RuntimeWarning):
    pass


def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


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

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

    @cached(name="cholesky_factor")
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double())
        return L

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros_like(self.variational_distribution.mean)
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    @cached(name="mean_covar_cache_memo")
    def mean_covar_cache(self, variational_mean, variational_covar):
        """
        Computes K_{uu}^{-1/2}m and K_{uu}^{-1/2}(I - LL')K_{uu}^{-1/2} using contour integral quadrature.
        """
        prior_dist = self.prior_distribution

        induc_induc_covar = prior_dist.lazy_covariance_matrix

        L = self._cholesky_factor(induc_induc_covar)

        device = induc_induc_covar.device
        dtype = induc_induc_covar.dtype
        mat_len = induc_induc_covar.matrix_shape[0]
        batch_shape = induc_induc_covar.batch_shape

        eye = DiagLazyTensor(torch.ones(*batch_shape, mat_len, dtype=dtype, device=device))

        inner_mat = (eye.mul(-1) + variational_covar).evaluate()

        right_rinv = torch.triangular_solve(inner_mat.double(), L.transpose(-2, -1).double(), upper=True)[0].transpose(-2, -1)

        var_mean = variational_mean - prior_dist.mean

        right_hand_sides = torch.cat((var_mean.unsqueeze(-1).double(), right_rinv), dim=-1)

        full_rinv, _ = torch.triangular_solve(right_hand_sides, L.transpose(-2, -1), upper=True)
        print(full_rinv.dtype)

        mean_cache = full_rinv[..., :, 0].contiguous().to(dtype=variational_mean.dtype)
        covar_cache = full_rinv[..., :, 1:].contiguous().to(dtype=variational_mean.dtype)

        return mean_cache, covar_cache

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        mean_cache, covar_cache = self.mean_covar_cache(inducing_values, variational_inducing_covar)

        predictive_mean = (
            torch.matmul(
                induc_data_covar.transpose(-1, -2), mean_cache.unsqueeze(-1)
            ).squeeze(-1)
            + test_mean
        )

        left_part = induc_data_covar.transpose(-2, -1).matmul(covar_cache)
        full_part = MatmulLazyTensor(left_part, induc_data_covar)
        predictive_covar = data_data_covar + full_part

        if self.training:
            predictive_covar = DiagLazyTensor(predictive_covar.diag())

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False):
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                whitened_mean = (
                    torch.triangular_solve((variational_dist.loc - prior_mean).unsqueeze(-1).double(), L, upper=False)[
                        0
                    ]
                    .squeeze(-1)
                    .to(variational_dist.loc.dtype)
                )
                whitened_covar = RootLazyTensor(
                    torch.triangular_solve(
                        variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate().double(),
                        L,
                        upper=False,
                    )[0].to(variational_dist.loc.dtype)
                )
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                if hasattr(self, "_memoize_cache"):
                    delattr(self, "_memoize_cache")
                    self._memoize_cache = dict()

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior)
