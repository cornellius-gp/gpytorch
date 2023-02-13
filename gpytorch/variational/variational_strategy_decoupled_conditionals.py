#!/usr/bin/env python3

import warnings
import torch
import pickle as pkl
from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.cholesky import psd_safe_cholesky
from ..utils.errors import CachingError
from ..utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from ._variational_strategy import _VariationalStrategy


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


class VariationalStrategyDecoupledConditionals(_VariationalStrategy):
    r"""
    The variational strategy using decoupled conditionals.
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
    :param ~gpytorch.kernels.Kernel: A kernel module for the predictive mean only.
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    """

    def __init__(self, model, inducing_points, variational_distribution, covar_module_mean, learn_inducing_locations=True):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.covar_module_mean = covar_module_mean

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLazyTensor(L)

    @cached(name="cholesky_factor_mean", ignore_args=True)
    def _cholesky_factor_mean(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLazyTensor(L)


    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model.forward(self.inducing_points)
        induc_induc_covar = out.lazy_covariance_matrix.add_jitter()
        kernel = self.covar_module_mean
        induc_induc_covar_mean = kernel(self.inducing_points).add_jitter()
        L = self._cholesky_factor(induc_induc_covar).evaluate()
        L_mean = self._cholesky_factor_mean(induc_induc_covar_mean)
        Lbar = L_mean.inv_matmul(L).to(induc_induc_covar_mean.dtype)
        # Kbar = CholLazyTensor(Lbar)
        Kbar = RootLazyTensor(Lbar)
        res = MultivariateNormal(out.mean, Kbar)
        return res

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        m = self.variational_distribution.loc
        L_s = self.variational_distribution.lazy_covariance_matrix.root
        out = self.model.forward(self.inducing_points)
        m_p = out.mean
    
        out = self.model.forward(self.inducing_points)
        induc_induc_covar = out.lazy_covariance_matrix.add_jitter()
        kernel = self.covar_module_mean
        induc_induc_covar_mean = kernel(self.inducing_points).add_jitter()
        L = self._cholesky_factor(induc_induc_covar)
        L_mean = self._cholesky_factor_mean(induc_induc_covar_mean)

        logdet_term = L.diag().log().sum() - L_mean.diag().log().sum() - self.variational_distribution.lazy_covariance_matrix.logdet()/2
        trace_term = L.inv_matmul(L_mean.evaluate() @ L_s.evaluate().type(_linalg_dtype_cholesky.value())).to(self.inducing_points.dtype).square().sum()
        Lm = (L_mean @ (m-m_p).to(dtype=L_mean.dtype)).reshape(-1,1)
        quad_term_half = L.inv_matmul(Lm.type(_linalg_dtype_cholesky.value())).to(self.inducing_points.dtype)
        quad_term = torch.norm(quad_term_half).square()
        
        res = logdet_term + (trace_term + quad_term - L_s.shape[0])/2
        return res


    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
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

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            # rename it to _cholesky_factor_covar?
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
        

        # Compute the mean of q(f)
        # Q_XZ Q_ZZ^{-1/2} (m - Q_ZZ^{-1/2} \mu_Z) + \mu_X
        kernel = self.covar_module_mean
        induc_data_covar_mean = kernel(inducing_points, x).evaluate()
        induc_induc_covar_mean = kernel(inducing_points, inducing_points)
        induc_induc_covar_mean = induc_induc_covar_mean + torch.diag(1e-4*torch.ones(num_induc)).to(induc_induc_covar_mean.device)

        L_mean = self._cholesky_factor_mean(induc_induc_covar_mean)
        if L_mean.shape != induc_induc_covar_mean.shape:
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor_mean")
            except CachingError:
                pass
            L_mean = self._cholesky_factor_mean(induc_induc_covar_mean)
        interp_term_mean = L_mean.inv_matmul(induc_data_covar_mean.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
        predictive_mean = (interp_term_mean.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
        
        
        # Compute the covariance of q(f)
        # K_XX + Q_XZ Q_ZZ^{-T/2} S Q_ZZ^{-1/2} Q_ZX - k_XZ K_ZZ^{-T/2} I K_ZZ^{-1/2} K_ZX

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                - interp_term.transpose(-1, -2) @ interp_term 
                + interp_term_mean.transpose(-1, -2) @ variational_inducing_covar.evaluate() @ interp_term_mean 
            )
        else:
            predictive_covar = SumLazyTensor(
                SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(-interp_term.transpose(-1, -2), interp_term),
                ), 
                MatmulLazyTensor(interp_term_mean.transpose(-1, -2), variational_inducing_covar @ interp_term_mean)   
                )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False, **kwargs):
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
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                whitened_mean = L.inv_matmul(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()
                covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                whitened_covar = RootLazyTensor(L.inv_matmul(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
