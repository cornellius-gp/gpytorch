#!/usr/bin/env python3

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from gpytorch import settings

from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution

from ..distributions import MultivariateNormal
from ..models import ApproximateGP
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.errors import CachingError
from ..utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from . import _VariationalDistribution


def _ensure_updated_strategy_flag_set(
    state_dict: Dict[str, Tensor],
    prefix: str,
    local_metadata: Dict[str, Any],
    strict: bool,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    error_msgs: Iterable[str],
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


class ComputePredictiveUpdates(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        chol: Tensor,
        induc_data_covar: Tensor,
        middle: Tensor,
        inducing_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the predictive mean and variance updates as in `VariationalStrategy._compute_predictive_updates`.

        This function doesn't compute the updates to the off-diagonal entries in the predictive covariance. Only the
        variance update is computed.
        """
        interp_term = torch.linalg.solve_triangular(chol, induc_data_covar, upper=False)

        mean_update = (interp_term.mT @ inducing_values.unsqueeze(-1)).squeeze(-1)
        variance_update = torch.sum(interp_term.mT * (interp_term.mT @ middle), dim=-1)

        # NOTE: The backward call does not need `induc_data_covar`. Access to it is always through `interp_term`.
        ctx.save_for_backward(chol, interp_term, middle, inducing_values)

        return mean_update, variance_update

    @staticmethod
    def backward(
        ctx,
        d_mean: Tensor,
        d_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""A custom backward pass more efficient than PyTorch's autograd by rearranging tensor operations.

        This backward is bottlenecked by two O(m^2 n) matmuls whre `m` is the number of inducing points and `n` is the
        number of data points. In contrast, PyTorch's backward pass would require three O(m^2 n) matmuls and a O(m^2 n)
        triangular solve. Thus, this implementation is about 2x faster when `m << n`.
        """
        chol, interp_term, middle, inducing_values = ctx.saved_tensors

        # Common terms that will be used more than once
        interp_term_times_dmean = interp_term @ d_mean.unsqueeze(-1)
        interp_term_scaled_dvariance = interp_term * d_variance.unsqueeze(-2)  # K_ZZ^{-1/2} K_ZX @ diag(d_variance)

        # `K_ZZ^{-1/2} @ (S - I)`
        # NOTE: Empirically, the triangular solve against `S - I` still seems to be stable in FP32. However, hitting
        # `S - I` twice `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}` would be numerically unstable, which should be avoided.
        inv_chol_times_middle = torch.linalg.solve_triangular(chol.mT, middle, upper=True)

        # `K_ZZ^{-1/2} m`
        inv_chol_times_inducing_values = torch.linalg.solve_triangular(
            chol.mT, inducing_values.unsqueeze(-1), upper=True
        )

        # The derivative of `S - I` from the variance
        d_middle = interp_term_scaled_dvariance @ interp_term.mT

        # The derivative of `K_XZ K_ZZ^{-1/2} m` with respect to `m`
        d_inducing_values = interp_term_times_dmean.squeeze(-1)

        # The derivative of `K_XZ` received from the predictive variance. There is a factor of 2 because `K_XZ` appears
        # twice in the predictive variance and we exploit symmetry.
        d_induc_data_covar = 2.0 * inv_chol_times_middle @ interp_term_scaled_dvariance

        # Then add derivative of `K_XZ` received from the predictive mean: `K_ZZ^{-1/2} @ m @ dm^T`
        d_induc_data_covar = d_induc_data_covar + inv_chol_times_inducing_values @ d_mean.unsqueeze(-2)

        # The derivative of `K_ZZ^{-1/2}` received from the predictive variance. Again, we exploit symmetry here since
        # `K_ZZ^{-1/2}` appears twice.
        d_chol = -2.0 * inv_chol_times_middle @ d_middle

        # Then add the derivative of `K_ZZ^{-1/2}` received from the predictive mean
        d_chol = d_chol - inv_chol_times_inducing_values @ interp_term_times_dmean.mT

        # NOTE: In principle, we need to zero out the lower triangular part because `chol` is lower triangular. It is
        # actually not necessary here, because `d_chol` is immediately fed into `cholesky_backward`, which does not
        # care about the upper triangular part. We keep it here for consistency with PyTorch's implementation.
        # https://github.com/pytorch/pytorch/blob/4a0693682a8574bdc36e1ca2ea7bd2ddf5c19340/torch/csrc/autograd/FunctionsManual.cpp#L1999-L2003
        # NOTE: If we want to get fancy, fusing this backward with `cholesky_backward` will save a matmul. It may not
        # be worth the effort. It's only useful when there are more inducing points than the data.
        d_chol = d_chol.tril()

        return d_chol, d_induc_data_covar, d_middle, d_inducing_values


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

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(self.inducing_points)
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_var_cov.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(Kmm_root.transpose(-1, -2))

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
        try:
            pseudo_target_covar = CholLinearOperator(inducing_covar.add_jitter(self.jitter_val).cholesky()).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val)).matmul(evecs.transpose(-1, -2)).to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean

    def _compute_predictive_updates(
        self,
        chol: LinearOperator,
        induc_data_covar: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None,
        prior_covar: LinearOperator,
        diag: bool = True,
    ) -> tuple[Tensor, LinearOperator]:
        r"""Compute the predictive mean and covariance updates. Adding the return values of this method to the prior
        mean and covariance yields the predictive mean and covariance.

        The predictive mean update is `K_{XZ} K_{ZZ}^{-1/2} m`.

        The predictive covariance update is `K_{XZ} K_{ZZ}^{-1/2} (S - I) K_{ZZ}^{-1/2} K_{ZX}`.

        :param chol: The Cholesky factor `K_{ZZ}^{-1/2}`.
        :param induc_data_covar: The covariance between the inducing points and the data `K_{ZX}`.
        :param inducing_values: The whitened variational mean `m`.
        :param variational_inducing_covar: The variational covariance `S`.
        :param prior_covar: The prior covariance, typically an identity matrix `I`.
        :param diag: If true, this method computes the predictive variance instead of the full covariance in train mode
            if there are more data than inducing points.
        :return: The predictive mean update and the predictive covariance update.
        """
        middle_term = prior_covar.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)  # `S - I`

        # The custom autograd function doesn't compute the off-diagonal entries. Besides, it's only optimized for the
        # setting where the batch size is larger than the number of inducing points.
        if diag and self.training and induc_data_covar.size(-2) < induc_data_covar.size(-1):
            predictive_mean_update, predictive_variance_update = ComputePredictiveUpdates.apply(
                chol.to_dense().type(induc_data_covar.dtype),
                induc_data_covar,
                middle_term.to_dense(),
                inducing_values,
            )
            return predictive_mean_update, DiagLinearOperator(predictive_variance_update)

        # The eval mode uses the same implementation as v1.14.3
        else:
            # NOTE: `torch.linalg.solve_triangular(A, B)` seems to support mixed precision solve when `A` and `B` have
            # different dtypes. `B` is likely cast to the dtype of `A` internally. Thus, there is no need for explicit
            # type casting. Removing the type casting would be slightly faster and avoid memory allocation. However, we
            # keep the explicit type casting here because this behavior is not documented on the PyTorch side.
            interp_term = chol.solve(induc_data_covar.type(chol.dtype))
            interp_term = interp_term.type(induc_data_covar.dtype)

            # Compute the predictive mean update
            # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z)
            predictive_mean_update = (interp_term.mT @ inducing_values.unsqueeze(-1)).squeeze(-1)

            if settings.trace_mode.on():
                middle_term = middle_term.to_dense()

            # Compute the predictive covariance update
            # k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
            predictive_covar_update = MatmulLinearOperator(interp_term.mT, middle_term @ interp_term)

        return predictive_mean_update, predictive_covar_update

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        diag: bool = True,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)

        mean_update, covar_update = self._compute_predictive_updates(
            chol=L,
            induc_data_covar=induc_data_covar,
            inducing_values=inducing_values,
            variational_inducing_covar=variational_inducing_covar,
            prior_covar=self.prior_distribution.lazy_covariance_matrix,
            diag=diag,
        )

        predictive_mean = test_mean + mean_update
        predictive_covar = SumLinearOperator(data_data_covar.add_jitter(self.jitter_val), covar_update)

        if trace_mode.on():
            predictive_covar = predictive_covar.to_dense()

        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x: Tensor, prior: bool = False, diag: bool = True, **kwargs) -> MultivariateNormal:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u). Whitening needs the full covariance.
                prior_function_dist = self(self.inducing_points, prior=True, diag=False)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultivariateNormal):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, diag=diag, **kwargs)
