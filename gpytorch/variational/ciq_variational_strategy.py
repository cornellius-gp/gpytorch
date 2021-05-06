#!/usr/bin/env python3

from typing import Optional, Tuple

import torch

from .. import settings
from ..distributions import Delta, MultivariateNormal
from ..lazy import DiagLazyTensor, MatmulLazyTensor, SumLazyTensor, lazify
from ..module import Module
from ..utils import linear_cg
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy
from .natural_variational_distribution import NaturalVariationalDistribution


class _NgdInterpTerms(torch.autograd.Function):
    """
    This function takes in

        - the kernel interpolation term K_ZZ^{-1/2} k_ZX
        - the natural parameters of the variational distribution

    and returns

        - the predictive distribution mean/covariance
        - the inducing KL divergence KL( q(u) || p(u))

    However, the gradients will be with respect to the **cannonical parameters**
    of the variational distribution, rather than the **natural parameters**.
    This corresponds to performing natural gradient descent on the variational distribution.
    """

    @staticmethod
    def forward(
        ctx, interp_term: torch.Tensor, natural_vec: torch.Tensor, natural_mat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute precision
        prec = natural_mat.mul(-2.0)
        diag = prec.diagonal(dim1=-1, dim2=-2).unsqueeze(-1)

        # Make sure that interp_term and natural_vec are the same batch shape
        batch_shape = _mul_broadcast_shape(interp_term.shape[:-2], natural_vec.shape[:-1])
        expanded_interp_term = interp_term.expand(*batch_shape, *interp_term.shape[-2:])
        expanded_natural_vec = natural_vec.expand(*batch_shape, natural_vec.size(-1))

        # Compute necessary solves with the precision. We need
        # m = expec_vec = S * natural_vec
        # S K^{-1/2} k
        solves = linear_cg(
            prec.matmul,
            torch.cat([expanded_natural_vec.unsqueeze(-1), expanded_interp_term], dim=-1),
            n_tridiag=0,
            max_iter=settings.max_cg_iterations.value(),
            tolerance=min(settings.eval_cg_tolerance.value(), settings.cg_tolerance.value()),
            max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=lambda x: x / diag,
        )
        expec_vec = solves[..., 0]
        s_times_interp_term = solves[..., 1:]

        # Compute the interpolated mean
        # k^T K^{-1/2} m
        interp_mean = (s_times_interp_term.transpose(-1, -2) @ natural_vec.unsqueeze(-1)).squeeze(-1)

        # Compute the interpolated variance
        # k^T K^{-1/2} S K^{-1/2} k = k^T K^{-1/2} (expec_mat - expec_vec expec_vec^T) K^{-1/2} k
        interp_var = (s_times_interp_term * interp_term).sum(dim=-2)

        # Let's not bother actually computing the KL-div in the foward pass
        # 1/2 ( -log | S | + tr(S) + m^T m - len(m) )
        # = 1/2 ( -log | expec_mat - expec_vec expec_vec^T | + tr(expec_mat) - len(m) )
        kl_div = torch.zeros_like(interp_mean[..., 0])

        # We're done!
        ctx.save_for_backward(interp_term, s_times_interp_term, interp_mean, natural_vec, expec_vec, prec)
        return interp_mean, interp_var, kl_div

    @staticmethod
    def backward(
        ctx, interp_mean_grad: torch.Tensor, interp_var_grad: torch.Tensor, kl_div_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the saved terms
        interp_term, s_times_interp_term, interp_mean, natural_vec, expec_vec, prec = ctx.saved_tensors

        # Expand data-depenedent gradients
        interp_mean_grad = interp_mean_grad.unsqueeze(-2)
        interp_var_grad = interp_var_grad.unsqueeze(-2)

        # Compute gradient of interp term (K^{-1/2} k)
        # interp_mean component: m
        # interp_var component: S K^{-1/2} k
        # kl component: 0
        interp_term_grad = (interp_var_grad * s_times_interp_term).mul(2.0) + (
            interp_mean_grad * expec_vec.unsqueeze(-1)
        )

        # Compute gradient of expected vector (m)
        # interp_mean component: K^{-1/2} k
        # interp_var component: (k^T K^{-1/2} m) K^{-1/2} k
        # kl component: S^{-1} m
        expec_vec_grad = sum(
            [
                (interp_var_grad * interp_mean.unsqueeze(-2) * interp_term).sum(dim=-1).mul(-2),
                (interp_mean_grad * interp_term).sum(dim=-1),
                (kl_div_grad.unsqueeze(-1) * natural_vec),
            ]
        )

        # Compute gradient of expected matrix (mm^T + S)
        # interp_mean component: 0
        # interp_var component: K^{-1/2} k k^T K^{-1/2}
        # kl component: 1/2 ( I - S^{-1} )
        eye = torch.eye(expec_vec.size(-1), device=expec_vec.device, dtype=expec_vec.dtype)
        expec_mat_grad = torch.add(
            (interp_var_grad * interp_term) @ interp_term.transpose(-1, -2),
            (kl_div_grad.unsqueeze(-1).unsqueeze(-1) * (eye - prec).mul(0.5)),
        )

        # We're done!
        return interp_term_grad, expec_vec_grad, expec_mat_grad, None  # Extra "None" for the kwarg


class CiqVariationalStrategy(_VariationalStrategy):
    r"""
    Similar to :class:`~gpytorch.variational.VariationalStrategy`,
    except the whitening operation is performed using Contour Integral Quadrature
    rather than Cholesky (see `Pleiss et al. (2020)`_ for more info).
    See the `CIQ-SVGP tutorial`_ for an example.

    Contour Integral Quadrature uses iterative matrix-vector multiplication to approximate
    the :math:`\mathbf K_{\mathbf Z \mathbf Z}^{-1/2}` matrix used for the whitening operation.
    This can be more efficient than the standard variational strategy for large numbers
    of inducing points (e.g. :math:`M > 1000`) or when the inducing points have structure
    (e.g. they lie on an evenly-spaced grid).

    .. note::

        It is recommended that this object is used in conjunction with
        :obj:`~gpytorch.variational.NaturalVariationalDistribution` and
        `natural gradient descent`_.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    .. _Pleiss et al. (2020):
        https://arxiv.org/pdf/2006.11267.pdf
    .. _CIQ-SVGP tutorial:
        examples/04_Variational_and_Approximate_GPs/SVGP_CIQ.html
    .. _natural gradient descent:
        examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html
    """

    def _ngd(self):
        return isinstance(self._variational_distribution, NaturalVariationalDistribution)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        if self._ngd():
            raise RuntimeError(
                "Variational distribution for NGD-CIQ should be computed during forward calls. "
                "This is probably a bug in GPyTorch."
            )
        return super().variational_distribution

    def forward(
        self,
        x: torch.Tensor,
        inducing_points: torch.Tensor,
        inducing_values: torch.Tensor,
        variational_inducing_covar: Optional[MultivariateNormal] = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter(1e-2)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:].add_jitter(1e-4)

        # Compute interpolation terms
        # K_XZ K_ZZ^{-1} \mu_z
        # K_XZ K_ZZ^{-1/2} \mu_Z
        with settings.max_preconditioner_size(0):  # Turn off preconditioning for CIQ
            interp_term = lazify(induc_induc_covar).sqrt_inv_matmul(induc_data_covar)

        # Compute interpolated mean and variance terms
        # We have separate computation rules for NGD versus standard GD
        if self._ngd():
            interp_mean, interp_var, kl_div = _NgdInterpTerms.apply(
                interp_term, self._variational_distribution.natural_vec, self._variational_distribution.natural_mat,
            )

            # Compute the covariance of q(f)
            predictive_var = data_data_covar.diag() - interp_term.pow(2).sum(dim=-2) + interp_var
            predictive_var = torch.clamp_min(predictive_var, settings.min_variance.value(predictive_var.dtype))
            predictive_covar = DiagLazyTensor(predictive_var)

            # Also compute and cache the KL divergence
            if not hasattr(self, "_memoize_cache"):
                self._memoize_cache = dict()
            self._memoize_cache["kl"] = kl_div

        else:
            # Compute interpolated mean term
            interp_mean = torch.matmul(
                interp_term.transpose(-1, -2), (inducing_values - self.prior_distribution.mean).unsqueeze(-1)
            ).squeeze(-1)

            # Compute the covariance of q(f)
            middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
            if variational_inducing_covar is not None:
                middle_term = SumLazyTensor(variational_inducing_covar, middle_term)
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = interp_mean + test_mean

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        """
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        if self._ngd():
            if hasattr(self, "_memoize_cache") and "kl" in self._memoize_cache:
                return self._memoize_cache["kl"]
            else:
                raise RuntimeError(
                    "KL divergence for NGD-CIQ should be computed during forward calls."
                    "This is probably a bug in GPyTorch."
                )
        else:
            return super().kl_divergence()

    def __call__(self, x: torch.Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        # This is mostly the same as _VariationalStrategy.__call__()
        # but with special rules for natural gradient descent (to prevent O(M^3) computation)

        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            if self._ngd():
                noise = torch.randn_like(self.prior_distribution.mean).mul_(1e-3)
                eye = torch.eye(noise.size(-1), dtype=noise.dtype, device=noise.device).mul(-0.5)
                self._variational_distribution.natural_vec.data.copy_(noise)
                self._variational_distribution.natural_mat.data.copy_(eye)
                self.variational_params_initialized.fill_(1)
            else:
                prior_dist = self.prior_distribution
                self._variational_distribution.initialize_variational_distribution(prior_dist)
                self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get q(f)
        if self._ngd():
            return Module.__call__(
                self, x, inducing_points, inducing_values=None, variational_inducing_covar=None, **kwargs,
            )
        else:
            # Get p(u)/q(u)
            variational_dist_u = self.variational_distribution

            if isinstance(variational_dist_u, MultivariateNormal):
                return Module.__call__(
                    self,
                    x,
                    inducing_points,
                    inducing_values=variational_dist_u.mean,
                    variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                    **kwargs,
                )
            elif isinstance(variational_dist_u, Delta):
                return Module.__call__(
                    self,
                    x,
                    inducing_points,
                    inducing_values=variational_dist_u.mean,
                    variational_inducing_covar=None,
                    ngd=False,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    f"Invalid variational distribuition ({type(variational_dist_u)}). "
                    "Expected a multivariate normal or a delta distribution."
                )
