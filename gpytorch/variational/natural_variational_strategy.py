#!/usr/bin/env python3

import torch

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor, lazify
from ..module import Module
from ..settings import record_ciq_stats
from ..utils import linear_cg
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class _InterpTermsChol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, interp_term, natural_vec, natural_mat, mode):
        # Compute precision
        prec = natural_mat.mul(-2.0)
        diag = prec.diagonal(dim1=-1, dim2=-2).unsqueeze(-1)

        # Compute necessary solves with the precision. We need
        # m = expec_vec = S * natural_vec
        # S K^{-1/2} k
        if mode == "ciq":
            solves = linear_cg(
                prec.matmul,
                torch.cat([natural_vec.unsqueeze(-1), interp_term], dim=-1),
                n_tridiag=0,
                max_iter=1000,
                max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
                preconditioner=lambda x: x / diag,
            )
            expec_vec = solves[..., 0]
            s_times_interp_term = solves[..., 1:]
        else:
            prec_chol = psd_safe_cholesky(prec.double())
            expec_vec = (
                torch.cholesky_solve(natural_vec.unsqueeze(-1).double(), prec_chol).squeeze(-1).to(interp_term.dtype)
            )
            s_times_interp_term = torch.cholesky_solve(interp_term.double(), prec_chol).to(interp_term.dtype)

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
    def backward(ctx, interp_mean_grad, interp_var_grad, kl_div_grad):
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
                (interp_var_grad * interp_mean * interp_term).sum(dim=-1).mul(-2),
                (interp_mean_grad * interp_term).sum(dim=-1),
                (kl_div_grad * natural_vec),
            ]
        )

        # Compute gradient of expected matrix (mm^T + S)
        # interp_mean component: 0
        # interp_var component: K^{-1/2} k k^T K^{-1/2}
        # kl component: 1/2 ( I - S^{-1} )
        eye = torch.eye(expec_vec.size(-1), device=expec_vec.device, dtype=expec_vec.dtype)
        expec_mat_grad = torch.add(
            (interp_var_grad * interp_term) @ interp_term.transpose(-1, -2), (kl_div_grad * (eye - prec).mul(0.5))
        )

        # We're done!
        return interp_term_grad, expec_vec_grad, expec_mat_grad, None  # Extra "None" for the kwarg


class VD(Module):
    def __init__(self, num_induc):
        # We're going to make our own variational distribution
        super().__init__()
        self.register_parameter("natural_vec", torch.nn.Parameter(torch.zeros(num_induc)))
        self.register_parameter("natural_mat", torch.nn.Parameter(torch.ones(num_induc).diag_embed().mul_(-0.5)))


class NaturalVariationalStrategy(_VariationalStrategy):
    def __init__(self, model, inducing_points, mode="standard", learn_inducing_locations=True):
        Module.__init__(self)

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Choose mode for computing the interp term
        self.mode = mode

        self._variational_distribution = VD(inducing_points.size(-2))
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros_like(self._variational_distribution.natural_vec)
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        raise RuntimeError

    def forward(self, x, inducing_points):
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
        if self.mode == "standard":
            L = induc_induc_covar.double().cholesky()
            interp_term = torch.triangular_solve(induc_data_covar.double(), L, upper=False)[0]
            interp_term = interp_term.to(induc_data_covar.dtype)
        elif self.mode == "ciq":
            interp_term = lazify(induc_induc_covar).sqrt_inv_matmul(induc_data_covar)
        elif self.mode == "eig" or self.mode == "eigqr":
            evals, evecs = induc_induc_covar.double().symeig(eigenvectors=True)
            if record_ciq_stats.on():
                record_ciq_stats.condition_number = evals.max().item() / evals.min().item()
                record_ciq_stats.minres_residual = (
                    (
                        (
                            (evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))).to(induc_data_covar.dtype)
                            - induc_induc_covar
                        )
                    )
                    .norm()
                    .div(induc_induc_covar.norm())
                    .item()
                )
            matrix_root = evecs @ (evecs.transpose(-1, -2) / (evals.sqrt().unsqueeze(-1)))
            if self.mode == "eig":
                interp_term = matrix_root.to(induc_data_covar.dtype) @ induc_data_covar
            elif self.mode == "eigqr":
                _, R = matrix_root.qr()
                interp_term = torch.triangular_solve(induc_data_covar.double(), R.transpose(-1, -2), upper=False)[0]
                interp_term = interp_term.to(x.dtype)
        else:
            raise RuntimeError

        # Compute interpolated mean and variance terms
        interp_mean, interp_var, kl_div = _InterpTermsChol().apply(
            interp_term,
            self._variational_distribution.natural_vec,
            self._variational_distribution.natural_mat,
            self.mode,
        )

        # Memoize the logdet
        if not hasattr(self, "_memoize_cache"):
            self._memoize_cache = dict()
        self._memoize_cache["kl"] = kl_div

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = interp_mean + test_mean

        # Compute the covariance of q(f)
        predictive_var = (data_data_covar.diag() - interp_term.pow(2).sum(dim=-2) + interp_var).clamp_min(1e-10)
        if record_ciq_stats.on():
            record_ciq_stats.min_var = predictive_var.min().item()
        predictive_covar = DiagLazyTensor(predictive_var)

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        """
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        if hasattr(self, "_memoize_cache") and "kl" in self._memoize_cache:
            return self._memoize_cache["kl"]
        else:
            raise RuntimeError

    def __call__(self, x, prior=False):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            noise = torch.randn_like(self.prior_distribution.mean).mul_(1e-3)
            eye = torch.eye(noise.size(-1), dtype=noise.dtype, device=noise.device).mul(-0.5)
            self._variational_distribution.natural_vec.data.copy_(noise)
            self._variational_distribution.natural_mat.data.copy_(eye)
            self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            x = x.expand(*batch_shape, *x.shape[-2:])

        return Module.__call__(self, x, inducing_points)
