#!/usr/bin/env python3

import warnings

import torch
from torch.autograd import Function

from .. import settings
from ..utils.lanczos import lanczos_tridiag_to_diag
from ..utils.stochastic_lq import StochasticLQ


class InvQuadLogdet(Function):
    """
    Given a PSD matrix A (or a batch of PSD matrices A), this function computes one or both
    of the following
    - The matrix solves A^{-1} b
    - logdet(A)

    This function uses preconditioned CG and Lanczos quadrature to compute the inverse quadratic
    and log determinant terms, using the variance reduction strategy outlined in:
    ``Reducing the Variance of Gaussian Process Hyperparameter Optimization with Preconditioning''
    (https://arxiv.org/abs/2107.00243)
    """

    @staticmethod
    def forward(
        ctx,
        representation_tree,
        precond_representation_tree,
        preconditioner,
        num_precond_args,
        inv_quad,
        probe_vectors,
        probe_vector_norms,
        *args,
    ):
        """
        *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
        If self.inv_quad is true, the first entry in *args is inv_quad_rhs (Tensor)
        - the RHS of the matrix solves.

        Returns:
        - (Scalar) The inverse quadratic form (or None, if self.inv_quad is False)
        - (Scalar) The log determinant (or None, self.if logdet is False)
        """

        ctx.representation_tree = representation_tree
        ctx.precond_representation_tree = precond_representation_tree
        ctx.preconditioner = preconditioner
        ctx.inv_quad = inv_quad
        ctx.num_precond_args = num_precond_args

        matrix_args = None
        precond_args = tuple()
        inv_quad_rhs = None
        if ctx.inv_quad:
            inv_quad_rhs = args[0]
            args = args[1:]
        if ctx.num_precond_args:
            matrix_args = args[:-num_precond_args]
            precond_args = args[-num_precond_args:]
        else:
            matrix_args = args

        # Get closure for matmul
        lazy_tsr = ctx.representation_tree(*matrix_args)
        precond_lt = ctx.precond_representation_tree(*precond_args)

        # Get info about matrix
        ctx.dtype = lazy_tsr.dtype
        ctx.device = lazy_tsr.device
        ctx.matrix_shape = lazy_tsr.matrix_shape
        ctx.batch_shape = lazy_tsr.batch_shape

        # Probe vectors
        if probe_vectors is None or probe_vector_norms is None:
            num_random_probes = settings.num_trace_samples.value()
            if settings.deterministic_probes.on():
                # NOTE: calling precond_lt.root_decomposition() is expensive
                # because it requires Lanczos
                # We don't have any other choice for when we want to use deterministic probes, however
                if precond_lt.size()[-2:] == torch.Size([1, 1]):
                    covar_root = precond_lt.evaluate().sqrt()
                else:
                    covar_root = precond_lt.root_decomposition().root

                warnings.warn(
                    "The deterministic probes feature is now deprecated. "
                    "See https://github.com/cornellius-gp/gpytorch/pull/1836.",
                    DeprecationWarning,
                )
                base_samples = settings.deterministic_probes.probe_vectors
                if base_samples is None or covar_root.size(-1) != base_samples.size(-2):
                    base_samples = torch.randn(
                        *precond_lt.batch_shape,
                        covar_root.size(-1),
                        num_random_probes,
                        dtype=precond_lt.dtype,
                        device=precond_lt.device,
                    )
                    settings.deterministic_probes.probe_vectors = base_samples

                probe_vectors = covar_root.matmul(base_samples).permute(-1, *range(precond_lt.dim() - 1))
            else:
                probe_vectors = precond_lt.zero_mean_mvn_samples(num_random_probes)
            probe_vectors = probe_vectors.unsqueeze(-2).transpose(0, -2).squeeze(0).transpose(-2, -1).contiguous()
            probe_vector_norms = torch.norm(probe_vectors, p=2, dim=-2, keepdim=True)
            probe_vectors = probe_vectors.div(probe_vector_norms)

        # Probe vectors
        ctx.probe_vectors = probe_vectors
        ctx.probe_vector_norms = probe_vector_norms

        # Collect terms for LinearCG
        # We use LinearCG for both matrix solves and for stochastically estimating the log det
        rhs_list = [ctx.probe_vectors]
        num_random_probes = ctx.probe_vectors.size(-1)
        num_inv_quad_solves = 0

        # RHS for inv_quad
        ctx.is_vector = False
        if ctx.inv_quad:
            if inv_quad_rhs.ndimension() == 1:
                inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
                ctx.is_vector = True
            rhs_list.append(inv_quad_rhs)
            num_inv_quad_solves = inv_quad_rhs.size(-1)

        # Perform solves (for inv_quad) and tridiagonalization (for estimating logdet)
        rhs = torch.cat(rhs_list, -1)
        solves, t_mat = lazy_tsr._solve(rhs, preconditioner, num_tridiag=num_random_probes)

        # Final values to return
        logdet_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)
        inv_quad_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)

        # Compute logdet from tridiagonalization
        if settings.skip_logdet_forward.off():
            if torch.any(torch.isnan(t_mat)).item():
                logdet_term = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
            else:
                if ctx.batch_shape is None:
                    t_mat = t_mat.unsqueeze(1)
                eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)
                slq = StochasticLQ()
                (logdet_term,) = slq.evaluate(ctx.matrix_shape, eigenvalues, eigenvectors, [lambda x: x.log()])

        # Extract inv_quad solves from all the solves
        if ctx.inv_quad:
            inv_quad_solves = solves.narrow(-1, num_random_probes, num_inv_quad_solves)
            inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        ctx.num_random_probes = num_random_probes
        ctx.num_inv_quad_solves = num_inv_quad_solves

        to_save = list(precond_args) + list(matrix_args) + [solves]
        ctx.save_for_backward(*to_save)

        return inv_quad_term, logdet_term

    @staticmethod
    def backward(ctx, inv_quad_grad_output, logdet_grad_output):
        # Get input arguments, and get gradients in the proper form
        if ctx.num_precond_args:
            precond_args = ctx.saved_tensors[: ctx.num_precond_args]
            matrix_args = ctx.saved_tensors[ctx.num_precond_args : -1]
        else:
            precond_args = []
            matrix_args = ctx.saved_tensors[:-1]
        solves = ctx.saved_tensors[-1]

        lazy_tsr = ctx.representation_tree(*matrix_args)
        precond_lt = ctx.precond_representation_tree(*precond_args)

        # Fix grad_output sizes
        if ctx.inv_quad:
            inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-2)
        logdet_grad_output = logdet_grad_output.unsqueeze(-1)
        logdet_grad_output.unsqueeze_(-1)

        # Un-normalize probe vector solves
        coef = 1.0 / ctx.probe_vectors.size(-1)
        probe_vector_solves = solves.narrow(-1, 0, ctx.num_random_probes).mul(coef)
        probe_vector_solves.mul_(ctx.probe_vector_norms).mul_(logdet_grad_output)

        # Apply preconditioner to probe vectors (originally drawn from N(0, P))
        # Now the probe vectors will be drawn from N(0, P^{-1})
        if ctx.preconditioner is not None:
            precond_probe_vectors = ctx.preconditioner(ctx.probe_vectors * ctx.probe_vector_norms)
        else:
            precond_probe_vectors = ctx.probe_vectors * ctx.probe_vector_norms

        # matrix gradient
        # Collect terms for arg grads
        left_factors_list = [probe_vector_solves]
        right_factors_list = [precond_probe_vectors]

        inv_quad_solves = None
        neg_inv_quad_solves_times_grad_out = None
        if ctx.inv_quad:
            inv_quad_solves = solves.narrow(-1, ctx.num_random_probes, ctx.num_inv_quad_solves)
            neg_inv_quad_solves_times_grad_out = inv_quad_solves.mul(inv_quad_grad_output).mul_(-1)
            left_factors_list.append(neg_inv_quad_solves_times_grad_out)
            right_factors_list.append(inv_quad_solves)

        left_factors = torch.cat(left_factors_list, -1)
        right_factors = torch.cat(right_factors_list, -1)
        matrix_arg_grads = lazy_tsr._quad_form_derivative(left_factors, right_factors)

        # precond gradient
        precond_arg_grads = precond_lt._quad_form_derivative(
            -precond_probe_vectors * coef, precond_probe_vectors * logdet_grad_output
        )

        # inv_quad_rhs gradients
        if ctx.inv_quad:
            inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul_(-2)
            if ctx.is_vector:
                inv_quad_rhs_grad.squeeze_(-1)
            res = [inv_quad_rhs_grad] + list(matrix_arg_grads) + list(precond_arg_grads)
        else:
            res = list(matrix_arg_grads) + list(precond_arg_grads)

        return tuple([None] * 7 + res)
