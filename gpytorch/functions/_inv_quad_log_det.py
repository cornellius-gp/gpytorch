#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils.lanczos import lanczos_tridiag_to_diag
from ..utils.stochastic_lq import StochasticLQ
from .. import settings


class InvQuadLogDet(Function):
    """
    Given a PSD matrix A (or a batch of PSD matrices A), this function computes one or both
    of the following
    - The matrix solves A^{-1} b
    - logdet(A)
    """
    @staticmethod
    def forward(
        ctx,
        representation_tree,
        dtype,
        device,
        matrix_shape,
        batch_shape=torch.Size(),
        inv_quad=False,
        logdet=False,
        probe_vectors=None,
        probe_vector_norms=None,
        *args
    ):
        """
        *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
        If self.inv_quad is true, the first entry in *args is inv_quad_rhs (Tensor)
        - the RHS of the matrix solves.

        Returns:
        - (Scalar) The inverse quadratic form (or None, if self.inv_quad is False)
        - (Scalar) The log determinant (or None, self.if logdet is False)
        """

        if not (inv_quad or logdet):
            raise RuntimeError("Either inv_quad or logdet must be true (or both)")

        ctx.representation_tree = representation_tree
        ctx.dtype = dtype
        ctx.device = device
        ctx.matrix_shape = matrix_shape
        ctx.batch_shape = batch_shape
        ctx.inv_quad = inv_quad
        ctx.logdet = logdet

        if (probe_vectors is None or probe_vector_norms is None) and logdet:
            num_random_probes = settings.num_trace_samples.value()
            probe_vectors = torch.empty(matrix_shape[-1], num_random_probes, dtype=dtype, device=device)
            probe_vectors.bernoulli_().mul_(2).add_(-1)
            probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
            if batch_shape is not None:
                probe_vectors = probe_vectors.expand(*batch_shape, matrix_shape[-1], num_random_probes)
                probe_vector_norms = probe_vector_norms.expand(*batch_shape, 1, num_random_probes)
            probe_vectors = probe_vectors.div(probe_vector_norms)

        ctx.probe_vectors = probe_vectors
        ctx.probe_vector_norms = probe_vector_norms

        if ctx.logdet and not ctx.probe_vectors.numel():
            raise RuntimeError("Probe vectors were not supplied for logdet computation")

        matrix_args = None
        inv_quad_rhs = None
        if ctx.inv_quad:
            matrix_args = args[1:]
            inv_quad_rhs = args[0]
        else:
            matrix_args = args

        # Get closure for matmul
        lazy_tsr = ctx.representation_tree(*matrix_args)
        with torch.no_grad():
            preconditioner, logdet_correction = lazy_tsr.detach()._preconditioner()

        # Collect terms for LinearCG
        # We use LinearCG for both matrix solves and for stochastically estimating the log det
        rhs_list = []
        num_random_probes = 0
        num_inv_quad_solves = 0

        # RHS for logdet
        if ctx.logdet:
            rhs_list.append(ctx.probe_vectors)
            num_random_probes = ctx.probe_vectors.size(-1)

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
        t_mat = None
        if ctx.logdet and settings.skip_logdet_forward.off():
            solves, t_mat = lazy_tsr._solve(rhs, preconditioner, num_tridiag=num_random_probes)

        else:
            solves = lazy_tsr._solve(rhs, preconditioner, num_tridiag=0)

        # Final values to return
        logdet_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)
        inv_quad_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)

        # Compute logdet from tridiagonalization
        if ctx.logdet and settings.skip_logdet_forward.off():
            if torch.any(torch.isnan(t_mat)).item():
                logdet_term = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
            else:
                if ctx.batch_shape is None:
                    t_mat = t_mat.unsqueeze(1)
                eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)
                slq = StochasticLQ()
                logdet_term, = slq.evaluate(ctx.matrix_shape, eigenvalues, eigenvectors, [lambda x: x.log()])

                # Add correction
                if logdet_correction is not None:
                    logdet_term = logdet_term + logdet_correction

        # Extract inv_quad solves from all the solves
        if ctx.inv_quad:
            inv_quad_solves = solves.narrow(-1, num_random_probes, num_inv_quad_solves)
            inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        ctx.num_random_probes = num_random_probes
        ctx.num_inv_quad_solves = num_inv_quad_solves

        to_save = list(matrix_args) + [solves, ]
        ctx.save_for_backward(*to_save)

        if settings.memory_efficient.off():
            ctx._lazy_tsr = lazy_tsr

        return inv_quad_term, logdet_term

    @staticmethod
    def backward(ctx, inv_quad_grad_output, logdet_grad_output):
        matrix_arg_grads = None
        inv_quad_rhs_grad = None

        # Which backward passes should we compute?
        compute_inv_quad_grad = inv_quad_grad_output.abs().sum() and ctx.inv_quad
        compute_logdet_grad = logdet_grad_output.abs().sum() and ctx.logdet

        # Get input arguments, and get gradients in the proper form
        matrix_args = ctx.saved_tensors[:-1]
        solves = ctx.saved_tensors[-1]

        if hasattr(ctx, "_lazy_tsr"):
            lazy_tsr = ctx._lazy_tsr
        else:
            lazy_tsr = ctx.representation_tree(*matrix_args)

        # Fix grad_output sizes
        if ctx.inv_quad:
            inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-2)
        if compute_logdet_grad:
            logdet_grad_output = logdet_grad_output.unsqueeze(-1)
            logdet_grad_output.unsqueeze_(-1)

        # Divide up the solves
        probe_vector_solves = None
        inv_quad_solves = None
        neg_inv_quad_solves_times_grad_out = None
        if compute_logdet_grad:
            coef = 1.0 / ctx.probe_vectors.size(-1)
            probe_vector_solves = solves.narrow(-1, 0, ctx.num_random_probes).mul(coef)
            probe_vector_solves.mul_(ctx.probe_vector_norms).mul_(logdet_grad_output)
            probe_vectors = ctx.probe_vectors.mul(ctx.probe_vector_norms)
        if ctx.inv_quad:
            inv_quad_solves = solves.narrow(-1, ctx.num_random_probes, ctx.num_inv_quad_solves)
            neg_inv_quad_solves_times_grad_out = inv_quad_solves.mul(inv_quad_grad_output).mul_(-1)

        # input_1 gradient
        if any(ctx.needs_input_grad):
            # Collect terms for arg grads
            left_factors_list = []
            right_factors_list = []

            if compute_logdet_grad:
                left_factors_list.append(probe_vector_solves)
                right_factors_list.append(probe_vectors)

            if compute_inv_quad_grad:
                left_factors_list.append(neg_inv_quad_solves_times_grad_out)
                right_factors_list.append(inv_quad_solves)

            left_factors = torch.cat(left_factors_list, -1)
            right_factors = torch.cat(right_factors_list, -1)
            matrix_arg_grads = lazy_tsr._quad_form_derivative(left_factors, right_factors)

        # input_2 gradients
        if compute_inv_quad_grad and ctx.needs_input_grad[9]:
            inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul_(-2)
        elif ctx.inv_quad:
            inv_quad_rhs_grad = torch.zeros_like(inv_quad_solves)
        if ctx.is_vector:
            inv_quad_rhs_grad.squeeze_(-1)

        if ctx.inv_quad:
            res = [inv_quad_rhs_grad] + list(matrix_arg_grads)
        else:
            res = matrix_arg_grads

        return tuple([None] * 9 + res)
