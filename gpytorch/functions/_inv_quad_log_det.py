#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils.lanczos import lanczos_tridiag_to_diag
from ..utils.stochastic_lq import StochasticLQ
from ..utils import linear_cg
from .. import settings


class InvQuadLogDet(Function):
    """
    Given a PSD matrix A (or a batch of PSD matrices A), this function computes one or both
    of the following
    - The matrix solves A^{-1} b
    - logdet(A)
    """

    def __init__(
        self,
        representation_tree,
        dtype,
        device,
        matrix_shape,
        batch_shape=torch.Size(),
        inv_quad=False,
        log_det=False,
        preconditioner=None,
        log_det_correction=None,
    ):
        if not (inv_quad or log_det):
            raise RuntimeError("Either inv_quad or log_det must be true (or both)")

        self.representation_tree = representation_tree
        self.dtype = dtype
        self.device = device
        self.matrix_shape = matrix_shape
        self.batch_shape = batch_shape
        self.inv_quad = inv_quad
        self.log_det = log_det
        self.preconditioner = preconditioner
        self.log_det_correction = log_det_correction

    def forward(self, *args):
        """
        *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
        If self.inv_quad is true, the first entry in *args is inv_quad_rhs (Tensor)
        - the RHS of the matrix solves.

        Returns:
        - (Scalar) The inverse quadratic form (or None, if self.inv_quad is False)
        - (Scalar) The log determinant (or None, self.if log_det is False)
        """
        matrix_args = None
        inv_quad_rhs = None
        if self.inv_quad:
            matrix_args = args[1:]
            inv_quad_rhs = args[0]
        else:
            matrix_args = args

        # Get closure for matmul
        lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul

        # Collect terms for LinearCG
        # We use LinearCG for both matrix solves and for stochastically estimating the log det
        rhs_list = []
        num_random_probes = 0
        num_inv_quad_solves = 0

        # Probe vector for lanczos quadrature (log_det estimation)
        probe_vectors = None
        probe_vector_norms = None
        if self.log_det:
            num_random_probes = settings.num_trace_samples.value()
            probe_vectors = torch.empty(self.matrix_shape[-1], num_random_probes, dtype=self.dtype, device=self.device)
            probe_vectors.bernoulli_().mul_(2).add_(-1)
            probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
            if self.batch_shape is not None:
                probe_vectors = probe_vectors.expand(*self.batch_shape, self.matrix_shape[-1], num_random_probes)
                probe_vector_norms = probe_vector_norms.expand(*self.batch_shape, 1, num_random_probes)
            probe_vectors = probe_vectors.div(probe_vector_norms)
            rhs_list.append(probe_vectors)

        # RHS for inv_quad
        self.is_vector = False
        if self.inv_quad:
            if inv_quad_rhs.ndimension() == 1:
                inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
                self.is_vector = True
            rhs_list.append(inv_quad_rhs)
            num_inv_quad_solves = inv_quad_rhs.size(-1)

        # Perform solves (for inv_quad) and tridiagonalization (for estimating log_det)
        rhs = torch.cat(rhs_list, -1)
        t_mat = None
        if self.log_det and not settings.skip_logdet_forward.on():
            solves, t_mat = linear_cg(
                matmul_closure,
                rhs,
                n_tridiag=num_random_probes,
                max_iter=settings.max_cg_iterations.value(),
                max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
                preconditioner=self.preconditioner,
            )

        else:
            solves = linear_cg(
                matmul_closure,
                rhs,
                n_tridiag=0,
                max_iter=settings.max_cg_iterations.value(),
                preconditioner=self.preconditioner,
            )

        # Final values to return
        log_det_term = torch.zeros(lazy_tsr.batch_shape, dtype=self.dtype, device=self.device)
        inv_quad_term = torch.zeros(lazy_tsr.batch_shape, dtype=self.dtype, device=self.device)

        # Compute log_det from tridiagonalization
        if self.log_det and not settings.skip_logdet_forward.on():
            if torch.any(torch.isnan(t_mat)).item():
                log_det_term = float("nan")
            else:
                if self.batch_shape is None:
                    t_mat = t_mat.unsqueeze(1)
                eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)
                slq = StochasticLQ()
                log_det_term, = slq.evaluate(self.matrix_shape, eigenvalues, eigenvectors, [lambda x: x.log()])

                # Add correction
                if self.log_det_correction is not None:
                    log_det_term = log_det_term + self.log_det_correction

        # Extract inv_quad solves from all the solves
        if self.inv_quad:
            inv_quad_solves = solves.narrow(-1, num_random_probes, num_inv_quad_solves)
            inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        self.num_random_probes = num_random_probes
        self.num_inv_quad_solves = num_inv_quad_solves

        to_save = list(matrix_args) + [solves, probe_vectors, probe_vector_norms]
        self.save_for_backward(*to_save)

        if not settings.memory_efficient.on():
            self._lazy_tsr = lazy_tsr

        return inv_quad_term, log_det_term

    def backward(self, inv_quad_grad_output, log_det_grad_output):
        matrix_arg_grads = None
        inv_quad_rhs_grad = None

        # Which backward passes should we compute?
        compute_inv_quad_grad = inv_quad_grad_output.abs().sum() and self.inv_quad
        compute_log_det_grad = log_det_grad_output.abs().sum() and self.log_det

        # Get input arguments, and get gradients in the proper form
        matrix_args = self.saved_tensors[:-3]
        solves = self.saved_tensors[-3]
        probe_vectors = self.saved_tensors[-2]
        probe_vector_norms = self.saved_tensors[-1]

        if hasattr(self, "_lazy_tsr"):
            lazy_tsr = self._lazy_tsr
        else:
            lazy_tsr = self.representation_tree(*matrix_args)

        # Fix grad_output sizes
        if self.inv_quad:
            inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-2)
        if compute_log_det_grad:
            log_det_grad_output = log_det_grad_output.unsqueeze(-1)
            log_det_grad_output.unsqueeze_(-1)

        # Divide up the solves
        probe_vector_solves = None
        inv_quad_solves = None
        neg_inv_quad_solves_times_grad_out = None
        if compute_log_det_grad:
            coef = 1.0 / probe_vectors.size(-1)
            probe_vector_solves = solves.narrow(-1, 0, self.num_random_probes).mul(coef)
            probe_vector_solves.mul_(probe_vector_norms).mul_(log_det_grad_output)
            probe_vectors = probe_vectors.mul(probe_vector_norms)
        if self.inv_quad:
            inv_quad_solves = solves.narrow(-1, self.num_random_probes, self.num_inv_quad_solves)
            neg_inv_quad_solves_times_grad_out = inv_quad_solves.mul(inv_quad_grad_output).mul_(-1)

        # input_1 gradient
        if any(self.needs_input_grad):
            # Collect terms for arg grads
            left_factors_list = []
            right_factors_list = []

            if compute_log_det_grad:
                left_factors_list.append(probe_vector_solves)
                right_factors_list.append(probe_vectors)

            if compute_inv_quad_grad:
                left_factors_list.append(neg_inv_quad_solves_times_grad_out)
                right_factors_list.append(inv_quad_solves)

            left_factors = torch.cat(left_factors_list, -1)
            right_factors = torch.cat(right_factors_list, -1)
            matrix_arg_grads = lazy_tsr._quad_form_derivative(left_factors, right_factors)

        # input_2 gradients
        if compute_inv_quad_grad and self.needs_input_grad[0]:
            inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul_(-2)
        elif self.inv_quad:
            inv_quad_rhs_grad = torch.zeros_like(inv_quad_solves)
        if self.is_vector:
            inv_quad_rhs_grad.squeeze_(-1)

        if self.inv_quad:
            res = tuple([inv_quad_rhs_grad] + list(matrix_arg_grads))
        else:
            res = matrix_arg_grads

        return tuple(res)
