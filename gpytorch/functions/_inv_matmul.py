#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils import linear_cg
from .. import settings


class InvMatmul(Function):
    def __init__(self, representation_tree, preconditioner=None):
        self.representation_tree = representation_tree
        self.preconditioner = preconditioner

    def forward(self, rhs, *matrix_args):
        lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul

        self.is_vector = False
        if rhs.ndimension() == 1:
            rhs.unsqueeze_(-1)
            self.is_vector = True

        # Perform solves (for inv_quad) and tridiagonalization (for estimating log_det)
        res = linear_cg(
            matmul_closure, rhs, max_iter=settings.max_cg_iterations.value(), preconditioner=self.preconditioner
        )

        if self.is_vector:
            res.squeeze_(-1)
            rhs.squeeze_(-1)

        args = [res, rhs] + list(matrix_args)
        self.save_for_backward(*args)
        if not settings.memory_efficient.on():
            self._lazy_tsr = lazy_tsr

        return res

    def backward(self, grad_output):
        # Extract items that were saved
        rhs_solves = self.saved_tensors[0]
        rhs = self.saved_tensors[1]
        matrix_args = self.saved_tensors[2:]

        # Get matrix functions
        if hasattr(self, "_lazy_tsr"):
            lazy_tsr = self._lazy_tsr
        else:
            lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul

        # Define gradient placeholders
        arg_grads = [None] * len(matrix_args)
        rhs_grad = None
        if any(self.needs_input_grad):
            # De-vectorize objects
            if self.is_vector:
                rhs = rhs.unsqueeze(-1)
                rhs_solves = rhs_solves.unsqueeze(-1)
                grad_output = grad_output.unsqueeze(-1)

            # Compute self^{-1} grad_output
            grad_output_solves = linear_cg(
                matmul_closure,
                grad_output,
                max_iter=settings.max_cg_iterations.value(),
                preconditioner=self.preconditioner,
            )

            # input_1 gradient
            if any(self.needs_input_grad[1:]):
                if lazy_tsr is not None:
                    arg_grads = lazy_tsr._quad_form_derivative(grad_output_solves, rhs_solves.mul(-1))
                else:
                    arg_grads = (torch.matmul(grad_output_solves, rhs_solves.mul(-1).transpose(-1, -2)),)

            # input_2 gradient
            if self.needs_input_grad[0]:
                rhs_grad = grad_output_solves
                if self.is_vector:
                    rhs_grad.squeeze_(-1)

        return tuple([rhs_grad] + list(arg_grads))
