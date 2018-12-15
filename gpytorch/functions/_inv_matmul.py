#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils import linear_cg
from .. import settings


class InvMatmul(Function):
    def __init__(self, representation_tree, preconditioner=None, has_left=False):
        self.representation_tree = representation_tree
        self.preconditioner = preconditioner
        self.has_left = has_left

    def forward(self, *args):
        left_tensor = None
        right_tensor = None
        matrix_args = None
        if self.has_left:
            left_tensor, right_tensor, *matrix_args = args
        else:
            right_tensor, *matrix_args = args
        lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul

        self.is_vector = False
        if right_tensor.ndimension() == 1:
            right_tensor.unsqueeze_(-1)
            self.is_vector = True

        # Perform solves (for inv_quad) and tridiagonalization (for estimating log_det)
        if self.has_left:
            rhs = torch.cat([left_tensor.transpose(-1, -2), right_tensor], -1)
            solves = linear_cg(
                matmul_closure, rhs, max_iter=settings.max_cg_iterations.value(), preconditioner=self.preconditioner
            )
            res = solves[..., left_tensor.size(-2):]
            res = left_tensor @ res
        else:
            solves = linear_cg(
                matmul_closure, right_tensor, max_iter=settings.max_cg_iterations.value(),
                preconditioner=self.preconditioner
            )
            res = solves

        if self.is_vector:
            res.squeeze_(-1)
            right_tensor.squeeze_(-1)

        if self.has_left:
            args = [solves, left_tensor, right_tensor] + list(matrix_args)
        else:
            args = [solves, right_tensor] + list(matrix_args)
        self.save_for_backward(*args)
        if settings.memory_efficient.off():
            self._lazy_tsr = lazy_tsr

        return res

    def backward(self, grad_output):
        # Extract items that were saved
        if self.has_left:
            solves, left_tensor, right_tensor, *matrix_args = self.saved_tensors
            left_solves = solves[..., :left_tensor.size(-2)]
            right_solves = solves[..., left_tensor.size(-2):]
        else:
            right_solves, right_tensor, *matrix_args = self.saved_tensors

        # Get matrix functions
        if hasattr(self, "_lazy_tsr"):
            lazy_tsr = self._lazy_tsr
        else:
            lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul

        # Define gradient placeholders
        arg_grads = [None] * len(matrix_args)
        left_grad = None
        right_grad = None
        if any(self.needs_input_grad):
            # De-vectorize objects
            if self.is_vector:
                right_tensor = right_tensor.unsqueeze(-1)

            if not self.has_left:
                if self.is_vector:
                    grad_output = grad_output.unsqueeze(-1)
                    right_solves.unsqueeze_(-1)

                # Compute self^{-1} grad_output
                left_solves = linear_cg(
                    matmul_closure,
                    grad_output,
                    max_iter=settings.max_cg_iterations.value(),
                    preconditioner=self.preconditioner,
                )

                if any(self.needs_input_grad[1:]):
                    arg_grads = lazy_tsr._quad_form_derivative(left_solves, right_solves.mul(-1))
                if self.needs_input_grad[0]:
                    right_grad = left_solves
                    if self.is_vector:
                        right_grad.squeeze_(-1)

                return tuple([right_grad] + list(arg_grads))

            else:
                if self.is_vector:
                    grad_output = grad_output.unsqueeze(-1)

                left_solves = left_solves @ grad_output

                if self.needs_input_grad[1]:
                    left_grad = grad_output @ right_solves.transpose(-1, -2)
                if any(self.needs_input_grad[2:]):
                    arg_grads = lazy_tsr._quad_form_derivative(left_solves, right_solves.mul(-1))
                if self.needs_input_grad[0]:
                    right_grad = left_solves
                    if self.is_vector:
                        right_grad.squeeze_(-1)

                return tuple([left_grad, right_grad] + list(arg_grads))
