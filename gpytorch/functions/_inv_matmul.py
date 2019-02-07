#!/usr/bin/env python3

import torch
from torch.autograd import Function
from .. import settings


class InvMatmul(Function):
    def __init__(self, representation_tree, has_left=False):
        self.representation_tree = representation_tree
        self.has_left = has_left

    def forward(self, *args):
        left_tensor = None
        right_tensor = None
        matrix_args = None
        if self.has_left:
            left_tensor, right_tensor, *matrix_args = args
        else:
            right_tensor, *matrix_args = args
        orig_right_tensor = right_tensor
        lazy_tsr = self.representation_tree(*matrix_args)

        with torch.no_grad():
            self.preconditioner = lazy_tsr.detach()._inv_matmul_preconditioner()

        self.is_vector = False
        if right_tensor.ndimension() == 1:
            right_tensor = right_tensor.unsqueeze(-1)
            self.is_vector = True

        # Perform solves (for inv_quad) and tridiagonalization (for estimating logdet)
        if self.has_left:
            rhs = torch.cat([left_tensor.transpose(-1, -2), right_tensor], -1)
            solves = lazy_tsr._solve(rhs, self.preconditioner)
            res = solves[..., left_tensor.size(-2):]
            res = left_tensor @ res
        else:
            solves = lazy_tsr._solve(right_tensor, self.preconditioner)
            res = solves

        if self.is_vector:
            res = res.squeeze(-1)

        if self.has_left:
            args = [solves, left_tensor, orig_right_tensor] + list(matrix_args)
        else:
            args = [solves, orig_right_tensor] + list(matrix_args)
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

        # Define gradient placeholders
        arg_grads = [None] * len(matrix_args)
        left_grad = None
        right_grad = None
        if any(self.needs_input_grad):
            # De-vectorize objects
            if self.is_vector:
                right_tensor = right_tensor.unsqueeze(-1)
                grad_output = grad_output.unsqueeze(-1)

            if not self.has_left:
                # Compute self^{-1} grad_output
                left_solves = lazy_tsr._solve(grad_output, self.preconditioner)

                if any(self.needs_input_grad[1:]):
                    arg_grads = lazy_tsr._quad_form_derivative(left_solves, right_solves.mul(-1))
                if self.needs_input_grad[0]:
                    right_grad = left_solves
                    if self.is_vector:
                        right_grad.squeeze_(-1)

                return tuple([right_grad] + list(arg_grads))

            else:
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
