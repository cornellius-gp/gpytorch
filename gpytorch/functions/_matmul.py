#!/usr/bin/env python3

from torch.autograd import Function
from .. import settings


class Matmul(Function):
    def __init__(self, representation_tree):
        self.representation_tree = representation_tree

    def forward(self, rhs, *matrix_args):
        orig_rhs = rhs

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        lazy_tsr = self.representation_tree(*matrix_args)
        res = lazy_tsr._matmul(rhs)

        to_save = [orig_rhs] + list(matrix_args)
        self.save_for_backward(*to_save)
        if not settings.memory_efficient.on():
            self._lazy_tsr = lazy_tsr

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def backward(self, grad_output):
        rhs = self.saved_tensors[0]
        matrix_args = self.saved_tensors[1:]
        rhs_shape = rhs.shape

        rhs_grad = None
        arg_grads = [None] * len(matrix_args)

        # input_1 gradient
        if any(self.needs_input_grad[1:]):
            rhs = rhs.unsqueeze(-1) if (rhs.ndimension() == 1) else rhs
            grad_output_matrix = grad_output.unsqueeze(-1) if grad_output.ndimension() == 1 else grad_output
            arg_grads = self.representation_tree(*matrix_args)._quad_form_derivative(grad_output_matrix, rhs)

        # input_2 gradient
        if self.needs_input_grad[0]:
            if hasattr(self, "_lazy_tsr"):
                lazy_tsr = self._lazy_tsr
            else:
                lazy_tsr = self.representation_tree(*matrix_args)
            rhs_grad = lazy_tsr._t_matmul(grad_output)
            rhs_grad = rhs_grad.view(rhs_shape)

        return tuple([rhs_grad] + list(arg_grads))
