import torch
import gpytorch
from gpytorch.lazy import LazyVariable
from gpytorch.utils import function_factory


class NonLazyVariable(LazyVariable):
    def __init__(self, var):
        """
        Not a lazy variable

        Args:
        - var (Variable: matrix) a variable
        """
        super(NonLazyVariable, self).__init__(var)
        self.var = var

    def _matmul_closure_factory(self, tensor):
        def closure(rhs_tensor):
            return torch.matmul(tensor, rhs_tensor)
        return closure

    def _t_matmul_closure_factory(self, tensor):
        def closure(rhs_tensor):
            return torch.matmul(tensor.transpose(-1, -2), rhs_tensor)
        return closure

    def _derivative_quadratic_form_factory(self, mat):
        orig_closure = function_factory._default_derivative_quadratic_form_factory(mat)

        def closure(*stuff):
            res = orig_closure(*stuff)[0]
            # Batch mode?
            if res.ndimension() > self.var.ndimension():
                res = res.sum(0)
            return res,

        return closure

    def _size(self):
        return self.var.size()

    def _transpose_nonbatch(self):
        return NonLazyVariable(self.var.transpose(-1, -2))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        return self.var[batch_indices.data, left_indices.data, right_indices.data]

    def _get_indices(self, left_indices, right_indices):
        return self.var[left_indices.data, right_indices.data]

    def add_diag(self, diag):
        return NonLazyVariable(gpytorch.add_diag(self.var, diag))

    def diag(self):
        return self.var.diag()

    def evaluate(self):
        return self.var

    def repeat(self, *sizes):
        return NonLazyVariable(self.var.repeat(*sizes))

    def __getitem__(self, index):
        return NonLazyVariable(self.var[index])
