import torch
import gpytorch
from gpytorch.lazy import LazyVariable
from gpytorch.utils import function_factory
from ..posterior import DefaultPosteriorStrategy


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

    def _derivative_quadratic_form_factory(self, mat):
        return function_factory._default_derivative_quadratic_form_factory(mat)

    def add_diag(self, diag):
        return NonLazyVariable(gpytorch.add_diag(self.var, diag))

    def diag(self):
        return self.var.diag()

    def evaluate(self):
        return self.var

    def mul(self, constant):
        return NonLazyVariable(self.var.mul(constant))

    def posterior_strategy(self):
        return DefaultPosteriorStrategy(self)

    def representation(self):
        return self.var,

    def size(self):
        return self.var.size()

    def t(self):
        return NonLazyVariable(self.var.t())

    def __getitem__(self, index):
        return NonLazyVariable(self.var[index])
