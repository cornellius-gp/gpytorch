import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from ..utils.toeplitz import sym_toeplitz_matmul, sym_toeplitz_derivative_quadratic_form


class ToeplitzLazyVariable(LazyVariable):
    def __init__(self, column):
        super(ToeplitzLazyVariable, self).__init__(column)
        self.column = column

    def _matmul_closure_factory(self, column):
        def closure(tensor):
            return sym_toeplitz_matmul(column, tensor)
        return closure

    def _derivative_quadratic_form_factory(self, column):
        def closure(left_vectors, right_vectors):
            if left_vectors.ndimension() == 1:
                left_factor = left_vectors.unsqueeze(0)
                right_factor = right_vectors.unsqueeze(0)
            else:
                left_factor = left_vectors
                right_factor = right_vectors

            return sym_toeplitz_derivative_quadratic_form(left_factor, right_factor),
        return closure

    def add_jitter(self):
        jitter = self.column.data.new(self.column.size(-1)).zero_()
        jitter.narrow(-1, 0, 1).fill_(1e-4)
        return ToeplitzLazyVariable(self.column.add(Variable(jitter)))

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.
        """
        diag_term = self.column.select(-1, 0)
        if self.column.ndimension() > 1:
            diag_term = diag_term.unsqueeze(-1)
        return diag_term.expand(*self.column.size())

    def repeat(self, *sizes):
        """
        Repeat elements of the Variable.
        Right now it only works to create a batched version of a ToeplitzLazyVariable.

        e.g. `var.repeat(3, 1, 1)` creates a batched version of length 3
        """

        return ToeplitzLazyVariable(self.column.repeat(sizes[0], 1))

    def size(self):
        if self.column.ndimension() == 2:
            return torch.Size((self.column.size(0), self.column.size(-1), self.column.size(-1)))
        else:
            return torch.Size((self.column.size(-1), self.column.size(-1)))

    def _transpose_nonbatch(self):
        return ToeplitzLazyVariable(self.column)

    def _get_indices(self, left_indices, right_indices):
        n_grid = self.column.size(-1)
        toeplitz_indices = (left_indices - right_indices).fmod(n_grid).abs().long()
        return self.column.index_select(0, toeplitz_indices)
