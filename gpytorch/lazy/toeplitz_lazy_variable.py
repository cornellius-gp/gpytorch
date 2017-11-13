import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .interpolated_lazy_variable import InterpolatedLazyVariable
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

    def representation(self):
        return self.column,

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

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        column = self.column

        squeeze_left = False
        squeeze_right = False
        if isinstance(index[-2], int):
            index[-2] = slice(index[-2], index[-2] + 1, None)
            squeeze_left = True
        if isinstance(index[-1], int):
            index[-1] = slice(index[-1], index[-1] + 1, None)
            squeeze_right = True

        # Handle batch dimensions
        isbatch = ndimension >= 3
        if isbatch:
            batch_index = tuple(index[:-2])
            column = self.column[batch_index]

        ndimension = column.ndimension() + 1

        # Handle index
        left_index = index[-2]
        right_index = index[-1]

        batch_sizes = list(column.size()[:-1])
        row_iter = column.data.new(column.size(-1)).long()
        torch.arange(0, self.column.size(-1), out=row_iter)

        left_interp_indices = row_iter[left_index].unsqueeze(-1)
        right_interp_indices = row_iter[right_index].unsqueeze(-1)

        left_interp_len = len(left_interp_indices)
        right_interp_len = len(right_interp_indices)
        for i in range(ndimension - 2):
            left_interp_indices.unsqueeze_(0)
            right_interp_indices.unsqueeze_(0)

        left_interp_indices = left_interp_indices.expand(*(batch_sizes + [left_interp_len, 1]))
        left_interp_values = left_interp_indices.new(left_interp_indices.size()).fill_(1).float()
        right_interp_indices = right_interp_indices.expand(*(batch_sizes + [right_interp_len, 1]))
        right_interp_values = right_interp_indices.new(right_interp_indices.size()).fill_(1).float()

        res = InterpolatedLazyVariable(ToeplitzLazyVariable(column), Variable(left_interp_indices),
                                       Variable(left_interp_values),
                                       Variable(right_interp_indices), Variable(right_interp_values))

        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res

    def _get_indices(self, left_indices, right_indices):
        n_grid = self.column.size(-1)
        toeplitz_indices = (left_indices - right_indices).fmod(n_grid).abs().long()
        return self.column.index_select(0, toeplitz_indices)
