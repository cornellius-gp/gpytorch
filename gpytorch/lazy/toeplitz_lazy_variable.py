import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.utils import toeplitz
from .lazy_variable import LazyVariable
from .mul_lazy_variable import MulLazyVariable
from ..posterior import InterpolatedPosteriorStrategy
from ..utils import sparse_eye, bdsmm
from ..utils.toeplitz import interpolated_sym_toeplitz_matmul, index_coef_to_sparse, sym_toeplitz_matmul, \
    sym_toeplitz_derivative_quadratic_form


class ToeplitzLazyVariable(LazyVariable):
    def __init__(self, c, J_left=None, C_left=None, J_right=None, C_right=None, added_diag=None):
        if not isinstance(c, Variable):
            raise RuntimeError('ToeplitzLazyVariable is intended to wrap Variable versions of \
                                the first column and row.')

        self.c = c
        self.J_left = J_left
        self.C_left = C_left
        self.J_right = J_right
        self.C_right = C_right
        self.added_diag = added_diag

    def _matmul_closure_factory(self, *args):
        if len(args) == 1:
            c, = args

            def closure(tensor):
                return sym_toeplitz_matmul(c, tensor)

        elif len(args) == 3:
            c, W_left, W_right = args

            def closure(tensor):
                return interpolated_sym_toeplitz_matmul(c, tensor, W_left, W_right)

        elif len(args) == 4:
            c, W_left, W_right, added_diag = args

            def closure(tensor):
                return interpolated_sym_toeplitz_matmul(c, tensor, W_left, W_right, added_diag)

        else:
            raise AttributeError('Invalid number of arguments')

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        def closure(left_vectors, right_vectors):
            if left_vectors.ndimension() == 1:
                left_factor = left_vectors.unsqueeze(0)
                right_factor = right_vectors.unsqueeze(0)
            else:
                left_factor = left_vectors
                right_factor = right_vectors
            if len(args) == 1:
                toeplitz_column, = args
                return sym_toeplitz_derivative_quadratic_form(left_factor, right_factor),
            elif len(args) == 3:
                toeplitz_column, W_left, W_right = args
                if W_left.ndimension() == 3:
                    left_factor = bdsmm(W_left.transpose(1, 2), left_factor.transpose(1, 2)).transpose(1, 2)
                    right_factor = bdsmm(W_right.transpose(1, 2), right_factor.transpose(1, 2)).transpose(1, 2)
                else:
                    left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                    right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()
                return sym_toeplitz_derivative_quadratic_form(left_factor, right_factor), None, None
            elif len(args) == 4:
                toeplitz_column, W_left, W_right, added_diag, = args

                diag_grad = left_vectors.new(len(added_diag)).zero_()
                diag_grad[0] = (left_vectors * right_vectors).sum()
                if W_left.ndimension() == 3:
                    left_factor = bdsmm(W_left.transpose(1, 2), left_factor.t(1, 2)).t(1, 2)
                    right_factor = bdsmm(W_right.t(1, 2), right_factor.t(1, 2)).t(1, 2)
                else:
                    left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                    right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()
                return sym_toeplitz_derivative_quadratic_form(left_factor, right_factor), None, None, diag_grad
        return closure

    def is_batch(self):
        if self.J_left is None:
            return self.c.ndimension() == 2
        return self.J_left.ndimension() == 3

    def add_diag(self, diag):
        if self.J_left is not None:
            toeplitz_diag = diag.expand(len(self.J_left))
        else:
            toeplitz_diag = diag.expand_as(self.c)

        return ToeplitzLazyVariable(self.c, self.J_left, self.C_left,
                                    self.J_right, self.C_right, toeplitz_diag)

    def add_jitter(self):
        jitter = self.c.data.new(self.c.size(-1)).zero_()
        jitter[0] = 1e-4
        return ToeplitzLazyVariable(self.c.add(Variable(jitter)), self.J_left, self.C_left,
                                    self.J_right, self.C_right, self.added_diag)

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.
        """
        if len(self.J_left) != len(self.J_right):
            raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
        n_data, n_interp = self.J_left.size()
        n_grid = self.c.size(-1)

        # For row k, we will calculate the diagonal element as sum_{i,j} w_left^k_i w_right^k_j T_{i,j}
        # Batch compute the non-zero values of the outer products w_left^k w_left^k^T
        left_interp_values = self.C_left.unsqueeze(2)
        right_interp_values = self.C_right.unsqueeze(1)
        interp_values = torch.matmul(left_interp_values, right_interp_values)

        # Batch compute Toeplitz values that will be non-zero for row k
        left_interp_indices = self.J_left.unsqueeze(2).expand(n_data, n_interp, n_interp)
        right_interp_indices = self.J_right.unsqueeze(1).expand(n_data, n_interp, n_interp)
        toeplitz_indices = (left_interp_indices - right_interp_indices).fmod(n_grid).abs().long()
        toeplitz_vals = self.c.index_select(0, Variable(toeplitz_indices.view(-1))).view(toeplitz_indices.size())

        # Compute batch sums to get diagonal elements
        diag = (Variable(interp_values) * toeplitz_vals).sum(2).sum(1)

        # Add added diag term
        if self.added_diag is not None:
            diag += self.added_diag
        return diag

    def evaluate(self):
        """
        Explicitly evaluate and return the Toeplitz matrix this object wraps as a float Tensor.
        To do this, we explicitly compute W_{left}TW_{right}^{T} and return it.

        Warning: as implicitly stored by this LazyVariable, W is very sparse and T requires O(n)
        storage, where as the full matrix requires O(n^2) storage. Calling evaluate can very easily
        lead to memory issues. As a result, using it should be a last resort.
        """

        if self.J_right is None:
            eye = Variable(self.c.data.new(self.c.size(-1)).fill_(1).diag())
            if self.c.ndimension() == 2:
                eye = eye.unsqueeze(0).expand(len(self.c), self.c.size(1), self.c.size(1))
        else:
            if self.J_right.ndimension() == 3:
                eye = Variable(self.c.data.new(self.J_right.size(1)).fill_(1).diag())
                eye = eye.unsqueeze(0).expand(len(self.c), self.J_right.size(1), self.J_right.size(1))
            else:
                eye = Variable(self.c.data.new(self.J_right.size(0)).fill_(1).diag())
        res = self.matmul(eye)
        return res

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        epsilon = Variable(torch.randn(self.c.size(-1), gpytorch.functions.num_trace_samples))
        samples = chol_var_covar.mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1).expand_as(samples)
        W_left = Variable(toeplitz.index_coef_to_sparse(self.J_left, self.C_left, self.c.size(-1)))
        samples = gpytorch.dsmm(W_left, samples)
        log_likelihood = log_probability_func(samples, train_y)

        return log_likelihood

    def mul(self, other):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant or another ToeplitzLazyVariable.
        To accomplish this, we multiply the Toeplitz component by the constant or the other ToeplitzLazyVariable's
        Toeplitz component. This way, the interpolation acts on the multiplied values.

        Args:
            - other (broadcastable with self.c) - Constant or ToeplitzLazyVariable to multiply by.
        Returns:
            - ToeplitzLazyVariable with c = c*(constant)
        """
        if isinstance(other, LazyVariable):
            return MulLazyVariable(self, other)
        else:
            return ToeplitzLazyVariable(self.c.mul(other), self.J_left, self.C_left,
                                        self.J_right, self.C_right, self.added_diag)

    def mul_(self, other):
        """
        In-place version of mul.
        """
        self.c.mul_(other)
        return self

    def posterior_strategy(self):
        if not hasattr(self, '_posterior_strategy'):
            toeplitz_column, interp_left, interp_right = self.representation()[:3]
            grid = ToeplitzLazyVariable(toeplitz_column)
            self._posterior_strategy = InterpolatedPosteriorStrategy(self, grid=grid, interp_left=interp_left,
                                                                     interp_right=interp_right)
        return self._posterior_strategy

    def representation(self):
        if self.J_left is None and self.C_left is None and self.J_right is None \
                and self.C_right is None and self.added_diag is None:
            return self.c,

        if self.J_left is None and self.C_left is None and self.J_right is None \
                and self.C_right is None and self.added_diag is None:
            return self.c,

        if self.J_left is not None and self.C_left is not None:
            W_left = Variable(index_coef_to_sparse(self.J_left, self.C_left, self.c.size(-1)))
        else:
            W_left = Variable(sparse_eye(self.c.size(-1)))
        if self.J_right is not None and self.C_right is not None:
            W_right = Variable(index_coef_to_sparse(self.J_right, self.C_right, self.c.size(-1)))
        else:
            W_right = Variable(sparse_eye(self.c.size(-1)))

        if self.added_diag is not None:
            return self.c, W_left, W_right, self.added_diag
        else:
            return self.c, W_left, W_right

    def repeat(self, *sizes):
        """
        Repeat elements of the Variable.
        Right now it only works to create a batched version of a ToeplitzLazyVariable.

        e.g. `var.repeat(3, 1, 1)` creates a batched version of length 3
        """
        if not len(sizes) == 3 and sizes[1] == 1 and sizes[2] == 1:
            raise RuntimeError('Repeat only works to create a batched version at the moment.')
        if self.J_left is None:
            raise RuntimeError('Repeat only works when the toeplitz variable is interpolated.')

        return ToeplitzLazyVariable(self.c.repeat(sizes[0], 1), self.J_left.repeat(*sizes), self.C_left.repeat(*sizes),
                                    self.J_right.repeat(*sizes), self.C_right.repeat(*sizes),
                                    self.added_diag)

    def size(self):
        if self.J_left is not None:
            if self.J_left.ndimension() == 3:
                return torch.Size((self.J_left.size(0), self.J_left.size(1), self.J_right.size(1)))
            else:
                return torch.Size((self.J_left.size(0), self.J_right.size(0)))
        if self.c.ndimension() == 2:
            return torch.Size((self.c.size(0), self.c.size(-1), self.c.size(-1)))
        else:
            return torch.Size((self.c.size(-1), self.c.size(-1)))

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if not self.is_batch():
                i = tuple([0] + list(i))
            elif len(i) == 2:
                i = tuple(list(i) + [slice(None, None, None)])

            batch_index = i[0]
            first_index = i[1]
            if not isinstance(first_index, slice):
                first_index = slice(first_index, first_index + 1, None)
            second_index = i[2]
            if not isinstance(second_index, slice):
                second_index = slice(second_index, second_index + 1, None)

            columns = self.c.unsqueeze(0) if self.c.ndimension() == 1 else self.c
            if self.J_left is None:
                # Pretend that the matrix is WTW, where W is an identity matrix, with appropriate slices
                # J[first_index, :], C[first_index, :]
                J_left_new = self.c.data.new(range(self.c.size(-1))[first_index]).unsqueeze(1)
                J_right_new = self.c.data.new(range(self.c.size(-1))[second_index]).unsqueeze(1)
                if isinstance(batch_index, slice):
                    batch_size = len(range(self.c.size(0))[batch_index])
                    J_left_new.unsqueeze_(0)
                    J_left_new = J_left_new.expand(batch_size, J_left_new.size(1), 1)
                    J_right_new.unsqueeze_(0)
                    J_right_new = J_right_new.expand(batch_size, J_right_new.size(1), 1)
                C_left_new = self.c.data.new().resize_as_(J_left_new).fill_(1)
                J_left_new = J_left_new.long()
                # J[second_index, :] C[second_index, :]
                C_right_new = self.c.data.new().resize_as_(J_right_new).fill_(1)
                J_right_new = J_right_new.long()
            else:
                J_left = self.J_left.unsqueeze(0) if self.J_left.ndimension() == 2 else self.J_left
                C_left = self.C_left.unsqueeze(0) if self.C_left.ndimension() == 2 else self.C_left
                J_right = self.J_right.unsqueeze(0) if self.J_right.ndimension() == 2 else self.J_right
                C_right = self.C_right.unsqueeze(0) if self.C_right.ndimension() == 2 else self.C_right
                # J[first_index, :], C[first_index, :]
                J_left_new = J_left[batch_index, first_index]
                C_left_new = C_left[batch_index, first_index]

                # J[second_index, :] C[second_index, :]
                J_right_new = J_right[batch_index, second_index]
                C_right_new = C_right[batch_index, second_index]

            if self.added_diag is not None:
                if len(J_left_new) != len(J_right_new):
                    raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                        diagonal component to make it non-square is probably not intended.\
                                        It is ambiguous which diagonal elements to choose')

                diag_new = self.added_diag[first_index]
            else:
                diag_new = None

            return ToeplitzLazyVariable(columns[batch_index], J_left_new, C_left_new,
                                        J_right_new, C_right_new, diag_new)

        else:
            if not isinstance(i, slice):
                i = slice(i, i + 1, None)

            if self.J_left is not None:
                # Batch mode
                if self.J_left.ndimension() == 3:
                    return ToeplitzLazyVariable(self.c[i], self.J_left[i], self.C_left[i],
                                                self.J_right[i], self.C_right[i], self.added_diag)
                else:
                    J_left_new = self.J_left[i]
                    C_left_new = self.C_left[i]
                    if self.added_diag is not None:
                        raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                            diagonal component to make it non-square is probably not intended.\
                                            It is ambiguous which diagonal elements to choose')
                    else:
                        diag_new = None

                return ToeplitzLazyVariable(self.c, J_left_new, C_left_new,
                                            self.J_right, self.C_right, diag_new)
            else:
                # Batch modej
                if self.c.ndimension() == 2:
                    return ToeplitzLazyVariable(self.c[i], None, None, None, None, self.added_diag)
                else:
                    raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                        unintended. If that was the intent, use evaluate() and slice the full matrix.')
