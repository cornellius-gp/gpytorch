import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.utils import toeplitz
from .lazy_variable import LazyVariable
from ..posterior import InterpolatedPosteriorStrategy
from ..utils import sparse_eye
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
            elif len(args) == 4:
                toeplitz_column, W_left, W_right, added_diag, = args

                if added_diag is not None:
                    diag_grad = torch.zeros(len(added_diag))
                    diag_grad[0] = (left_vectors * right_vectors).sum()
                else:
                    diag_grad = None

                left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()

                return tuple([sym_toeplitz_derivative_quadratic_form(left_factor, right_factor)] + [None] * 2 + [diag_grad])
        return closure

    def add_diag(self, diag):
        if self.J_left is not None:
            toeplitz_diag = diag.expand(len(self.J_left))
        else:
            toeplitz_diag = diag.expand_as(self.c)

        return ToeplitzLazyVariable(self.c, self.J_left, self.C_left,
                                    self.J_right, self.C_right, toeplitz_diag)

    def add_jitter(self):
        jitter = torch.zeros(len(self.c))
        jitter[0] = 1e-4
        return ToeplitzLazyVariable(self.c.add(Variable(jitter)), self.J_left, self.C_left,
                                    self.J_right, self.C_right, self.added_diag)

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.
        """
        if len(self.J_left) != len(self.J_right):
            raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
        WTW_diag = Variable(torch.zeros(len(self.J_right)))
        for i in range(len(self.J_right)):
            WTW_diag[i] = self[i:i + 1, i:i + 1].evaluate()

        return WTW_diag

    def evaluate(self):
        """
        Explicitly evaluate and return the Toeplitz matrix this object wraps as a float Tensor.
        To do this, we explicitly compute W_{left}TW_{right}^{T} and return it.

        Warning: as implicitly stored by this LazyVariable, W is very sparse and T requires O(n)
        storage, where as the full matrix requires O(n^2) storage. Calling evaluate can very easily
        lead to memory issues. As a result, using it should be a last resort.
        """

        if self.J_left is not None:
            n_left = len(self.J_left)
            n_right = len(self.J_right)
            W_left = toeplitz.index_coef_to_sparse(self.J_left, self.C_left, len(self.c))
            W_right = toeplitz.index_coef_to_sparse(self.J_right, self.C_right, len(self.c))
            if n_left <= n_right:
                W_left_T = self.explicit_interpolate_T(self.J_left, self.C_left)
                WTW = gpytorch.dsmm(Variable(W_right), W_left_T.t()).t()
            else:
                W_right_T = self.explicit_interpolate_T(self.J_right, self.C_right)
                WTW = gpytorch.dsmm(Variable(W_left), W_right_T.t())
        else:
            WTW = ToeplitzLazyVariable(self.c).matmul(Variable(torch.eye(len(self.c))))

        if self.added_diag is not None:
            WTW = WTW + torch.diag(self.added_diag)

        return WTW

    def explicit_interpolate_T(self, J, C):
        """
        Multiplies the Toeplitz matrix T this object represents (by a column c and row r)
        by an interpolation matrix W, to get WT, without explicitly forming the Toeplitz
        matrix T. This is a much more space-efficient approach.

        Args:
            - J (matrix n-by-k) - Index matrix for interpolation matrix W
            - C (matrix n-by-k) - Coefficients matrix for interpolation matrix W
        Returns:
            - Matrix (n-by-m) - The result of the multiplication WT
        """
        m = len(self.c)
        n, num_coefficients = J.size()

        result_matrix = Variable(torch.zeros(n, m))

        for i in range(n):
            for j in range(m):
                entry = 0
                for k in range(num_coefficients):
                    row = J[i, k]
                    entry += C[i, k] * toeplitz.sym_toeplitz_getitem(self.c, row, j)
                result_matrix[i, j] = entry

        return result_matrix

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        epsilon = Variable(torch.randn(len(self.c), gpytorch.functions.num_trace_samples))
        samples = chol_var_covar.mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1).expand_as(samples)
        W_left = Variable(toeplitz.index_coef_to_sparse(self.J_left, self.C_left, len(self.c)))
        samples = gpytorch.dsmm(W_left, samples)
        log_likelihood = log_probability_func(samples, train_y)

        return log_likelihood

    def mul(self, constant):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the Toeplitz component by the constant. This way, the interpolation acts on the
        multiplied values in T, and the entire kernel is ultimately multiplied by this constant.

        Args:
            - constant (broadcastable with self.c) - Constant to multiply by.
        Returns:
            - ToeplitzLazyVariable with c = c*(constant)
        """
        return ToeplitzLazyVariable(self.c.mul(constant), self.J_left, self.C_left,
                                    self.J_right, self.C_right, self.added_diag)

    def mul_(self, constant):
        """
        In-place version of mul.
        """
        return self.c.mul_(constant)

    def posterior_strategy(self):
        if not hasattr(self, '_posterior_strategy'):
            toeplitz_column, interp_left, interp_right, added_diag = self.representation()
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
            W_left = Variable(index_coef_to_sparse(self.J_left, self.C_left, len(self.c)))
        else:
            W_left = Variable(sparse_eye(len(self.c)))
        if self.J_right is not None and self.C_right is not None:
            W_right = Variable(index_coef_to_sparse(self.J_right, self.C_right, len(self.c)))
        else:
            W_right = Variable(sparse_eye(len(self.c)))
        if self.added_diag is not None:
            added_diag = self.added_diag
        else:
            added_diag = Variable(torch.zeros(1))
        return self.c, W_left, W_right, added_diag

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if self.J_left is None:
                # Pretend that the matrix is WTW, where W is an identity matrix, with appropriate slices
                # J[i[0], :], C[i[0], :]
                J_left_new = self.c.data.new(range(len(self.c))[i[0]]).unsqueeze(1)
                C_left_new = self.c.data.new().resize_as_(J_left_new).fill_(1)
                J_left_new = J_left_new.long()
                # J[i[1], :] C[i[1], :]
                J_right_new = self.c.data.new(range(len(self.c))[i[1]]).unsqueeze(1)
                C_right_new = self.c.data.new().resize_as_(J_right_new).fill_(1)
                J_right_new = J_right_new.long()
            else:
                # J[i[0], :], C[i[0], :]
                J_left_new = self.J_left[i[0]]
                C_left_new = self.C_left[i[0]]

                # J[i[1], :] C[i[1], :]
                J_right_new = self.J_right[i[1]]
                C_right_new = self.C_right[i[1]]

            if self.added_diag is not None:
                if len(J_left_new) != len(J_right_new):
                    raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                        diagonal component to make it non-square is probably not intended.\
                                        It is ambiguous which diagonal elements to choose')

                diag_new = self.added_diag[i[0]]
            else:
                diag_new = None

            return ToeplitzLazyVariable(self.c, J_left_new, C_left_new,
                                        J_right_new, C_right_new, diag_new)

        else:
            if self.J_left is not None:
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
                raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                    unintended. If that was the intent, use evaluate() and slice the full matrix.')
