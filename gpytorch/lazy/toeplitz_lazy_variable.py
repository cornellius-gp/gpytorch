import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.utils import toeplitz
from .lazy_variable import LazyVariable
from gpytorch.functions.lazy_toeplitz import InterpolatedToeplitzGPMarginalLogLikelihood
from ..utils import function_factory
from ..utils.toeplitz import interpolated_sym_toeplitz_mul, index_coef_to_sparse


def _mm_closure_factory(W_left, W_right, c):
    return lambda mat2: interpolated_sym_toeplitz_mul(c, mat2, W_left, W_right)


_mm_class = function_factory.mm_factory(_mm_closure_factory)
_invmm_class = function_factory.invmm_factory(_mm_closure_factory)


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
            WTW = toeplitz.sym_toeplitz(self.c.data)

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

    def invmm(self, rhs_mat):
        W_test_left = index_coef_to_sparse(self.J_left, self.C_left, len(self.c))
        W_test_right = index_coef_to_sparse(self.J_right, self.C_right, len(self.c))
        return _invmm_class(W_test_left, W_test_right)(self.c, rhs_mat)

    def mm(self, rhs_mat):
        W_test_left = index_coef_to_sparse(self.J_left, self.C_left, len(self.c))
        W_test_right = index_coef_to_sparse(self.J_right, self.C_right, len(self.c))
        return _mm_class(W_test_left, W_test_right)(self.c, rhs_mat)

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
        self.c.mul_(constant)

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.

        By definition of a Toeplitz matrix, every element along the diagonal is equal
        to c[0] == r[0]. Therefore, we return a vector of length len(self.c) with
        each element equal to c[0].

        If the interpolation matrices exist, then the diagonal of WTW^{T} is simply
        W(T_diag)W^{T}.
        """
        if self.J_left is not None:
            if len(self.J_left) != len(self.J_right):
                raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
            WTW_diag = self.c[0].expand(len(self.J_left))
        else:
            WTW_diag = self.c[0].expand_as(self.c)

        if self.added_diag is not None:
            if len(self.added_diag) > len(WTW_diag):
                raise RuntimeError('Additional diagonal component length does not \
                                    match the rest of this implicit tensor.')
            WTW_diag = WTW_diag + self.added_diag

        return WTW_diag

    def gp_marginal_log_likelihood(self, target):
        W_left = Variable(toeplitz.index_coef_to_sparse(self.J_left, self.C_left, len(self.c)))
        W_right = Variable(toeplitz.index_coef_to_sparse(self.J_right, self.C_right, len(self.c)))
        noise_diag = self.added_diag
        return InterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right)(self.c, target, noise_diag)

    def add_diag(self, diag):
        if self.J_left is not None:
            toeplitz_diag = diag.expand(len(self.J_left))
        else:
            toeplitz_diag = diag.expand_as(self.c)

        return ToeplitzLazyVariable(self.c, self.J_left, self.C_left,
                                    self.J_right, self.C_right, toeplitz_diag)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if self.J_left is not None:
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

                return ToeplitzLazyVariable(self.c, J_left_new, C_left_new, J_right_new, C_right_new, diag_new)
            else:
                if i[0] != i[1]:
                    raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                        unintended. If that was the intent, use evaluate() and slice the full matrix.')
                c_new = self.c[i[1]]
                if self.added_diag is not None:
                    diag_new = self.added_diag[i[0]]
                else:
                    diag_new = None

                return ToeplitzLazyVariable(c_new, diag=diag_new)
        else:
            if self.J_left is not None:
                J_left_new = self.J_left[i]
                C_left_new = self.C_left[i]
                if self.added_diag is not None:
                    diag_new = self.added_diag[i]
                else:
                    diag_new = None

                return ToeplitzLazyVariable(self.c, J_left_new, C_left_new,
                                            self.J_right, self.C_right, diag_new)
            else:
                raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                    unintended. If that was the intent, use evaluate() and slice the full matrix.')
