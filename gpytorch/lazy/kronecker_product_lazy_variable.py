import torch
import math
import gpytorch
from .lazy_variable import LazyVariable
from .toeplitz_lazy_variable import ToeplitzLazyVariable
from torch.autograd import Variable
from ..posterior import InterpolatedPosteriorStrategy
from ..utils import sparse_eye
from ..utils.kronecker_product import sym_kronecker_product_toeplitz_matmul, kp_interpolated_toeplitz_matmul, \
    kp_sym_toeplitz_derivative_quadratic_form, list_of_indices_and_values_to_sparse, kronecker_product


class KroneckerProductLazyVariable(LazyVariable):
    def __init__(self, columns, J_lefts=None, C_lefts=None, J_rights=None, C_rights=None, added_diag=None):
        if not isinstance(columns, Variable):
            raise RuntimeError('KroneckerProductLazyVariable is intended to wrap Variable versions of \
                                the first column and row.')

        self.columns = columns
        self.J_lefts = J_lefts
        self.C_lefts = C_lefts
        self.J_rights = J_rights
        self.C_rights = C_rights
        self.added_diag = added_diag
        self.kronecker_product_size = int(math.pow(self.columns.size()[1], self.columns.size()[0]))
        if J_lefts is not None:
            self.size = (self.J_lefts.size()[1], self.J_rights.size()[1])
        else:
            self.size = (self.kronecker_product_size, self.kronecker_product_size)

    def _matmul_closure_factory(self, *args):
        if len(args) == 1:
            columns, = args

            def closure(mat2):
                return sym_kronecker_product_toeplitz_matmul(columns, mat2)

        elif len(args) == 4:
            columns, W_lefts, W_rights, added_diag = args

            def closure(mat2):
                return kp_interpolated_toeplitz_matmul(columns, mat2, W_lefts, W_rights, added_diag)

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
                columns, = args
                return kp_sym_toeplitz_derivative_quadratic_form(columns, left_factor, right_factor),
            elif len(args) == 4:
                columns, W_left, W_right, added_diag, = args

                if added_diag is not None:
                    diag_grad = torch.zeros(len(added_diag))
                    diag_grad[0] = (left_factor * right_factor).sum()
                else:
                    diag_grad = None

                left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()

                res = kp_sym_toeplitz_derivative_quadratic_form(columns, left_factor, right_factor)
                return tuple([res] + [None] * 2 + [diag_grad])
        return closure

    def add_diag(self, diag):
        if self.J_lefts is not None:
            kronecker_product_diag = diag.expand(self.size[0])
        else:
            kronecker_product_diag = diag.expand_as(self.kronecker_product_size)

        return KroneckerProductLazyVariable(self.columns, self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, kronecker_product_diag)

    def add_jitter(self):
        jitter = torch.zeros(self.columns.size())
        jitter[:, 0] = 1e-4
        return KroneckerProductLazyVariable(self.columns.add(Variable(jitter)), self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, self.added_diag)

    def diag(self):
        """
        Gets the diagonal of the Kronecker Product matrix wrapped by this object.
        """
        if len(self.J_lefts[0]) != len(self.J_rights[0]):
            raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
        WKW_diag = Variable(torch.zeros(len(self.J_rights[0])))
        for i in range(len(self.J_rights[0])):
            WKW_diag[i] = self[i:i + 1, i:i + 1].evaluate()

        return WKW_diag

    def explicit_interpolate_K(self, Js, Cs):
        """
        Multiplies the Kronecker Product matrix K this object represents (by columns and rows)
        by an interpolation matrix W, to get WK, without explicitly forming the Kronecker Product
        matrix K. This is a much more space-efficient approach.

        Args:
            - Js (matrix d x n x k) - d Index matrices for interpolation matrix W
            - Cs (matrix d x n x k) - d Coefficients matrices for interpolation matrix W
        Returns:
            - Matrix (n x m) - The result of the multiplication WK
        """
        m = self.kronecker_product_size
        d, n, num_coefficients = Js.size()
        m0 = self.columns.size()[1]

        result_matrix = Variable(torch.zeros(n, m))

        result_dim = Variable(torch.zeros(d, n, m0))
        for i in range(d):
            result_dim[i] = ToeplitzLazyVariable(self.columns[i]).explicit_interpolate_T(Js[i], Cs[i])

        for i in range(n):
            result_matrix[i] = kronecker_product(result_dim[:, i, :].unsqueeze(1))

        return result_matrix

    def evaluate(self):
        """
        Explicitly evaluate and return the Kronecer Product matrix this object wraps as a float Tensor.
        To do this, we explicitly compute W_{left}TW_{right}^{T} and return it.

        Warning: as implicitly stored by this LazyVariable, W is very sparse and T requires O(m)
        storage, where as the full matrix requires O(m^2) storage. Calling evaluate can very easily
        lead to memory issues. As a result, using it should be a last resort.
        """

        if self.J_lefts is not None:
            n_left, n_right = self.size
            W_left = list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns)
            W_right = list_of_indices_and_values_to_sparse(self.J_rights, self.C_rights, self.columns)
            if n_left <= n_right:
                W_left_K = self.explicit_interpolate_K(self.J_lefts, self.C_lefts)
                WKW = gpytorch.dsmm(Variable(W_right), W_left_K.t()).t()
            else:
                W_right_K = self.explicit_interpolate_K(self.J_rights, self.C_rights)
                WKW = gpytorch.dsmm(Variable(W_left), W_right_K.t())
        else:
            WKW = KroneckerProductLazyVariable(self.columns).matmul(Variable(torch.eye(self.kronecker_product_size)))

        if self.added_diag is not None:
            WKW = WKW + torch.diag(self.added_diag)

        return WKW

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        epsilon = Variable(torch.randn(self.kronecker_product_size, gpytorch.functions.num_trace_samples))
        samples = chol_var_covar.mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1).expand_as(samples)
        W_left = Variable(list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns))
        samples = gpytorch.dsmm(W_left, samples)
        log_likelihood = log_probability_func(samples, train_y)
        return log_likelihood

    def mul(self, constant):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the first Toeplitz component of this KroneckerProductLazyVariable by the constant.

        Args:
            - constant (broadcastable with self.columns[0]) - Constant to multiply by.
        Returns:
            - KroneckerProductLazyVariable with columns[0] = columns[0]*(constant) and columns[i] = columns[i] for i>0
        """
        columns = self.columns
        constant_tensor = torch.zeros(columns.size()) + 1
        constant_tensor[0] = constant_tensor[0] * constant.data
        constant_variable = Variable(constant_tensor)
        columns = columns * constant_variable
        return KroneckerProductLazyVariable(columns, self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, self.added_diag)

    def mul_(self, constant):
        """
        In-place version of mul.
        """
        self.columns[0].mul_(constant)
        return self

    def posterior_strategy(self):
        if not hasattr(self, '_posterior_strategy'):
            columns, interp_left, interp_right, added_diag = self.representation()
            grid = KroneckerProductLazyVariable(columns)
            self._posterior_strategy = InterpolatedPosteriorStrategy(self, grid=grid, interp_left=interp_left,
                                                                     interp_right=interp_right)
        return self._posterior_strategy

    def representation(self):
        if self.J_lefts is None and self.C_lefts is None and self.J_rights is None \
                and self.C_rights is None and self.added_diag is None:
            return self.columns,

        if self.J_lefts is None and self.C_lefts is None and self.J_rights is None \
                and self.C_rights is None and self.added_diag is None:
            return self.columns,

        if self.J_lefts is not None and self.C_lefts is not None:
            W_left = Variable(list_of_indices_and_values_to_sparse(self.J_lefts,
                                                                   self.C_lefts,
                                                                   self.columns))
        else:
            W_left = Variable(sparse_eye(self.kronecker_product_size))
        if self.J_rights is not None and self.C_rights is not None:
            W_right = Variable(list_of_indices_and_values_to_sparse(self.J_rights,
                                                                    self.C_rights,
                                                                    self.columns))
        else:
            W_right = Variable(sparse_eye(self.kronecker_product_size))
        if self.added_diag is not None:
            added_diag = self.added_diag
        else:
            added_diag = Variable(torch.zeros(1))
        return self.columns, W_left, W_right, added_diag

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if self.J_lefts is None:
                # Pretend that the matrix is WTW, where W is an identity matrix, with appropriate slices
                # # J[:, i[0], :], C[:, i[0], :]
                # J_left_new = self.c.data.new(range(len(self.c))[i[0]]).unsqueeze(1)
                # C_left_new = self.c.data.new().resize_as_(J_left_new).fill_(1)
                # J_left_new = J_left_new.long()

                # # J[:, i[1], :], C[:, i[1], :]
                # J_right_new = self.c.data.new(range(len(self.c))[i[1]]).unsqueeze(1)
                # C_right_new = self.c.data.new().resize_as_(J_right_new).fill_(1)
                # J_right_new = J_right_new.long()

                d, m0 = self.columns.size()
                len_i0 = len(range(self.kronecker_product_size)[i[0]])
                len_i1 = len(range(self.kronecker_product_size)[i[1]])
                J_lefts_new = torch.zeros(d, len_i0)
                J_rights_new = torch.zeros(d, len_i1)
                for j in range(d):
                    J_lefts_new_tensor = torch.arange(0, self.kronecker_product_size)[i[0]] / pow(m0, d - j - 1)
                    J_lefts_new[j] = self.columns.data.new(J_lefts_new_tensor)
                    J_rights_new_tensor = torch.arange(0, self.kronecker_product_size)[i[1]] / pow(m0, d - j - 1)
                    J_rights_new[j] = self.columns.data.new(J_rights_new_tensor)
                C_lefts_new = self.columns.data.new().resize_as_(J_lefts_new).fill_(1).unsqueeze(2)
                C_rights_new = self.columns.data.new().resize_as_(J_lefts_new).fill_(1).unsqueeze(2)
                J_lefts_new = J_lefts_new.long().unsqueeze(2)
                J_rights_new = J_rights_new.long().unsqueeze(2)
            else:
                # J[:, i[0], :], C[:, i[0], :]
                J_lefts_new = self.J_lefts[:, i[0], :]
                C_lefts_new = self.C_lefts[:, i[0], :]

                # J[:, i[1], :], C[:, i[1], :]
                J_rights_new = self.J_rights[:, i[1], :]
                C_rights_new = self.C_rights[:, i[1], :]

            if self.added_diag is not None:
                if len(J_lefts_new[0]) != len(J_rights_new[0]):
                    raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                        diagonal component to make it non-square is probably not intended.\
                                        It is ambiguous which diagonal elements to choose')

                diag_new = self.added_diag[i[0]]
            else:
                diag_new = None

            return KroneckerProductLazyVariable(self.columns, J_lefts_new, C_lefts_new,
                                                J_rights_new, C_rights_new, diag_new)

        else:
            if self.J_lefts is not None:
                J_lefts_new = self.J_lefts[:, i, :]
                C_lefts_new = self.C_lefts[:, i, :]
                if self.added_diag is not None:
                    raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                        diagonal component to make it non-square is probably not intended.\
                                        It is ambiguous which diagonal elements to choose')
                else:
                    diag_new = None

                return KroneckerProductLazyVariable(self.columns, J_lefts_new, C_lefts_new,
                                                    self.J_rights, self.C_rights, diag_new)
            else:
                raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                    unintended. If that was the intent, use evaluate() and slice the full matrix.')
