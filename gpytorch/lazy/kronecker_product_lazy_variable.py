import torch
import math
from .lazy_variable import LazyVariable
from .mul_lazy_variable import MulLazyVariable
from torch.autograd import Variable
from ..posterior import InterpolatedPosteriorStrategy
from ..utils import sparse_eye
from ..utils.kronecker_product import sym_kronecker_product_toeplitz_matmul, kp_interpolated_toeplitz_matmul, \
    kp_sym_toeplitz_derivative_quadratic_form, list_of_indices_and_values_to_sparse


class KroneckerProductLazyVariable(LazyVariable):
    def __init__(self, columns, J_lefts=None, C_lefts=None, J_rights=None, C_rights=None, added_diag=None):
        super(KroneckerProductLazyVariable, self).__init__(columns, J_lefts, C_lefts, J_rights, C_rights, added_diag)
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
            self._size = (self.J_lefts.size()[1], self.J_rights.size()[1])
        else:
            self._size = (self.kronecker_product_size, self.kronecker_product_size)

    def _matmul_closure_factory(self, *args):
        if len(args) == 1:
            columns, = args

            def closure(mat2):
                return sym_kronecker_product_toeplitz_matmul(columns, mat2)

        elif len(args) == 3:
            columns, W_lefts, W_rights = args

            def closure(mat2):
                return kp_interpolated_toeplitz_matmul(columns, mat2, W_lefts, W_rights, None)

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
            elif len(args) == 3:
                columns, W_left, W_right = args
                left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()

                res = kp_sym_toeplitz_derivative_quadratic_form(columns, left_factor, right_factor)
                return res, None, None
            elif len(args) == 4:
                columns, W_left, W_right, added_diag, = args
                diag_grad = columns.new(len(added_diag)).zero_()
                diag_grad[0] = (left_factor * right_factor).sum()

                left_factor = torch.dsmm(W_left.t(), left_factor.t()).t()
                right_factor = torch.dsmm(W_right.t(), right_factor.t()).t()

                res = kp_sym_toeplitz_derivative_quadratic_form(columns, left_factor, right_factor)
                return res, None, None, diag_grad

        return closure

    def add_diag(self, diag):
        if self.J_lefts is not None:
            kronecker_product_diag = diag.expand(self._size[0])
        else:
            kronecker_product_diag = diag.expand(self.kronecker_product_size)

        return KroneckerProductLazyVariable(self.columns, self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, kronecker_product_diag)

    def add_jitter(self):
        jitter = self.columns.data.new(self.columns.size()).zero_()
        jitter[:, 0] = 1e-4
        return KroneckerProductLazyVariable(self.columns.add(Variable(jitter)), self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, self.added_diag)

    def diag(self):
        """
        Gets the diagonal of the Kronecker Product matrix wrapped by this object.
        """
        if len(self.J_lefts[0]) != len(self.J_rights[0]):
            raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
        d, n_data, n_interp = self.J_lefts.size()
        n_grid = len(self.columns[0])

        left_interps_values = self.C_lefts.unsqueeze(3)
        right_interps_values = self.C_rights.unsqueeze(2)
        interps_values = torch.matmul(left_interps_values, right_interps_values)

        left_interps_indices = self.J_lefts.unsqueeze(3).expand(d, n_data, n_interp, n_interp)
        right_interps_indices = self.J_rights.unsqueeze(2).expand(d, n_data, n_interp, n_interp)

        toeplitz_indices = (left_interps_indices - right_interps_indices).fmod(n_grid).abs().long()
        toeplitz_vals = Variable(self.columns.data.new(d, n_data * n_interp * n_interp).zero_())

        mask = self.columns.data.new(d, n_data * n_interp * n_interp).zero_()
        for i in range(d):
            mask[i] += torch.ones(n_data * n_interp * n_interp)
            temp = self.columns.index_select(1, Variable(toeplitz_indices.view(d, -1)[i]))
            toeplitz_vals += Variable(mask) * temp.view(toeplitz_indices.size())
            mask[i] -= torch.ones(n_data * n_interp * n_interp)

        diag = (Variable(interps_values) * toeplitz_vals).sum(3).sum(2)
        diag = diag.prod(0)

        if self.added_diag is not None:
            diag += self.added_diag

        return diag

    def mul(self, other):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the first Toeplitz component of this KroneckerProductLazyVariable by the constant.
        Args:
            - other (broadcastable with self.columns[0]) - Constant to multiply by.
        Returns:
            - KroneckerProductLazyVariable with columns[0] = columns[0]*(constant) and columns[i] = columns[i] for i>0
        """
        if isinstance(other, LazyVariable):
            return MulLazyVariable(self, other)
        else:
            columns = self.columns
            mask = torch.zeros(columns.size())
            mask[0] = mask[0] + 1
            mask = Variable(mask)
            other = mask * (other - 1).expand_as(mask) + 1
            columns = columns * other
            return KroneckerProductLazyVariable(columns, self.J_lefts, self.C_lefts,
                                                self.J_rights, self.C_rights, self.added_diag)

    def posterior_strategy(self):
        if not hasattr(self, '_posterior_strategy'):
            columns, interp_left, interp_right = self.representation()[:3]
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
            return self.columns, W_left, W_right, self.added_diag
        else:
            return self.columns, W_left, W_right

    def size(self):
        return self._size

    def _transpose_nonbatch(self):
        return KroneckerProductLazyVariable(self.columns, J_lefts=self.J_rights,
                                            C_lefts=self.C_rights, J_rights=self.J_lefts,
                                            C_rights=self.C_lefts, added_diag=self.added_diag)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            first_index = i[0]
            if not isinstance(first_index, slice):
                first_index = slice(first_index, first_index + 1, None)
            second_index = i[1]
            if not isinstance(second_index, slice):
                second_index = slice(second_index, second_index + 1, None)

            if self.J_lefts is None:
                d, m0 = self.columns.size()
                len_i0 = len(range(self.kronecker_product_size)[first_index])
                len_i1 = len(range(self.kronecker_product_size)[second_index])
                J_lefts_new = self.columns.data.new(d, len_i0).zero_()
                J_rights_new = self.columns.data.new(d, len_i1).zero_()
                for j in range(d):
                    J_lefts_new_tensor = torch.arange(0, self.kronecker_product_size)[first_index] / pow(m0, d - j - 1)
                    J_lefts_new[j] = self.columns.data.new(J_lefts_new_tensor)
                    J_rights_new_tensor = torch.arange(0, self.kronecker_product_size)[second_index] / pow(m0,
                                                                                                           d - j - 1)
                    J_rights_new[j] = self.columns.data.new(J_rights_new_tensor)
                C_lefts_new = self.columns.data.new().resize_as_(J_lefts_new).fill_(1).unsqueeze(2)
                C_rights_new = self.columns.data.new().resize_as_(J_lefts_new).fill_(1).unsqueeze(2)
                J_lefts_new = J_lefts_new.long().unsqueeze(2)
                J_rights_new = J_rights_new.long().unsqueeze(2)
            else:
                # J[:, i[0], :], C[:, i[0], :]
                J_lefts_new = self.J_lefts[:, first_index, :]
                C_lefts_new = self.C_lefts[:, first_index, :]

                # J[:, i[1], :], C[:, i[1], :]
                J_rights_new = self.J_rights[:, second_index, :]
                C_rights_new = self.C_rights[:, second_index, :]

            if self.added_diag is not None:
                if len(J_lefts_new[0]) != len(J_rights_new[0]):
                    raise RuntimeError('Slicing in to interpolated Toeplitz matrix that has an additional \
                                        diagonal component to make it non-square is probably not intended.\
                                        It is ambiguous which diagonal elements to choose')

                diag_new = self.added_diag[first_index]
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
