from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from .root_lazy_variable import RootLazyVariable
from ..utils import prod


class MulLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars, **kwargs):
        '''
        Args:
            - lazy_vars (A list of LazyVariable) - A list of LazyVariable to multiplicate with.
        '''
        lazy_vars = list(lazy_vars)
        if len(lazy_vars) == 1:
            raise RuntimeError('MulLazyVariable should have more than one lazy variables')

        for i, lazy_var in enumerate(lazy_vars):
            if not isinstance(lazy_var, LazyVariable):
                if isinstance(lazy_var, Variable):
                    lazy_vars[i] = NonLazyVariable(lazy_var)
                else:
                    raise RuntimeError('All arguments of a MulLazyVariable should be lazy variables or vairables')

        super(MulLazyVariable, self).__init__(*lazy_vars)
        self.lazy_vars = lazy_vars

    @property
    def left_lazy_var(self):
        return self._mul_args_memo[0]

    @property
    def right_lazy_var(self):
        return self._mul_args_memo[1]

    @property
    def _args(self):
        if not hasattr(self, '_mul_args_memo'):
            # Sort lazy variables by root decomposition size (rank)
            lazy_vars = sorted(self.lazy_vars, key=lambda lv: lv.root_decomposition_size())

            # Recursively construct lazy variables
            # Make sure the recursive components get a mix of low_rank and high_rank variables
            if len(lazy_vars) > 2:
                interleaved_lazy_vars = lazy_vars[0::2] + lazy_vars[1::2]
                if len(interleaved_lazy_vars) > 3:
                    left_lazy_var = MulLazyVariable(*interleaved_lazy_vars[:len(interleaved_lazy_vars) // 2])
                    if left_lazy_var.root_decomposition_size() < left_lazy_var.size(-1):
                        left_lazy_var = RootLazyVariable(left_lazy_var.root_decomposition())
                    else:
                        left_lazy_var = NonLazyVariable(left_lazy_var.evaluate())
                else:
                    # Make sure we're not constructing a MulLazyVariable of length 1
                    left_lazy_var = interleaved_lazy_vars[0]

                right_lazy_var = MulLazyVariable(*interleaved_lazy_vars[len(interleaved_lazy_vars) // 2:])
                if right_lazy_var.root_decomposition_size() < right_lazy_var.size(-1):
                    right_lazy_var = RootLazyVariable(right_lazy_var.root_decomposition())
                else:
                    right_lazy_var = NonLazyVariable(right_lazy_var.evaluate())
            else:
                left_lazy_var = lazy_vars[0]
                right_lazy_var = lazy_vars[1]

            # Choose which we're doing: root decomposition or exact
            if left_lazy_var.root_decomposition_size() < left_lazy_var.size(-1):
                left_lazy_var = RootLazyVariable(left_lazy_var.root_decomposition())
                right_lazy_var = RootLazyVariable(right_lazy_var.root_decomposition())
            else:
                left_lazy_var = NonLazyVariable(left_lazy_var.evaluate())
                right_lazy_var = NonLazyVariable(right_lazy_var.evaluate())

            self._mul_args_memo = [
                left_lazy_var,
                right_lazy_var,
            ]

        return self._mul_args_memo

    @_args.setter
    def _args(self, args):
        # This is a no-op. We do something different here
        pass

    def _matmul_closure_factory(self, left_repr, right_repr):
        right_matmul_closure = self.right_lazy_var._matmul_closure_factory(right_repr)

        def closure(rhs_mat):
            is_vector = False
            if rhs_mat.ndimension() == 1:
                rhs_mat = rhs_mat.unsqueeze(1)
                is_vector = True
            batch_size = max(rhs_mat.size(0), self.size(0)) if rhs_mat.ndimension() == 3 else None

            # Here we have a root decomposition
            if isinstance(self.left_lazy_var, RootLazyVariable):
                rank = left_repr.size(-1)
                n = self.size(-1)
                m = rhs_mat.size(-1)
                # Now implement the formula (A . B) v = diag(A D_v B)
                left_res = (rhs_mat.unsqueeze(-2) * left_repr.unsqueeze(-1))
                left_res = left_res.view(n, rank * m) if batch_size is None else left_res.view(batch_size, n, rank * m)
                left_res = right_matmul_closure(left_res)
                left_res = left_res.view(n, rank, m) if batch_size is None else left_res.view(batch_size, n, rank, m)
                res = left_res.mul_(left_repr.unsqueeze(-1)).sum(-2)
            # This is the case where we're not doing a root decomposition, because the matrix is too small
            else:
                res = (left_repr * right_repr).matmul(rhs_mat)
            res = res.squeeze(-1) if is_vector else res
            return res

        return closure

    def _derivative_quadratic_form_factory(self, left_repr, right_repr):
        left_deriv_closure = self.left_lazy_var._derivative_quadratic_form_factory(left_repr)
        right_deriv_closure = self.right_lazy_var._derivative_quadratic_form_factory(right_repr)

        def closure(left_vecs, right_vecs):
            if left_vecs.ndimension() == 1:
                left_vecs = left_vecs.unsqueeze(0)
                right_vecs = right_vecs.unsqueeze(0)
            left_vecs_t = left_vecs.transpose(-1, -2)
            right_vecs_t = right_vecs.transpose(-1, -2)
            batch_size = self.size(0) if self.ndimension() == 3 else None

            n = left_vecs.size(-1)
            vecs_num = left_vecs.size(-2)

            if isinstance(self.right_lazy_var, RootLazyVariable):
                right_rank = right_repr.size(-1)
                left_factor = left_vecs_t.unsqueeze(-2) * right_repr.unsqueeze(-1)
                right_factor = right_vecs_t.unsqueeze(-2) * right_repr.unsqueeze(-1)
            else:
                right_rank = n
                eye = right_repr.new(n).fill_(1).diag()
                left_factor = left_vecs_t.unsqueeze(-2) * right_repr.unsqueeze(-1)
                right_factor = right_vecs_t.unsqueeze(-2) * eye.unsqueeze(-1)

            if batch_size is None:
                left_factor = left_factor.view(n, vecs_num * right_rank)
                right_factor = right_factor.view(n, vecs_num * right_rank)
            else:
                left_factor = left_factor.view(batch_size, n, vecs_num * right_rank)
                right_factor = right_factor.view(batch_size, n, vecs_num * right_rank)
            left_deriv_args = left_deriv_closure(left_factor.transpose(-1, -2), right_factor.transpose(-1, -2))

            if isinstance(self.left_lazy_var, RootLazyVariable):
                left_rank = left_repr.size(-1)
                left_factor = left_vecs_t.unsqueeze(-2) * left_repr.unsqueeze(-1)
                right_factor = right_vecs_t.unsqueeze(-2) * left_repr.unsqueeze(-1)
            else:
                left_rank = n
                eye = left_repr.new(n).fill_(1).diag()
                left_factor = left_vecs_t.unsqueeze(-2) * left_repr.unsqueeze(-1)
                right_factor = right_vecs_t.unsqueeze(-2) * eye.unsqueeze(-1)

            if batch_size is None:
                left_factor = left_factor.view(n, vecs_num * left_rank)
                right_factor = right_factor.view(n, vecs_num * left_rank)
            else:
                left_factor = left_factor.view(batch_size, n, vecs_num * left_rank)
                right_factor = right_factor.view(batch_size, n, vecs_num * left_rank)
            right_deriv_args = right_deriv_closure(left_factor.transpose(-1, -2), right_factor.transpose(-1, -2))

            return tuple(list(left_deriv_args) + list(right_deriv_args))
        return closure

    def diag(self):
        res = prod([lazy_var.diag() for lazy_var in self.lazy_vars])
        return res

    def evaluate(self):
        res = prod([lazy_var.evaluate() for lazy_var in self.lazy_vars])
        return res

    def mul(self, other):
        if isinstance(other, int) or isinstance(other, float) or (isinstance(other, Variable) and other.numel() == 1):
            lazy_vars = list(self.lazy_vars[:-1])
            lazy_vars.append(self.lazy_vars[-1] * other)
            return MulLazyVariable(*lazy_vars)
        elif isinstance(other, MulLazyVariable):
            res = list(self.lazy_vars) + list(other.lazy_vars)
            return MulLazyVariable(*res)
        elif isinstance(other, LazyVariable):
            return MulLazyVariable(*(list(self.lazy_vars) + [other]))
        else:
            raise RuntimeError('other must be a LazyVariable, int or float.')

    def _size(self):
        return self.lazy_vars[0].size()

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = prod([lazy_var._batch_get_indices(batch_indices, left_indices, right_indices)
                    for lazy_var in self.lazy_vars])
        return res

    def _get_indices(self, left_indices, right_indices):
        res = prod([lazy_var._get_indices(left_indices, right_indices)
                    for lazy_var in self.lazy_vars])
        return res

    def _transpose_nonbatch(self):
        # mul_lazy_variable only works with symmetric matrices
        return self
