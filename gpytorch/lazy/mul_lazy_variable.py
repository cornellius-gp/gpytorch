from .lazy_variable import LazyVariable
from ..posterior import DefaultPosteriorStrategy
import torch
from torch.autograd import Variable
from ..utils.trace import trace_components


class MulLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars, **kwargs):
        if not all([isinstance(lazy_var, LazyVariable) for lazy_var in lazy_vars]):
            raise RuntimeError('All arguments of a MulLazyVariable should be lazy variables')
        self.lazy_vars = lazy_vars
        if 'added_diag' in kwargs:
            self.added_diag = kwargs['added_diag']
        else:
            self.added_diag = None

    def _matmul_closure_factory(self, *args):
        len_mul_args = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            len_mul_args = len_mul_args + len_repr
        if len(args) > len_mul_args:
            added_diag = args[-1]
            args = args[:-1]
        else:
            added_diag = None

        if len(self.lazy_vars) == 1:
            def closure(rhs_mat):
                if_vector = False
                if rhs_mat.ndimension() == 1:
                    rhs_mat = rhs_mat.unsqueeze(1)
                    if_vector = True
                res_mul = self.lazy_vars[0]._matmul_closure_factory(*args)(rhs_mat)
                if added_diag is not None:
                    res_diag = rhs_mat.mul(added_diag.expand_as(rhs_mat.t()).t())
                    res = res_mul + res_diag
                else:
                    res = res_mul
                res = res.squeeze(1) if if_vector else res
                return res
            return closure

        first_len_repr = len(self.lazy_vars[0].representation())
        first_args = args[0:first_len_repr]

        first_closure = self.lazy_vars[0]._matmul_closure_factory(*first_args)
        second_closure = MulLazyVariable(*(self.lazy_vars[1:]))._matmul_closure_factory(*(args[first_len_repr:]))

        def closure(rhs_mat):
            if_vector = False
            if rhs_mat.ndimension() == 1:
                rhs_mat = rhs_mat.unsqueeze(1)
                if_vector = True
            n, m = rhs_mat.size()

            def left_matmul_closure(samples_matrix):
                n, s = samples_matrix.size()
                return samples_matrix.expand(m, n, s).contiguous()

            def right_matmul_closure(samples_matrix):
                _, s = samples_matrix.size()
                second_mul_sample_matrix = second_closure(samples_matrix)
                rhs_mat_expand = rhs_mat.expand(s, n, m).transpose(0, 2).contiguous()
                second_mul_rhs_mat = rhs_mat_expand.mul(second_mul_sample_matrix)
                res_high = second_mul_rhs_mat.transpose(1, 2).contiguous().view(m * s, n).transpose(0, 1)
                res = first_closure(res_high.contiguous()).transpose(0, 1).contiguous()
                return res.view(m, s, n).transpose(1, 2).contiguous()

            left_matrix, right_matrix = trace_components(left_matmul_closure, right_matmul_closure,
                                                         size=n, tensor_cls=type(rhs_mat))
            res_mul = (left_matrix * right_matrix).sum(2).transpose(0, 1).contiguous()

            if added_diag is not None:
                res_diag = rhs_mat.mul(added_diag.expand_as(rhs_mat.t()).t())
                res = res_mul + res_diag
            else:
                res = res_mul
            res = res.squeeze(1) if if_vector else res
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        args_index = []
        args_index.append(0)
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            i = i + len_repr
            args_index.append(i)
        if len(args) > args_index[-1]:
            added_diag = args[-1]
            args = args[:-1]
        else:
            added_diag = None

        def closure(left_vecs, right_vecs):
            if left_vecs.ndimension() == 1:
                left_vecs = left_vecs.unsqueeze(0)
                right_vecs = right_vecs.unsqueeze(0)
            vecs_num, n = left_vecs.size()

            res = []
            for i in range(len(self.lazy_vars)):
                i1 = args_index[i]
                i2 = args_index[i + 1]
                first_deriv_closre = self.lazy_vars[i]._derivative_quadratic_form_factory(*args[i1:i2])
                second_lazy_vars = list(self.lazy_vars[:i]) + list(self.lazy_vars[i + 1:])
                second_args = list(args[:i1]) + list(args[i2:])

                second_mul_closure = MulLazyVariable(*second_lazy_vars)._matmul_closure_factory(*second_args)

                def left_matmul_closure(samples_matrix):
                    _, s = samples_matrix.size()
                    left_vecs_expand = left_vecs.expand(s, vecs_num, n).transpose(0, 1).contiguous()
                    return left_vecs_expand.mul(samples_matrix).view(s * vecs_num, n)

                def right_matmul_closure(samples_matrix):
                    _, s = samples_matrix.size()
                    right_vecs_expand = right_vecs.expand(s, vecs_num, n).transpose(0, 1).contiguous()
                    second_var_sample_matrix = second_mul_closure(samples_matrix.t().contiguous()).t().contiguous()
                    return right_vecs_expand.mul(second_var_sample_matrix).view(s * vecs_num, n)

                left_matrix, right_matrix = trace_components(left_matmul_closure, right_matmul_closure,
                                                             size=n, tensor_cls=type(left_vecs))
                deriv_args_i = list(first_deriv_closre(left_matrix, right_matrix))
                res = res + deriv_args_i

            if added_diag is not None:
                diag_grad = torch.zeros(len(added_diag))
                diag_grad[0] = (left_vecs * right_vecs).sum()
                res = res + [diag_grad]

            return tuple(res)
        return closure

    def add_diag(self, diag):
        if self.added_diag is None:
            return MulLazyVariable(*self.lazy_vars, added_diag=diag.expand(self.size()[0]))
        else:
            return MulLazyVariable(*self.lazy_vars, added_diag=self.added_diag + diag)

    def add_jitter(self):
        new_lazy_vars = list(lazy_var.add_jitter() for lazy_var in self.lazy_vars)
        return MulLazyVariable(*new_lazy_vars, added_diag=self.added_diag)

    def diag(self):
        res = Variable(torch.ones(self.size()[0]))
        for lazy_var in self.lazy_vars:
            res = res * lazy_var.diag()

        if self.added_diag is not None:
            res = res + self.added_diag
        return res

    def evaluate(self):
        res = Variable(torch.ones(self.size()))
        for lazy_var in self.lazy_vars:
            res = res * lazy_var.evaluate()

        if self.added_diag is not None:
            res = res + self.added_diag.diag()
        return res

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        raise NotImplementedError

    def mul(self, other):
        if isinstance(other, int) or isinstance(other, float):
            lazy_vars = list(self.lazy_vars[:-1])
            lazy_vars.append(self.lazy_vars[-1] * other)
            added_diag = self.added_diag * other
            return MulLazyVariable(*lazy_vars, added_diag=added_diag)
        elif isinstance(other, MulLazyVariable):
            if self.added_diag is not None:
                res = list((self, other))
                return MulLazyVariable(*res)
            return MulLazyVariable(*(list(self.lazy_vars) + list(other.lazy_vars)))
        elif isinstance(other, LazyVariable):
            if self.added_diag is not None:
                res = list((self, other))
                return MulLazyVariable(*res)
            return MulLazyVariable(*(list(self.lazy_vars) + [other]))
        else:
            raise RuntimeError('other must be a LazyVariable, int or float.')

    def representation(self):
        res = list(var for lazy_var in self.lazy_vars for var in lazy_var.representation())
        if self.added_diag is not None:
            res = res + [self.added_diag]
        return tuple(res)

    def posterior_strategy(self):
        return DefaultPosteriorStrategy(self)

    def size(self):
        return self.lazy_vars[0].size()

    def __getitem__(self, i):
        sliced_lazy_vars = [lazy_var.__getitem__(i) for lazy_var in self.lazy_vars]
        if self.added_diag is not None:
            if isinstance(i, tuple):
                first_index = i[0]
                if not isinstance(first_index, slice):
                    first_index = slice(first_index, first_index + 1, None)
                second_index = i[1]
                if not isinstance(second_index, slice):
                    second_index = slice(second_index, second_index + 1, None)
                if first_index == second_index:
                    return MulLazyVariable(*sliced_lazy_vars, added_diag=self.added_diag[first_index])
            raise RuntimeError('Slicing in to a hadamard product of matrces that has an additional \
                                diagonal component to make it non-square is probably not intended.\
                                It is ambiguous which diagonal elements to choose')
        return MulLazyVariable(*sliced_lazy_vars)
