from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from ..posterior import DefaultPosteriorStrategy
import torch
from torch.autograd import Variable
from ..utils.trace import trace_components
from gpytorch.utils import StochasticLQ


class MulLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars, **kwargs):
        '''
        Args:
            - lazy_vars (A list of LazyVariable) - A list of LazyVariable to multiplicate with.
            - matmul_mode (String) - deterministic(default), stochastic or approximate
            - max_iter (int) - the maximum iteration in lanczos decomposition in when matmul_mode=approximate
            - num_samples (int) - the samples number when matmul_mode=stochastic
        '''
        lazy_vars = list(lazy_vars)
        for i, lazy_var in enumerate(lazy_vars):
            if not isinstance(lazy_var, LazyVariable):
                if isinstance(lazy_var, Variable):
                    lazy_vars[i] = NonLazyVariable(lazy_var)
                else:
                    raise RuntimeError('All arguments of a SumLazyVariable should be lazy variables or vairables')
        super(MulLazyVariable, self).__init__(*lazy_vars, **kwargs)

        self.lazy_vars = lazy_vars

        if 'matmul_mode' in kwargs:
            self.matmul_mode = kwargs['matmul_mode']
        else:
            self.matmul_mode = 'deterministic'

        if 'added_diag' in kwargs:
            self.added_diag = kwargs['added_diag']
        else:
            self.added_diag = None

        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        else:
            self.max_iter = 15

        if self.matmul_mode == 'approximate':
            if len(lazy_vars) > 1:
                half_d = int(len(lazy_vars) / 2)
                self.left_var = MulLazyVariable(*lazy_vars[:half_d], matmul_mode='approximate',
                                                max_iter=self.max_iter)
                self.right_var = MulLazyVariable(*lazy_vars[half_d:], matmul_mode='approximate',
                                                 max_iter=self.max_iter)
        if self.matmul_mode == 'deterministic':
            self.num_samples = lazy_vars[0].size()[0]
        else:
            if 'num_samples' in kwargs:
                self.num_samples = kwargs['num_samples']
            else:
                self.num_samples = 200

    def _matmul_closure_factory(self, *args):
        n1, n2 = self.size()
        if n1 != n2:
            temp_samples = self.num_samples
            self.num_samples = self.lazy_vars[0].size()[0]
            result = self._stoch_deter_matmul_closure_factory(*args)
            self.num_samples = temp_samples
            return result
        elif self.matmul_mode == 'approximate':
            return self._approx_matmul_closure_factory(*args)
        elif self.matmul_mode == 'stochastic' or self.matmul_mode == 'deterministic':
            return self._stoch_deter_matmul_closure_factory(*args)
        else:
            raise RuntimeError('matmul_mode should be approximate, stochastic or deterministic')

    def _stoch_deter_matmul_closure_factory(self, *args):
        sub_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_closure = lazy_var._matmul_closure_factory(*args[i:i + len_repr])
            sub_closures.append(sub_closure)
            i = i + len_repr
        if len(args) > i:
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
        else:
            def closure(rhs_mat):
                if_vector = False
                if rhs_mat.ndimension() == 1:
                    rhs_mat = rhs_mat.unsqueeze(1)
                    if_vector = True
                n, m = rhs_mat.size()
                dim = len(self.lazy_vars)

                if self.num_samples < n:
                    sample_matrix = torch.sign(rhs_mat.new(dim - 1, self.num_samples, n, m).normal_())
                    num_samples = self.num_samples
                else:
                    sample_matrix = torch.diag(rhs_mat.new(n).fill_(1)).expand(dim - 1, m, n, n).transpose(1, 3)
                    sample_matrix = sample_matrix.contiguous()
                    num_samples = n

                sample_matrix_1 = torch.cat((sample_matrix, rhs_mat.expand(1, num_samples, n, m)), dim=0)
                sample_matrix_2 = torch.cat((rhs_mat.new(1, num_samples, n, m).fill_(1), sample_matrix), dim=0)

                right_factor = (sample_matrix_1 * sample_matrix_2).transpose(1, 2).contiguous()
                right_factor = right_factor.view(dim, n, num_samples * m)

                res_mul = torch.ones(n, num_samples * m)
                for i in range(dim):
                    res_mul *= sub_closures[i](right_factor[i])

                res_mul = res_mul.view(n, num_samples, m).sum(1)

                if self.num_samples < n:
                    res_mul.div_(num_samples)

                if added_diag is not None:
                    res_diag = rhs_mat.mul(added_diag.expand_as(rhs_mat.t()).t())
                    res = res_mul + res_diag
                else:
                    res = res_mul
                res = res.squeeze(1) if if_vector else res
                return res

        return closure

    def _approx_matmul_closure_factory(self, *args):
        args_length = 0
        half_d = int(len(self.lazy_vars) / 2)
        for i in range(half_d):
            args_length += len(self.lazy_vars[i].representation())
        half_args_length = args_length
        for i in range(half_d, len(self.lazy_vars)):
            args_length += len(self.lazy_vars[i].representation())
        if len(args) > args_length:
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
        else:
            def closure(rhs_mat):
                if_vector = False
                if rhs_mat.ndimension() == 1:
                    rhs_mat = rhs_mat.unsqueeze(1)
                    if_vector = True
                Q_1, T_1 = self.left_var._lanczos_quadrature_form(*args[:half_args_length])
                Q_2, T_2 = self.right_var._lanczos_quadrature_form(*args[half_args_length:])

                n, k_1 = Q_1.size()
                _, m = rhs_mat.size()
                _, k_2 = Q_2.size()

                if not hasattr(self, '_Q_2_T_2'):
                    self._Q_2_T_2 = Q_2.matmul(T_2)
                Q_2_T_2 = self._Q_2_T_2
                rhs_mat_expand = rhs_mat.expand(k_2, n, m).transpose(0, 2).contiguous()
                rhs_mat_Q_2_T_2 = rhs_mat_expand.mul(Q_2_T_2)
                if not hasattr(self, '_T_1_Q_1_t'):
                    self._T_1_Q_1_t = T_1.matmul(Q_1.t())
                T_1_Q_1_t = self._T_1_Q_1_t

                m_res = T_1_Q_1_t.matmul(rhs_mat_Q_2_T_2)
                res_mul = Q_1.matmul(m_res).mul(Q_2).sum(2).transpose(0, 1).contiguous()

                if added_diag is not None:
                    res_diag = rhs_mat.mul(added_diag.expand_as(rhs_mat.t()).t())
                    res = res_mul + res_diag
                else:
                    res = res_mul
                res = res.squeeze(1) if if_vector else res
                return res
        return closure

    def _lanczos_quadrature_form(self, *args):
        if not hasattr(self, '_lanczos_quadrature'):
            n = self.size()[0]
            z = args[0].new(n, 1).normal_()
            z = z / torch.norm(z, 2, 0)

            def tensor_matmul_closure(rhs):
                return self._matmul_closure_factory(*args)(rhs)

            Q, T = StochasticLQ(cls=type(z), max_iter=self.max_iter).lanczos_batch(tensor_matmul_closure, z)
            Q = Q[0]
            T = T[0]
            self._lanczos_quadrature = Q, T
        return self._lanczos_quadrature

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

                second_mul_closure = MulLazyVariable(*second_lazy_vars,
                                                     matmul_mode=self.matmul_mode,
                                                     max_iter=self.max_iter,
                                                     num_samples=self.num_samples)._matmul_closure_factory(*second_args)

                def left_matmul_closure(samples_matrix):
                    _, s = samples_matrix.size()
                    left_vecs_expand = left_vecs.expand(s, vecs_num, n).transpose(0, 1).contiguous()
                    return left_vecs_expand.mul(samples_matrix.t()).view(s * vecs_num, n)

                def right_matmul_closure(samples_matrix):
                    _, s = samples_matrix.size()
                    right_vecs_expand = right_vecs.expand(s, vecs_num, n).transpose(0, 1).contiguous()
                    second_var_sample_matrix = second_mul_closure(samples_matrix)
                    return right_vecs_expand.mul(second_var_sample_matrix.t()).view(s * vecs_num, n)

                left_matrix, right_matrix = trace_components(left_matmul_closure, right_matmul_closure,
                                                             size=n, tensor_cls=type(left_vecs))
                deriv_args_i = list(first_deriv_closre(left_matrix, right_matrix))
                res = res + deriv_args_i

            if added_diag is not None:
                diag_grad = added_diag.new(len(added_diag)).fill_(0)
                diag_grad[0] = (left_vecs * right_vecs).sum()
                res = res + [diag_grad]

            return tuple(res)
        return closure

    def add_diag(self, diag):
        if self.added_diag is None:
            return MulLazyVariable(*self.lazy_vars,
                                   matmul_mode=self.matmul_mode,
                                   max_iter=self.max_iter,
                                   num_samples=self.num_samples,
                                   added_diag=diag.expand(self.size()[0]))
        else:
            return MulLazyVariable(*self.lazy_vars,
                                   matmul_mode=self.matmul_mode,
                                   max_iter=self.max_iter,
                                   num_samples=self.num_samples,
                                   added_diag=self.added_diag + diag)

    def add_jitter(self):
        new_lazy_vars = list(lazy_var.add_jitter() for lazy_var in self.lazy_vars)
        return MulLazyVariable(*new_lazy_vars,
                               matmul_mode=self.matmul_mode,
                               max_iter=self.max_iter,
                               num_samples=self.num_samples,
                               added_diag=self.added_diag)

    def diag(self):
        res = Variable(torch.ones(self.size()[0]))
        for lazy_var in self.lazy_vars:
            res = res * lazy_var.diag()

        if self.added_diag is not None:
            res = res + self.added_diag
        return res

    def evaluate(self):
        res = None
        for lazy_var in self.lazy_vars:
            if res is None:
                res = lazy_var.evaluate()
            else:
                res = res * lazy_var.evaluate()

        if self.added_diag is not None:
            res = res + self.added_diag.diag()
        return res

    def mul(self, other):
        if isinstance(other, int) or isinstance(other, float):
            lazy_vars = list(self.lazy_vars[:-1])
            lazy_vars.append(self.lazy_vars[-1] * other)
            added_diag = self.added_diag * other
            return MulLazyVariable(*lazy_vars, added_diag=added_diag, matmul_mode=self.matmul_mode)
        elif isinstance(other, MulLazyVariable):
            if self.added_diag is not None:
                res = list((self, other))
                return MulLazyVariable(*res)
            return MulLazyVariable(*(list(self.lazy_vars) + list(other.lazy_vars)), matmul_mode=self.matmul_mode)
        elif isinstance(other, LazyVariable):
            if self.added_diag is not None:
                res = list((self, other))
                return MulLazyVariable(*res, matmul_mode=self.matmul_mode)
            return MulLazyVariable(*(list(self.lazy_vars) + [other]), matmul_mode=self.matmul_mode)
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

    def _transpose_nonbatch(self):
        lazy_vars_t = list(lazy_var.t() for lazy_var in self.lazy_vars)
        return MulLazyVariable(*lazy_vars_t,
                               matmul_mode=self.matmul_mode,
                               max_iter=self.max_iter,
                               num_samples=self.num_samples,
                               added_diag=self.added_diag)

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
                    return MulLazyVariable(*sliced_lazy_vars,
                                           added_diag=self.added_diag[first_index],
                                           matmul_mode=self.matmul_mode)
            raise RuntimeError('Slicing in to a hadamard product of matrces that has an additional \
                                diagonal component to make it non-square is probably not intended.\
                                It is ambiguous which diagonal elements to choose')
        return MulLazyVariable(*sliced_lazy_vars, matmul_mode=self.matmul_mode)
