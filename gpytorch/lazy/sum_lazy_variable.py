from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from torch.autograd import Variable


class SumLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars):
        lazy_vars = list(lazy_vars)
        for i, lazy_var in enumerate(lazy_vars):
            if not isinstance(lazy_var, LazyVariable):
                if isinstance(lazy_var, Variable):
                    lazy_vars[i] = NonLazyVariable(lazy_var)
                else:
                    raise RuntimeError('All arguments of a SumLazyVariable should be lazy variables or vairables')
        super(SumLazyVariable, self).__init__(*lazy_vars)

        self.lazy_vars = lazy_vars

    def _matmul_closure_factory(self, *args):
        sub_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_closure = lazy_var._matmul_closure_factory(*args[i:i + len_repr])
            sub_closures.append(sub_closure)
            i = i + len_repr

        def closure(rhs_mat):
            return sum(sub_closure(rhs_mat) for sub_closure in sub_closures)
        return closure

    def _t_matmul_closure_factory(self, *args):
        sub_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_closure = lazy_var._t_matmul_closure_factory(*args[i:i + len_repr])
            sub_closures.append(sub_closure)
            i = i + len_repr

        def closure(rhs_mat):
            return sum(sub_closure(rhs_mat) for sub_closure in sub_closures)
        return closure

    def _derivative_quadratic_form_factory(self, *args):
        sub_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_closure = lazy_var._derivative_quadratic_form_factory(*args[i:i + len_repr])
            sub_closures.append(sub_closure)
            i = i + len_repr

        def closure(*closure_args):
            return tuple(var for sub_closure in sub_closures for var in sub_closure(*closure_args))
        return closure

    def _size(self):
        return self.lazy_vars[0].size()

    def _transpose_nonbatch(self):
        lazy_vars_t = list(lazy_var.t() for lazy_var in self.lazy_var)
        return SumLazyVariable(*lazy_vars_t)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        return sum(lazy_var._batch_get_indices(batch_indices, left_indices, right_indices)
                   for lazy_var in self.lazy_vars)

    def _get_indices(self, left_indices, right_indices):
        return sum(lazy_var._get_indices(left_indices, right_indices)
                   for lazy_var in self.lazy_vars)

    def add_jitter(self):
        lazy_vars = list(self.lazy_vars[:-1])
        lazy_vars.append(self.lazy_vars[-1].add_jitter())
        return SumLazyVariable(*lazy_vars)

    def __add__(self, other):
        if isinstance(other, SumLazyVariable):
            return SumLazyVariable(*(list(self.lazy_vars) + list(other.lazy_vars)))
        elif isinstance(other, LazyVariable):
            return SumLazyVariable(*(list(self.lazy_vars) + [other]))
        else:
            raise AttributeError('other must be a LazyVariable')

    def diag(self):
        return sum(lazy_var.diag() for lazy_var in self.lazy_vars)
