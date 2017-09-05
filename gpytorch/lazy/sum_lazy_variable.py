from .lazy_variable import LazyVariable
from ..posterior import DefaultPosteriorStrategy


class SumLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars):
        if not all([isinstance(lazy_var, LazyVariable) for lazy_var in lazy_vars]):
            raise RuntimeError('All arguments of a SumLazyVariable should be lazy variables')

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

    def add_diag(self, diag):
        lazy_vars = list(self.lazy_vars[:-1])
        lazy_vars.append(self.lazy_vars[-1].add_diag(diag))
        return SumLazyVariable(*lazy_vars)

    def add_jitter(self):
        lazy_vars = list(self.lazy_vars[:-1])
        lazy_vars.append(self.lazy_vars[-1].add_jitter())
        return SumLazyVariable(*lazy_vars)

    def evaluate(self):
        return sum(lazy_var.evaluate() for lazy_var in self.lazy_vars)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        raise NotImplementedError

    def mul(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise RuntimeError('Can only multiply by scalars')
        return SumLazyVariable(*(lazy_var * other for lazy_var in self.lazy_vars))

    def representation(self):
        res = tuple(var for lazy_var in self.lazy_vars for var in lazy_var.representation())
        return res

    def posterior_strategy(self):
        return DefaultPosteriorStrategy(self)

    def __add__(self, other):
        if isinstance(other, SumLazyVariable):
            return SumLazyVariable(*(list(self.lazy_vars) + list(other.lazy_vars)))
        elif isinstance(other, LazyVariable):
            return SumLazyVariable(*(list(self.lazy_vars) + [other]))
        else:
            raise AttributeError('other must be a LazyVariable')

    def __getitem__(self, i):
        sliced_lazy_vars = [lazy_var.__getitem__(i) for lazy_var in self.lazy_vars]
        return SumLazyVariable(*sliced_lazy_vars)
