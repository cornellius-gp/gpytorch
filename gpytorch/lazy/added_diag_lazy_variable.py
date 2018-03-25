from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .sum_lazy_variable import SumLazyVariable
from .diag_lazy_variable import DiagLazyVariable


class AddedDiagLazyVariable(SumLazyVariable):
    """
    A SumLazyVariable, but of only two lazy variables, the second of which must be
    a DiagLazyVariable.
    """
    def __init__(self, *lazy_vars):
        lazy_vars = list(lazy_vars)
        super(AddedDiagLazyVariable, self).__init__(*lazy_vars)
        if len(lazy_vars) > 2:
            raise RuntimeError('An AddedDiagLazyVariable can only have two components')

        if isinstance(lazy_vars[0], DiagLazyVariable) and isinstance(lazy_vars[1], DiagLazyVariable):
            raise RuntimeError('Trying to lazily add two DiagLazyVariables. Create a single DiagLazyVariable instead.')
        elif isinstance(lazy_vars[0], DiagLazyVariable):
            self._diag_var = lazy_vars[0]
            self._lazy_var = lazy_vars[1]
        elif isinstance(lazy_vars[1], DiagLazyVariable):
            self._diag_var = lazy_vars[1]
            self._lazy_var = lazy_vars[0]
        else:
            raise RuntimeError('One of the LazyVariables input to AddedDiagLazyVariable must be a DiagLazyVariable!')
