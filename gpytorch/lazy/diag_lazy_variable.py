from gpytorch.lazy import LazyVariable
from .non_lazy_variable import NonLazyVariable


class DiagLazyVariable(LazyVariable):
    def __init__(self, diag):
        """
        Diagonal lazy variable

        Args:
        - diag (Variable: n) diagonal of matrix
        """
        super(DiagLazyVariable, self).__init__(diag)
        self._diag = diag

    def _matmul_closure_factory(self, diag):
        def closure(tensor):
            if tensor.ndimension() == 1:
                return diag * tensor
            else:
                res = diag.unsqueeze(1).expand_as(tensor) * tensor
                return res

        return closure

    def _derivative_quadratic_form_factory(self, diag):
        def closure(left_factor, right_factor):
            res = left_factor * right_factor
            if res.ndimension() == 2:
                res = res.sum(0)
            return res,

        return closure

    def add_diag(self, added_diag):
        return DiagLazyVariable(self._diag + added_diag.expand_as(self._diag))

    def diag(self):
        return self._diag

    def evaluate(self):
        return self._diag.diag()

    def representation(self):
        return self._diag,

    def __getitem__(self, index):
        return NonLazyVariable(self.evaluate())[index]

    def size(self):
        return self._diag.size(0), self._diag.size(0)
