from torch.autograd import Variable
from .lazy_variable import LazyVariable


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
            if tensor.ndimension() == 1 and self.ndimension() == 2:
                return diag * tensor
            else:
                res = diag.unsqueeze(-1).expand_as(tensor) * tensor
                return res

        return closure

    def _t_matmul_closure_factory(self, diag):
        return self._matmul_closure_factory(diag)

    def _derivative_quadratic_form_factory(self, diag):
        def closure(left_factor, right_factor):
            res = left_factor * right_factor
            if res.ndimension() > diag.ndimension():
                res = res.sum(-2)
            return res,

        return closure

    def _size(self):
        if self._diag.ndimension() == 2:
            return self._diag.size(0), self._diag.size(-1), self._diag.size(-1)
        else:
            return self._diag.size(-1), self._diag.size(-1)

    def _transpose_nonbatch(self):
        return self

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        equal_indices = left_indices.eq(right_indices).type_as(self._diag.data)
        return self._diag[batch_indices, left_indices] * equal_indices

    def _get_indices(self, left_indices, right_indices):
        equal_indices = left_indices.eq(right_indices).type_as(self._diag.data)
        return self._diag[left_indices] * equal_indices

    def add_diag(self, added_diag):
        return DiagLazyVariable(self._diag + added_diag.expand_as(self._diag))

    def diag(self):
        return self._diag

    def evaluate(self):
        if self.ndimension() == 2:
            return self._diag.diag()
        else:
            return super(DiagLazyVariable, self).evaluate()

    def zero_mean_mvn_samples(self, n_samples):
        if self.ndimension() == 3:
            base_samples = Variable(self.tensor_cls(self._diag.size(0), self._diag.size(1), n_samples).normal_())
        else:
            base_samples = Variable(self.tensor_cls(self._diag.size(0), n_samples).normal_())
        samples = self._diag.unsqueeze(-1) * base_samples
        return samples
