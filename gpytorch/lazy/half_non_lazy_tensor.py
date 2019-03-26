import torch
from .non_lazy_tensor import NonLazyTensor


class HalfNonLazyTensor(NonLazyTensor):
    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return super()._get_indices(left_indices, right_indices, *batch_indices).float()

    def _getitem(self, *indices):
        return super()._getitem(*indices)

    def _matmul(self, rhs):
        return super()._matmul(rhs.half()).float()

    def _t_matmul(self, rhs):
        return super()._t_matmul(rhs.half()).float()

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return (res.float() for res in super()._quad_form_derivative(left_vecs, right_vecs))

    def diag(self):
        return super().diag().float()

    @property
    def dtype(self):
        return torch.float32

    def evaluate(self):
        return self.tensor.float()