import torch
from .non_lazy_tensor import NonLazyTensor
from .. import settings
import mixed


class HalfNonLazyTensor(NonLazyTensor):
    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return super()._get_indices(left_indices, right_indices, *batch_indices).float()

    def _getitem(self, *indices):
        return super()._getitem(*indices)

    def _matmul(self, rhs):
        if settings.use_fp16_mult.on():
            # Note we do not support broadcasting like torch.matmul yet
            if self.batch_dim > 0:
                matrix_shape = self.matrix_shape
                tsr = self.tensor.view(-1, *matrix_shape)
                rhs = rhs.view(-1, *rhs.shape[-2:])
                res = mixed.bmm(tsr, rhs, None).view(*self.batch_shape, *matrix_shape)
            else:
                res = mixed.mm(self.tensor, rhs, None)
        else:
            res = super()._matmul(rhs.half()).float()
        return res

    def _t_matmul(self, rhs):
        if settings.use_fp16_mult.on():
            # Note we do not support broadcasting like torch.matmul yet
            if self.batch_dim > 0:
                matrix_shape = self.matrix_shape
                tsr = self.tensor.view(-1, *matrix_shape).transpose(-1, -2)
                rhs = rhs.view(-1, *rhs.shape[-2:])
                res = mixed.bmm(tsr, rhs, None).view(*self.batch_shape, *matrix_shape)
            else:
                res = mixed.mm(self.tensor.transpose(-1, -2), rhs, None)
        else:
            res = self._t_matmul(rhs.half()).float()
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return tuple(res.half() for res in super()._quad_form_derivative(left_vecs, right_vecs))

    def diag(self):
        return super().diag().float()

    @property
    def dtype(self):
        return torch.float32

    def evaluate(self):
        return self.tensor.float()
