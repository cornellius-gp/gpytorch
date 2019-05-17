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
            m, n = self.tensor.shape[-2:]
            if (m * n) % 8 != 0:
                print(f"HalfNonLazyTensor has last two dimensions size {(m, n)} "
                       "which are not multiples of 8 to use tensor cores")
            # pad to use tensor cores assuming self.tensor is already correctly sized
            n_vecs = rhs.size(-1)
            pad = (0, 8 - (n_vecs % 8))
            rhs = torch.nn.functional.pad(rhs, pad)
            # Note we do not support broadcasting like torch.matmul yet
            if settings.use_fp32_acc.on():
                if self.batch_dim > 0:
                    matrix_shape = self.matrix_shape
                    tsr = self.tensor.view(-1, *matrix_shape)
                    rhs = rhs.view(-1, *rhs.shape[-2:])
                    res = mixed.bmm(tsr, rhs, None).view(*self.batch_shape, *matrix_shape)
                else:
                    res = mixed.mm(self.tensor, rhs, None)
            else:
                res = super()._matmul(rhs.half()).float()
            res = res[..., :n_vecs]
        else:
            res = super()._matmul(rhs.half()).float()
        return res

    def _t_matmul(self, rhs):
        if settings.use_fp16_mult.on():
            m, n = self.tensor.shape[-2:]
            if (m * n) % 8 != 0:
                print(f"HalfNonLazyTensor has last two dimensions size {(m, n)} "
                       "which are not multiples of 8 to use tensor cores")
            # pad to use tensor cores assuming self.tensor is already correctly sized
            n_vecs = rhs.size(-1)
            pad = (0, 8 - (n_vecs % 8))
            rhs = torch.nn.functional.pad(rhs, pad)
            # Note we do not support broadcasting like torch.matmul yet
            if settings.use_fp32_acc.on():
                if self.batch_dim > 0:
                    matrix_shape = self.matrix_shape
                    tsr = self.tensor.view(-1, *matrix_shape).transpose(-1, -2)
                    rhs = rhs.view(-1, *rhs.shape[-2:])
                    res = mixed.bmm(tsr, rhs, None).view(*self.batch_shape, *matrix_shape)
                else:
                    res = mixed.mm(self.tensor.transpose(-1, -2), rhs, None)
            else:
                res = super()._t_matmul(rhs.half()).float()
            res = res[..., :n_vecs]
        else:
            res = super()._t_matmul(rhs.half()).float()
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
