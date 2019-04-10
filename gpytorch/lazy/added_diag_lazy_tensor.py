#!/usr/bin/env python3

import torch
import warnings
from .sum_lazy_tensor import SumLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .psd_sum_lazy_tensor import PsdSumLazyTensor
from .root_lazy_tensor import RootLazyTensor

from ..utils import broadcasting, pivoted_cholesky, woodbury
from .. import settings


class AddedDiagLazyTensor(SumLazyTensor):
    """
    A SumLazyTensor, but of only two lazy tensors, the second of which must be
    a DiagLazyTensor.
    """

    def __init__(self, *lazy_tensors):
        lazy_tensors = list(lazy_tensors)
        super(AddedDiagLazyTensor, self).__init__(*lazy_tensors)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")

        broadcasting._mul_broadcast_shape(lazy_tensors[0].shape, lazy_tensors[1].shape)

        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. " "Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

    def _matmul(self, rhs):
        return torch.addcmul(self._lazy_tensor._matmul(rhs), self._diag_tensor._diag.unsqueeze(-1), rhs)

    def add_diag(self, added_diag):
        return AddedDiagLazyTensor(self._lazy_tensor, self._diag_tensor.add_diag(added_diag))

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor

        if isinstance(other, DiagLazyTensor):
            return AddedDiagLazyTensor(self._lazy_tensor, self._diag_tensor + other)
        else:
            return AddedDiagLazyTensor(self._lazy_tensor + other, self._diag_tensor)

    def _preconditioner(self):
        if settings.max_preconditioner_size.value() == 0:
            return None, None

        if not (hasattr(self, "_woodbury_cache") or hasattr(self, "self._q_cache")):
            max_iter = settings.max_preconditioner_size.value()
            self._piv_chol_self = pivoted_cholesky.pivoted_cholesky(self._lazy_tensor, max_iter)
            if torch.any(torch.isnan(self._piv_chol_self)).item():
                warnings.warn(
                    "NaNs encountered in preconditioner computation. Attempting to continue without preconditioning."
                )
                return None, None

            if self._piv_chol_self.dim() == 2:  # TODO: Whenever PyTorch supports batch mode
                *batch_shape, n, k = self._piv_chol_self.shape
                self._noise = self._diag_tensor.diag().unsqueeze(-1)

                constant_diag = torch.equal(self._noise, self._noise[0] * torch.ones_like(self._noise))

                eye = torch.eye(k, dtype=self._piv_chol_self.dtype, device=self._piv_chol_self.device)

                if constant_diag:
                    # We can factor out the noise for for both QR and solves.
                    noise_constant = self._noise[0].squeeze()
                    self._q_cache, self._r_cache = torch.qr(
                        torch.cat((self._piv_chol_self, noise_constant.sqrt() * eye))
                    )
                    self._q_cache = self._q_cache[:n, :]

                    # Use the matrix determinant lemma for the logdet, using the fact that R'R = L_k'L_k + s*I
                    logdet = self._r_cache.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1).mul(2)
                    logdet = logdet + (n - k) * noise_constant.log()
                    self._precond_logdet_cache = logdet.view(*batch_shape) if len(batch_shape) else logdet.squeeze()

                    def precondition_closure(rhs: torch.Tensor):
                        return (1 / noise_constant) * (rhs - self._q_cache.matmul(self._q_cache.t().matmul(rhs)))

                else:
                    # With non-constant diagonals, we cant factor out the noise as easily
                    self._q_cache, self._r_cache = torch.qr(torch.cat((self._piv_chol_self / self._noise.sqrt(), eye)))
                    self._q_cache = self._q_cache[:n, :] / self._noise.sqrt()

                    logdet = self._r_cache.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1).mul(2) - (
                        1.0 / self._noise
                    ).log().sum([-1, -2])
                    self._precond_logdet_cache = logdet.view(*batch_shape) if len(batch_shape) else logdet.squeeze()

                    def precondition_closure(rhs: torch.Tensor):
                        return (rhs / self._noise) - self._q_cache.matmul(self._q_cache.t().matmul(rhs))

            else:
                self._woodbury_cache, self._inv_scale, self._precond_logdet_cache = woodbury.woodbury_factor(
                    self._piv_chol_self, self._piv_chol_self, self._diag_tensor.diag(), logdet=True
                )
                self._scaled_inv_diag = self._inv_scale / self._diag_tensor.diag().unsqueeze(-1)
                self._scaled_inv_diag_piv_chol_self = self._piv_chol_self * self._scaled_inv_diag

                # preconditioner
                def precondition_closure(tensor):
                    res = woodbury.woodbury_solve(
                        tensor,
                        self._scaled_inv_diag_piv_chol_self,
                        self._woodbury_cache,
                        self._scaled_inv_diag,
                        self._inv_scale,
                    )
                    return res

            self.preconditioner_lt = PsdSumLazyTensor(RootLazyTensor(self._piv_chol_self), self._diag_tensor)

            self.precondition_closure = precondition_closure

        return self.precondition_closure, self.preconditioner_lt, self._precond_logdet_cache
