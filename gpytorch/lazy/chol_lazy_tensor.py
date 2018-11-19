#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class CholLazyTensor(RootLazyTensor):
    def __init__(self, chol):
        if isinstance(chol, LazyTensor):  # Probably is an instance of NonLazyTensor
            chol = chol.evaluate()

        # Check that we have a lower triangular matrix
        mask = torch.full((chol.size(-2), chol.size(-2)), fill_value=-1, dtype=chol.dtype, device=chol.device)
        mask.tril_().add_(1)
        if chol.ndimension() == 3:
            mask.unsqueeze_(0)
        if torch.max(chol.mul(mask)).item() > 1e-3 and torch.equal(chol, chol):
            raise RuntimeError("CholLazyVaraiable should take a lower-triangular " "matrix in the constructor.")

        # Run super constructor
        super(CholLazyTensor, self).__init__(chol)

        # Check that the diagonal is
        if not torch.equal(self._chol_diag.abs(), self._chol_diag):
            raise RuntimeError("The diagonal of the cholesky decomposition should be positive.")

    @property
    def _chol(self):
        if not hasattr(self, "_chol_memo"):
            self._chol_memo = self.root.evaluate()
        return self._chol_memo

    @property
    def _chol_diag(self):
        if not hasattr(self, "_chol_diag_memo"):
            if self._chol.ndimension() == 3:
                batch_size, diag_size, _ = self._chol.size()
                batch_index = torch.arange(0, batch_size, dtype=torch.long, device=self.device)
                batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
                diag_index = torch.arange(0, diag_size, dtype=torch.long, device=self.device)
                diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
                self._chol_diag_memo = self._chol[batch_index, diag_index, diag_index].view(batch_size, diag_size)
            else:
                self._chol_diag_memo = self._chol.diag()
        return self._chol_diag_memo

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
        inv_quad_term = None
        log_det_term = None

        if inv_quad_rhs is not None:
            inv_quad_term, _ = super(CholLazyTensor, self).inv_quad_log_det(
                inv_quad_rhs, log_det=False, reduce_inv_quad=reduce_inv_quad
            )

        if log_det:
            log_det_term = self._chol_diag.log().sum(-1).mul(2)

        return inv_quad_term, log_det_term
