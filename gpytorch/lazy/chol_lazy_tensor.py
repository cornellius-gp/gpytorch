#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class CholLazyTensor(RootLazyTensor):
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

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term, _ = super(CholLazyTensor, self).inv_quad_logdet(
                inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )

        if logdet:
            logdet_term = self._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term
