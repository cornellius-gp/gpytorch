import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .root_lazy_variable import RootLazyVariable


class CholLazyVariable(RootLazyVariable):

    def __init__(self, chol):
        if isinstance(chol, LazyVariable):  # Probably is an instance of NonLazyVariable
            chol = chol.evaluate()

        # Check that we have a lower triangular matrix
        mask = Variable(
            chol.data.new(chol.size(-2), chol.size(-2)).fill_(-1).tril_().add_(1)
        )
        if chol.ndimension() == 3:
            mask.data.unsqueeze_(0)
        if torch.max(chol.mul(mask)).item() > 1e-3 and torch.equal(chol, chol):
            raise RuntimeError(
                "CholLazyVaraiable should take a lower-triangular "
                "matrix in the constructor."
            )

        # Run super constructor
        super(CholLazyVariable, self).__init__(chol)

        # Check that the diagonal is
        if not torch.equal(self._chol_diag.abs(), self._chol_diag):
            raise RuntimeError(
                "The diagonal of the cholesky decomposition should be positive."
            )

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
                batch_index = self._chol.data.new(batch_size).long()
                torch.arange(0, batch_size, out=batch_index)
                batch_index = (batch_index.unsqueeze(1).repeat(1, diag_size).view(-1))
                diag_index = self._chol.data.new(diag_size).long()
                torch.arange(0, diag_size, out=diag_index)
                diag_index = (diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1))
                self._chol_diag_memo = self._chol[
                    batch_index, diag_index, diag_index
                ].view(
                    batch_size, diag_size
                )
            else:
                self._chol_diag_memo = self._chol.diag()
        return self._chol_diag_memo

    def inv_matmul(self, rhs):
        if self.ndimension() == 2:
            res = torch.potrs(rhs, self._chol, upper=False)
        else:
            res = super(CholLazyVariable, self).inv_matmul(rhs)
        return res

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False):
        inv_quad_term = None
        log_det_term = None
        is_batch = (self.ndimension() == 3)

        if inv_quad_rhs is not None:
            inv_quad_term = (
                self.inv_matmul(inv_quad_rhs).mul(inv_quad_rhs).sum(-1).sum(
                    -1, keepdim=(not is_batch)
                )
            )

        if log_det:
            log_det_term = (self._chol_diag.log().sum(-1).mul(2))

        return inv_quad_term, log_det_term
