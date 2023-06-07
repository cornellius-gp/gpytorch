from typing import Optional, Union

import torch
from linear_operator import LinearOperator
from torch import Tensor


class MaskedLinearOperator(LinearOperator):
    def __init__(self, base: LinearOperator, row_mask: Tensor, col_mask: Tensor):
        super().__init__(base, row_mask, col_mask)
        self.base = base
        self.row_mask = row_mask
        self.col_mask = col_mask
        self.row_eq_col_mask = row_mask is not None and col_mask is not None and torch.equal(row_mask, col_mask)

    def _matmul(self, rhs: Tensor) -> Tensor:
        if self.col_mask is not None:
            rhs_expanded = torch.zeros(
                *rhs.shape[:-2],
                self.base.size(-1),
                rhs.shape[-1],
                device=rhs.device,
                dtype=rhs.dtype,
            )
            rhs_expanded[..., self.col_mask, :] = rhs
            rhs = rhs_expanded

        res = self.base.matmul(rhs)

        if self.row_mask is not None:
            res = res[..., self.row_mask, :]

        return res

    def _size(self) -> torch.Size:
        base_size = list(self.base.size())
        if self.row_mask is not None:
            base_size[-2] = torch.count_nonzero(self.row_mask)
        if self.col_mask is not None:
            base_size[-1] = torch.count_nonzero(self.col_mask)
        return torch.Size(tuple(base_size))

    def _transpose_nonbatch(self) -> LinearOperator:
        return MaskedLinearOperator(self.base.mT, self.col_mask, self.row_mask)

    def _getitem(
        self,
        row_index: Union[slice, torch.LongTensor],
        col_index: Union[slice, torch.LongTensor],
        *batch_indices: tuple[Union[int, slice, torch.LongTensor], ...],
    ) -> LinearOperator:
        raise NotImplementedError("Indexing with %r, %r, %r not supported." % (batch_indices, row_index, col_index))

    def _get_indices(
        self,
        row_index: torch.LongTensor,
        col_index: torch.LongTensor,
        *batch_indices: tuple[torch.LongTensor, ...],
    ) -> torch.Tensor:
        def map_indices(index: torch.LongTensor, mask: Optional[Tensor], base_size: int) -> torch.LongTensor:
            if mask is None:
                return index
            map = torch.arange(base_size, device=self.base.device)[mask]
            return map[index]

        if len(batch_indices) == 0:
            row_index = map_indices(row_index, self.row_mask, self.base.size(-2))
            col_index = map_indices(col_index, self.col_mask, self.base.size(-1))
            return self.base._get_indices(row_index, col_index)

        raise NotImplementedError("Indexing with %r, %r, %r not supported." % (batch_indices, row_index, col_index))

    def _diagonal(self) -> Tensor:
        if not self.row_eq_col_mask:
            raise NotImplementedError()
        diag = self.base.diagonal()
        return diag[self.row_mask]

    def to_dense(self) -> torch.Tensor:
        full_dense = self.base.to_dense()
        return full_dense[..., self.row_mask, :][..., :, self.col_mask]

    def _cholesky_solve(self, rhs, upper: bool = False) -> LinearOperator:
        raise NotImplementedError()

    def _expand_batch(self, batch_shape: torch.Size) -> LinearOperator:
        raise NotImplementedError()

    def _isclose(self, other, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
        raise NotImplementedError()

    def _prod_batch(self, dim: int) -> LinearOperator:
        raise NotImplementedError()

    def _sum_batch(self, dim: int) -> LinearOperator:
        raise NotImplementedError()
