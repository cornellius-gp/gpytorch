import torch
from .lazy_tensor import LazyTensor
from ..utils.memoize import cached
from ..utils.getitem import _noop_index


class KeOpsLazyTensor(LazyTensor):
    def __init__(self, x1, x2, covar_func, **params):
        super().__init__(x1, x2, covar_func=covar_func, **params)

        self.x1 = x1
        self.x2 = x2
        self.covar_func = covar_func
        self.params = params

    @cached(name="kernel_diag")
    def diag(self):
        """
        Getting the diagonal of a kernel can be handled more efficiently by
        transposing the batch and data dimension before calling the kernel.
        Implementing it this way allows us to compute predictions more efficiently
        in cases where only the variances are required.
        """
        return self.covar_func(self.x1, self.x2, diag=True)

    @property
    @cached(name="covar_mat")
    def covar_mat(self):
        return self.covar_func(self.x1, self.x2, **self.params)

    def _matmul(self, rhs):
        # TODO: replace with `self.covar_mat @ rhs` on next PyKeOps release.
        batch_shape = self.batch_shape

        # Equivalent to dim=-3, but KeOps LT doesn't support relative index in sum.
        return (self.covar_mat * rhs[..., None, :, :]).sum(dim=len(batch_shape) + 1)

    def _size(self):
        return torch.Size(self.covar_mat.shape)

    def _transpose_nonbatch(self):
        return KeOpsLazyTensor(self.x2, self.x1, self.covar_func)

    def _get_indices(self, row_index, col_index, *batch_indices):
        x1_ = self.x1[(*batch_indices, row_index)]
        x2_ = self.x2[(*batch_indices, col_index)]
        return self.covar_func(x1_, x2_, diag=True, **self.params)

    def _getitem(self, row_index, col_index, *batch_indices):
        x1 = self.x1
        x2 = self.x2
        dim_index = _noop_index

        # Get the indices of x1 and x2 that matter for the kernel
        # Call x1[*batch_indices, row_index, :]
        try:
            x1 = x1[(*batch_indices, row_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x1 = x1.expand(1, *self.x1.shape[-2:])[(*batch_indices, row_index, dim_index)]
            elif isinstance(batch_indices, tuple):
                if any(not isinstance(bi, slice) for bi in batch_indices):
                    raise RuntimeError(
                        f"Attempting to tensor index a non-batch matrix's batch dimensions. "
                        "Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x1 = x1.expand(*([1] * len(batch_indices)), *self.x1.shape[-2:])
                x1 = x1[(*batch_indices, row_index, dim_index)]

        # Call x2[*batch_indices, row_index, :]
        try:
            x2 = x2[(*batch_indices, col_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x2 = x2.expand(1, *self.x2.shape[-2:])[(*batch_indices, row_index, dim_index)]
            elif isinstance(batch_indices, tuple):
                if any([not isinstance(bi, slice) for bi in batch_indices]):
                    raise RuntimeError(
                        f"Attempting to tensor index a non-batch matrix's batch dimensions. "
                        "Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x2 = x2.expand(*([1] * len(batch_indices)), *self.x2.shape[-2:])
                x2 = x2[(*batch_indices, row_index, dim_index)]

        # Now construct a kernel with those indices
        return self.__class__(x1, x2, covar_func=self.covar_func, **self.params)
