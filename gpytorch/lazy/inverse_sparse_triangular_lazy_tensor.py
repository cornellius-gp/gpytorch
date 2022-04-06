import torch
from .lazy_tensor import LazyTensor

from gpytorch.utils.memoize import cached


class InverseSparseTriangularLazyTensor(LazyTensor):
    def __init__(self, indices, values, upper=False):
        """
        The first column of values has to be the diagonal entries.
        """
        super().__init__(indices, values, upper=upper)

        self.indices = indices
        self.values = values

        self.upper = upper

        from gpytorch.utils.sparse import make_sparse_from_indices_and_values
        self.coo_tensor = make_sparse_from_indices_and_values(
            indices,
            values,
            indices.size(-2)
        )

        """
        make_sparse_from_indices_and_values creates the transpose of the sparse tensor
        """
        if not upper:
            self.coo_tensor = self.coo_tensor.t()

        """
        Preprocessing self.coo_tensor using coalesce() speeds up sparse matmul.
        """
        self.coo_tensor = self.coo_tensor.coalesce()

    def _size(self):
        return self.indices.size(-2), self.indices.size(-2)

    def _transpose_nonbatch(self):
        return InverseSparseTriangularLazyTensor(self.indices, self.values, upper=not self.upper)

    def inv_matmul(self, rhs):
        """
        TODO:
        - implement the batch mode
        """
        return self.coo_tensor.matmul(rhs)

    def diag(self):
        return self.values[:, 0].reciprocal()

    @property
    def dtype(self):
        return self.values.dtype

    def logdet(self):
        return self.diag().abs().log().sum()

    def _matmul(self, rhs):
        from gpytorch.cusparse import sparse_triangular_solve
        return sparse_triangular_solve(self.coo_tensor, rhs, self.upper)

    def _matmul_python(self, rhs):
        """
        The forward and backward substitution are implemented in Python and might be slow.
        Assume that rhs is unsqueezed if it is a vector.
        """
        n = self._size()[-2]

        ret = torch.zeros_like(rhs)

        if not self.upper:
            for i in range(n):
                """
                Writing self.values[i] is the same as self.values[i, 1:].
                Similarly, writing self.indices[i] is the same as self.indices[i, 1:].
                Because ret[self.indices[i, 0]] = 0 before the update and
                it does not contribute to the inner product.
                """
                ret[i] = (rhs[i] - self.values[i].unsqueeze(-2).matmul(ret[self.indices[i]])) / self.values[i, 0]
        else:
            for i in range(n - 1, -1, -1):
                col_idx, row_idx = torch.nonzero(self.indices.eq(i), as_tuple=True)
                inner_product = self.values[col_idx, row_idx].unsqueeze(-2).matmul(ret[col_idx])
                ret[i] = (rhs[i] - inner_product) / self.values[i, 0]

        return ret


if __name__ == "__main__":
    pass
