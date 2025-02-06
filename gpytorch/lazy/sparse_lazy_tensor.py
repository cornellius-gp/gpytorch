import copy

import torch
from torch import Tensor

from ..utils.getitem import _is_noop_index, _noop_index
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor


class SparseLazyTensor(LazyTensor):
    def _check_args(self, indices, values, sparse_size):

        if indices.shape[:-2] != values.shape[:-1]:
            return (
                "indices size ({}) is incompatible with values size ({}). Make sure the two have the "
                "same number of batch dimensions".format(indices.size(), values.size())
            )

        if indices.size()[-1] != values.size()[-1]:
            return "Expected number of indices ({}) to have the same size as values ({})".format(
                indices.size()[-1], values.size()[-1]
            )

        if indices.size()[-2] != len(sparse_size):
            return "Expected number dimenions ({}) to have the same as length of size ({})".format(
                indices.size()[-2], len(sparse_size)
            )

    def __init__(self, indices: Tensor, values: Tensor, sparse_size: Tensor):
        """
        Sparse Lazy Tensor. Lazify torch.sparse_coo_tensor and supports arbitrary batch sizes.
        Args:
            :param indices: `b1 x ... x bk x ndim x nse` `tensor` containing indices of a `b1 x ... x bk`-sized batch
                    of sparse matrices with `sparse_size`.
            :param values: `b1 x ... x bk x nse` `tensor` containing values of a `b1 x ... x bk`-sized batch
                    of sparse matrices with `sparse_size`.
            :param sparse_size: `tensor` containing shape of non-batched dimensions of sparse matrices.

        TODO: revisit this as it seems to me that ndim=2 is sufficient for most cases.
        """
        super().__init__(indices, values, sparse_size)

        # Local variable to keep batch shape as batch dimensions are squeezed in _tensor for efficiency.
        self._batch_shape = indices.shape[:-2]

        num_batches, ndim, nse = self._batch_shape.numel(), indices.shape[-2], indices.shape[-1]

        self.ndim = ndim  # dimension of the sparse matrices
        self.nse = nse  # number of specified elements
        self.sparse_size = sparse_size

        if num_batches > 1:
            indices = indices.reshape(num_batches, ndim, nse)
            values = values.reshape(num_batches, nse)
            tensor_size = (num_batches, *sparse_size.to(torch.int64).numpy())
            tensor = self.setup_3dtensor(indices=indices, values=values, tensor_size=tensor_size)

        else:
            tensor = torch.sparse_coo_tensor(
                indices=indices, values=values, size=tuple(sparse_size.to(torch.int64).numpy()), device=indices.device
            )

        self._tensor = tensor.coalesce()

    def setup_3dtensor(self, indices, values, tensor_size):

        batch_indices = torch.hstack([torch.ones(self.nse) * i for i in torch.arange(indices.shape[0])])
        indices = torch.vstack([batch_indices, torch.hstack(list(indices))])

        values = values.reshape(-1)

        return torch.sparse_coo_tensor(
            indices=indices, values=values, size=tensor_size, device=indices.device, requires_grad=False
        )

    def to_dense(self):
        return self._tensor.to_dense().reshape(*self.size())

    def _size(self):
        return torch.Size(self._batch_shape + self._tensor.shape[-2:])

    def compute_effective_batch_index(self, *batch_indices):
        shifted_shapes = (*self.batch_shape[:-1], 1)[1:]
        return sum(bs * bi for bs, bi in zip(shifted_shapes, batch_indices))

    def _transpose_nonbatch(self):
        # TODO: this is implemented assuming ndim is 2.
        tensor_indices = self._tensor._indices().clone()
        new_indices = torch.zeros_like(tensor_indices)
        new_indices[..., 0, :] = new_indices[..., 1, :]
        new_indices[..., 1, :] = new_indices[..., 0, :]
        return self.__class__(indices=new_indices, values=self._tensor._values(), sparse_size=self.sparse_size)

    @cached
    def evaluate(self):
        return self._tensor.to_dense().reshape(self.shape)

    def _matmul(self, rhs: Tensor) -> Tensor:
        # TODO: test for rhs with both 2-D and 3-D shapes, i.e, * X * and b X * X * .
        # Most likely, I'd need some usage of _mul_broadcast_shape.
        if self.ndimension() == 3:
            return torch.bmm(self._tensor, rhs).reshape(*self.shape[:-1], -1)
        else:
            return torch.sparse.mm(self._tensor, rhs)

    def matmul(self, tensor):
        return self._matmul(rhs=tensor)

    def _mul_constant(self, constant):

        if self.ndimension() > 2:
            ndim, nse = self._tensor.indices().shape[-2:]
            return self.__class__(
                indices=self._tensor._indices().reshape(*self.batch_shape, ndim, nse),
                values=constant * self._tensor._values.reshape(*self.batch_shape, nse),
                sparse_size=self.sparse_size,
            )
        else:
            return self.__class__(
                indices=self._tensor._indices(), values=constant * self._tensor._values(), sparse_size=self.sparse_size,
            )

    def _t_matmul(self, rhs):
        return self._transpose_nonbatch().matmul(rhs)

    def _expand_batch(self, batch_shape):

        if not self._tensor.is_coalesced():
            self._tensor = self._tensor.coalesce()

        indices = self._tensor.indices().unsqueeze(0).expand(*batch_shape, self.ndim, self.nse)
        values = self._tensor.values().unsqueeze(0).expand(*batch_shape, self.nse)

        return self.__class__(indices=indices, values=values, sparse_size=self.sparse_size,)

    def _getitem(self, row_index, col_index, *batch_indices):
        if len(self.batch_shape) > 0:
            effective_batch_index = self.compute_effective_batch_index(batch_indices)
            return self._tensor[(effective_batch_index, row_index, col_index)]
        else:
            print("tensor: ", self._tensor, type(row_index), col_index)
            print(
                "done --> ",
                row_index is _noop_index,
                row_index is _noop_index,
                _is_noop_index(row_index),
                _is_noop_index(col_index),
            )
            return self._tensor[row_index, col_index]

    # def _get_indices(self, row_index, col_index, *batch_indices):
    #     if len(self.batch_shape) > 0:
    #         effective_batch_index = self.compute_effective_batch_index(batch_indices)
    #         return self._tensor[(effective_batch_index, row_index, col_index)]
    #     else:
    #         print("tensor: ", self._tensor, self._tensor[0, 1], row_index, col_index)
    #         return self._tensor[row_index, col_index]

    def _unsqueeze_batch(self, dim):
        new_batch_shape = torch.Size((*self._batch_shape[:dim], 1, *self._batch_shape[dim:]))
        return self.__class__(
            indices=self._tensor.indices().reshape(*new_batch_shape, self.ndim, self.nse),
            values=self._tensor.values().reshape(*new_batch_shape, self.nse),
            sparse_size=self.sparse_size,
        )

    def __add__(self, other):
        if isinstance(other, SparseLazyTensor):
            new_sparse_lazy_tensor = copy.deepcopy(self)
            new_sparse_lazy_tensor._tensor += other._tensor
            return new_sparse_lazy_tensor
        return super(SparseLazyTensor, self).__add__(other)

    def _sum_batch(self, dim):

        indices = self._tensor.indices().reshape(self.batch_shape, self.ndim, self.nse)
        values = self._tensor.values().reshape(self.batch_shape, self.nse)

        indices_splits = torch.split(indices, indices.shape[dim], dim)
        values_splits = torch.split(values, indices.shape[dim], dim)

        return sum(
            [
                self.__class__(indices=indices_split, values=values_split, sparse_size=self.sparse_size)
                for indices_split, values_split in zip(indices_splits, values_splits)
            ]
        )

    def _permute_batch(self, *dims):
        indices = self._tensor.indices().reshape(self.batch_shape, self.ndim, self.nse)
        values = self._tensor.values().reshape(self.batch_shape, self.nse)
        indices = indices.permute(*dims, -2, -1)
        values = values.permute(*dims, -1)
        return self.__class__(indices=indices, values=values, sparse_size=self.sparse_size)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # TODO: keep this as a reminder to revisit
        return super()._quad_form_derivative(left_vecs=left_vecs, right_vecs=right_vecs)

    def _cholesky_solve(self, rhs, upper: bool = False):
        raise NotImplementedError
