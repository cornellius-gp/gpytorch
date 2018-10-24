from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_tensor import LazyTensor


class DiagLazyTensor(LazyTensor):
    def __init__(self, diag, exact_root_decomposition=False):
        """
        Diagonal lazy tensor

        Args:
            - diag (tensor) (batch) diagonal of matrix
            - exact_root_decomposition (bool, default False) return an exact root decomposition.
                Set to True only if the represented diagonal is relatively small.
        """
        super(DiagLazyTensor, self).__init__(diag)
        self._diag = diag
        self.exact_root_decomposition = exact_root_decomposition

    def _matmul(self, rhs):
        if rhs.ndimension() == 1 and self.ndimension() == 2:
            return self._diag * rhs
        else:
            res = self._diag.unsqueeze(-1).expand_as(rhs) * rhs
            return res

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs * right_vecs
        if res.ndimension() > self._diag.ndimension():
            res = res.sum(-1)
        return (res,)

    def _size(self):
        if self._diag.ndimension() == 2:
            return self._diag.size(0), self._diag.size(-1), self._diag.size(-1)
        else:
            return self._diag.size(-1), self._diag.size(-1)

    def _transpose_nonbatch(self):
        return self

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        equal_indices = left_indices.eq(right_indices).type_as(self._diag)
        return self._diag[batch_indices, left_indices] * equal_indices

    def _get_indices(self, left_indices, right_indices):
        equal_indices = left_indices.eq(right_indices).type_as(self._diag)
        return self._diag[left_indices] * equal_indices

    def add_diag(self, added_diag):
        return DiagLazyTensor(self._diag + added_diag.expand_as(self._diag))

    def __add__(self, other):
        from .added_diag_lazy_tensor import AddedDiagLazyTensor
        if isinstance(other, DiagLazyTensor):
            return DiagLazyTensor(self._diag + other._diag)
        else:
            return AddedDiagLazyTensor(other, self)

    def diag(self):
        return self._diag

    def evaluate(self):
        if self.ndimension() == 2:
            return self._diag.diag()
        else:
            return super(DiagLazyTensor, self).evaluate()

    def sum_batch(self, sum_batch_size=None):
        if sum_batch_size is None:
            diag = self._diag.view(-1, self._diag.size(-1))
        else:
            diag = self._diag.view(-1, sum_batch_size, self._diag.size(-1))

        return self.__class__(diag.sum(-2))

    def inv_matmul(self, tensor):
        is_vec = False
        if (self.dim() == 2 and tensor.dim() == 1):
            tensor = tensor.unsqueeze(-1)
            is_vec = True

            if self.shape[-1] != tensor.numel():
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, tensor.shape
                    )
                )
        elif self.dim() != tensor.dim():
            raise RuntimeError(
                "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                "of dimensions.".format(self.shape, tensor.shape)
            )
        elif self.batch_shape != tensor.shape[:-2] or self.shape[-1] != tensor.shape[-2]:
            raise RuntimeError(
                "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                    self.shape, tensor.shape
                )
            )

        res = tensor.div(self._diag.unsqueeze(-1))
        if is_vec:
            res = res.squeeze(-1)
        return res

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            if (self.dim() == 2 and inv_quad_rhs.dim() == 1):
                inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.batch_shape != inv_quad_rhs.shape[:-2] or self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term = None
        if inv_quad_rhs is None:
            inv_quad_term = torch.tensor([], dtype=self.dtype, device=self.device)
        else:
            inv_quad_term = inv_quad_rhs.div(self._diag.unsqueeze(-1)).mul(inv_quad_rhs).sum(-2)
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        log_det_term = None
        if log_det:
            log_det_term = self._diag.log().sum(-1)
        else:
            log_det_term = torch.tensor([], dtype=self.dtype, device=self.device)

        return inv_quad_term, log_det_term

    def root_decomposition(self):
        if self.exact_root_decomposition:
            return self.__class__(self._diag.sqrt()).evaluate()
        else:
            return super(DiagLazyTensor, self).root_decomposition()

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        if self.exact_root_decomposition:
            return self.__class__(self._diag.sqrt().reciprocal()).evaluate()
        else:
            return super(DiagLazyTensor, self).root_decomposition()

    def zero_mean_mvn_samples(self, num_samples):
        if self.ndimension() == 3:
            base_samples = torch.randn(
                num_samples, self._diag.size(0), self._diag.size(1), dtype=self.dtype, device=self.device
            )
        else:
            base_samples = torch.randn(num_samples, self._diag.size(0), dtype=self.dtype, device=self.device)
        samples = self._diag.unsqueeze(0).sqrt() * base_samples
        return samples
