#!/usr/bin/env python3

from .root_lazy_tensor import RootLazyTensor


class LowRankRootLazyTensor(RootLazyTensor):
    """
    Very thin wrapper around RootLazyTensor that denotes that the tensor specifically represents a low rank
    decomposition of a full rank matrix.

    The rationale for this class existing is that we can create LowRankAddedDiagLazyTensor without having to
    write custom _getitem, _get_indices, etc, leading to much better code reuse.
    """

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Tensor)
        """
        from .diag_lazy_tensor import ConstantDiagLazyTensor, DiagLazyTensor
        from .low_rank_root_added_diag_lazy_tensor import LowRankRootAddedDiagLazyTensor

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        diag_shape = diag.shape
        if len(diag_shape) == 0:
            # interpret scalar tensor as constant diag
            diag_tensor = ConstantDiagLazyTensor(diag.unsqueeze(-1), diag_shape=self.shape[-1])
        elif diag_shape[-1] == 1:
            # interpret single-trailing element as constant diag
            diag_tensor = ConstantDiagLazyTensor(diag, diag_shape=self.shape[-1])
        else:
            try:
                expanded_diag = diag.expand(self.shape[:-1])
            except RuntimeError:
                raise RuntimeError(
                    "add_diag for LazyTensor of size {} received invalid diagonal of size {}.".format(
                        self.shape, diag_shape
                    )
                )
            diag_tensor = DiagLazyTensor(expanded_diag)

        return LowRankRootAddedDiagLazyTensor(self, diag_tensor)

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor
        from .low_rank_root_added_diag_lazy_tensor import LowRankRootAddedDiagLazyTensor

        if isinstance(other, DiagLazyTensor):
            return LowRankRootAddedDiagLazyTensor(self, other)
        else:
            return super().__add__(other)
