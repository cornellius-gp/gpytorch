#!/usr/bin/env python3

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from .. import settings
from ..utils import broadcasting, pivoted_cholesky
from ..utils.memoize import cached
from ..utils.warnings import NumericalWarning
from .diag_lazy_tensor import ConstantDiagLazyTensor, DiagLazyTensor
from .lazy_tensor import LazyTensor
from .psd_sum_lazy_tensor import PsdSumLazyTensor
from .root_lazy_tensor import RootLazyTensor
from .sum_lazy_tensor import SumLazyTensor


class AddedDiagLazyTensor(SumLazyTensor):
    """
    A SumLazyTensor, but of only two lazy tensors, the second of which must be
    a DiagLazyTensor.
    """

    def __init__(self, *lazy_tensors, preconditioner_override=None):
        lazy_tensors = list(lazy_tensors)
        super(AddedDiagLazyTensor, self).__init__(*lazy_tensors, preconditioner_override=preconditioner_override)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")

        broadcasting._mul_broadcast_shape(lazy_tensors[0].shape, lazy_tensors[1].shape)

        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

        self.preconditioner_override = preconditioner_override

        # Placeholders
        self._constant_diag = None
        self._noise = None
        self._piv_chol_self = None  # <- Doesn't need to be an attribute, but used for testing purposes
        self._precond_lt = None
        self._precond_logdet_cache = None
        self._q_cache = None
        self._r_cache = None

    def _matmul(self, rhs):
        return torch.addcmul(self._lazy_tensor._matmul(rhs), self._diag_tensor._diag.unsqueeze(-1), rhs)

    def add_diag(self, added_diag):
        return self.__class__(self._lazy_tensor, self._diag_tensor.add_diag(added_diag))

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor

        if isinstance(other, DiagLazyTensor):
            return self.__class__(self._lazy_tensor, self._diag_tensor + other)
        else:
            return self.__class__(self._lazy_tensor + other, self._diag_tensor)

    def _preconditioner(self):
        r"""
        Here we use a partial pivoted Cholesky preconditioner:

        K \approx L L^T + D

        where L L^T is a low rank approximation, and D is a diagonal.
        We can compute the preconditioner's inverse using Woodbury

        (L L^T + D)^{-1} = D^{-1} - D^{-1} L (I + L D^{-1} L^T)^{-1} L^T D^{-1}

        This function returns:
        - A function `precondition_closure` that computes the solve (L L^T + D)^{-1} x
        - A LazyTensor `precondition_lt` that represents (L L^T + D)
        - The log determinant of (L L^T + D)
        """

        if self.preconditioner_override is not None:
            return self.preconditioner_override(self)

        if settings.max_preconditioner_size.value() == 0 or self.size(-1) < settings.min_preconditioning_size.value():
            return None, None, None

        # Cache a QR decomposition [Q; Q'] R = [D^{-1/2}; L]
        # This makes it fast to compute solves and log determinants with it
        #
        # Through woodbury, (L L^T + D)^{-1} reduces down to (D^{-1} - D^{-1/2} Q Q^T D^{-1/2})
        # Through matrix determinant lemma, log |L L^T + D| reduces down to 2 log |R|
        if self._q_cache is None:
            max_iter = settings.max_preconditioner_size.value()
            self._piv_chol_self = pivoted_cholesky.pivoted_cholesky(self._lazy_tensor, max_iter)
            if torch.any(torch.isnan(self._piv_chol_self)).item():
                warnings.warn(
                    "NaNs encountered in preconditioner computation. Attempting to continue without preconditioning.",
                    NumericalWarning,
                )
                return None, None, None
            self._init_cache()

        # NOTE: We cannot memoize this precondition closure as it causes a memory leak
        def precondition_closure(tensor):
            # This makes it fast to compute solves with it
            qqt = self._q_cache.matmul(self._q_cache.transpose(-2, -1).matmul(tensor))
            if self._constant_diag:
                return (1 / self._noise) * (tensor - qqt)
            return (tensor / self._noise) - qqt

        return (precondition_closure, self._precond_lt, self._precond_logdet_cache)

    def _init_cache(self):
        *batch_shape, n, k = self._piv_chol_self.shape
        self._noise = self._diag_tensor.diag().unsqueeze(-1)

        # the check for constant diag needs to be done carefully for batches.
        noise_first_element = self._noise[..., :1, :]
        self._constant_diag = torch.equal(self._noise, noise_first_element * torch.ones_like(self._noise))
        eye = torch.eye(k, dtype=self._piv_chol_self.dtype, device=self._piv_chol_self.device)
        eye = eye.expand(*batch_shape, k, k)

        if self._constant_diag:
            self._init_cache_for_constant_diag(eye, batch_shape, n, k)
        else:
            self._init_cache_for_non_constant_diag(eye, batch_shape, n)

        self._precond_lt = PsdSumLazyTensor(RootLazyTensor(self._piv_chol_self), self._diag_tensor)

    def _init_cache_for_constant_diag(self, eye, batch_shape, n, k):
        # We can factor out the noise for for both QR and solves.
        self._noise = self._noise.narrow(-2, 0, 1)
        self._q_cache, self._r_cache = torch.qr(torch.cat((self._piv_chol_self, self._noise.sqrt() * eye), dim=-2))
        self._q_cache = self._q_cache[..., :n, :]

        # Use the matrix determinant lemma for the logdet, using the fact that R'R = L_k'L_k + s*I
        logdet = self._r_cache.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1).mul(2)
        logdet = logdet + (n - k) * self._noise.squeeze(-2).squeeze(-1).log()
        self._precond_logdet_cache = logdet.view(*batch_shape) if len(batch_shape) else logdet.squeeze()

    def _init_cache_for_non_constant_diag(self, eye, batch_shape, n):
        # With non-constant diagonals, we cant factor out the noise as easily
        self._q_cache, self._r_cache = torch.qr(torch.cat((self._piv_chol_self / self._noise.sqrt(), eye), dim=-2))
        self._q_cache = self._q_cache[..., :n, :] / self._noise.sqrt()

        # Use the matrix determinant lemma for the logdet, using the fact that R'R = L_k'L_k + s*I
        logdet = self._r_cache.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1).mul(2)
        logdet -= (1.0 / self._noise).log().sum([-1, -2])
        self._precond_logdet_cache = logdet.view(*batch_shape) if len(batch_shape) else logdet.squeeze()

    @cached(name="svd")
    def _svd(self) -> Tuple["LazyTensor", Tensor, "LazyTensor"]:
        if isinstance(self._diag_tensor, ConstantDiagLazyTensor):
            U, S_, V = self._lazy_tensor.svd()
            S = S_ + self._diag_tensor.diag()
            return U, S, V
        return super()._svd()

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LazyTensor]]:
        if isinstance(self._diag_tensor, ConstantDiagLazyTensor):
            evals_, evecs = self._lazy_tensor.symeig(eigenvectors=eigenvectors)
            evals = evals_ + self._diag_tensor.diag()
            return evals, evecs
        return super()._symeig(eigenvectors=eigenvectors)

    def evaluate_kernel(self):
        """
        Overriding this is currently necessary to allow for subclasses of AddedDiagLT to be created. For example,
        consider the following:

            >>> covar1 = covar_module(x).add_diag(torch.tensor(1.)).evaluate_kernel()
            >>> covar2 = covar_module(x).evaluate_kernel().add_diag(torch.tensor(1.))

        Unless we override this method (or find a better solution), covar1 and covar2 might not be the same type.
        In particular, covar1 would *always* be a standard AddedDiagLazyTensor, but covar2 might be a subtype.
        """
        added_diag_lazy_tsr = self.representation_tree()(*self.representation())
        return added_diag_lazy_tsr._lazy_tensor + added_diag_lazy_tsr._diag_tensor
