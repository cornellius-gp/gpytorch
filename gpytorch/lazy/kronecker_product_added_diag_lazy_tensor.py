#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
from torch import Tensor

from .. import settings
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import ConstantDiagLazyTensor, DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductDiagLazyTensor, KroneckerProductLazyTensor
from .lazy_tensor import LazyTensor
from .matmul_lazy_tensor import MatmulLazyTensor


def _constant_kpadlt_constructor(lt, dlt):
    # computes the components of the diagonal solve for constant diagonals
    # Each sub-matrix D_i^{-1} has constant diagonal, so we may scale the eigenvalues of the
    # eigendecomposition of K_i by its inverse to get an eigendecomposition of K_i D_i^{-1}.
    sub_evals, sub_evecs = [], []
    for lt_, dlt_ in zip(lt.lazy_tensors, dlt.lazy_tensors):
        evals_, evecs_ = lt_.diagonalization()
        sub_evals.append(DiagLazyTensor(evals_ / dlt_.diag_values))
        sub_evecs.append(evecs_)
    evals = KroneckerProductDiagLazyTensor(*sub_evals)
    evals_p_i = DiagLazyTensor(evals.diag() + 1.0)
    evecs = KroneckerProductLazyTensor(*sub_evecs)
    return evals_p_i, evecs


def _symmetrize_kpadlt_constructor(lt, dlt):
    # computes the components of the symmetrization solve.
    # (K + D)^{-1} = D^{-1/2}(D^{-1/2}KD^{-1/2} + I)^{-1}D^{-1/2}

    dlt_inv_root = dlt.sqrt().inverse()
    symm_prod = KroneckerProductLazyTensor(
        *[d.matmul(k).matmul(d) for k, d in zip(lt.lazy_tensors, dlt_inv_root.lazy_tensors)]
    )
    evals, evecs = symm_prod.diagonalization()
    evals_plus_i = DiagLazyTensor(evals + 1.0)

    return dlt_inv_root, evals_plus_i, evecs


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        super().__init__(*lazy_tensors, preconditioner_override=preconditioner_override)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[0]
            self.lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[1]
            self.lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")
        self._diag_is_constant = isinstance(self.diag_tensor, ConstantDiagLazyTensor)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
        else:
            inv_quad_term = None
        logdet_term = self._logdet() if logdet else None
        return inv_quad_term, logdet_term

    def _logdet(self):
        if self._diag_is_constant:
            # symeig requires computing the eigenvectors for it to be differentiable
            evals, _ = self.lazy_tensor.symeig(eigenvectors=True)
            evals_plus_diag = evals + self.diag_tensor.diag()
            return torch.log(evals_plus_diag).sum(dim=-1)
        if self.shape[-1] >= settings.max_cholesky_size.value() and isinstance(
            self.diag_tensor, KroneckerProductDiagLazyTensor
        ):
            # If the diagonal has the same Kronecker structure as the full matrix, with each factor being
            # constant, wee can compute the logdet efficiently
            if len(self.lazy_tensor.lazy_tensors) == len(self.diag_tensor.lazy_tensors) and all(
                isinstance(dt, ConstantDiagLazyTensor) for dt in self.diag_tensor.lazy_tensors
            ):
                # here the log determinant identity is |D + K| = | D| |I + D^{-1} K|
                # as D is assumed to have constant components, we can look solely at the diag_values
                diag_term = self.diag_tensor.diag().clamp(min=1e-7).log().sum(dim=-1)
                # symeig requires computing the eigenvectors for it to be differentiable
                evals, _ = self.lazy_tensor.symeig(eigenvectors=True)
                const_times_evals = KroneckerProductLazyTensor(
                    *[ee * d.diag_values for ee, d in zip(evals.lazy_tensors, self.diag_tensor.lazy_tensors)]
                )
                first_term = (const_times_evals.diag() + 1).log().sum(dim=-1)
                return diag_term + first_term

            else:
                # we use the same matrix determinant identity: |K + D| = |D| |I + D^{-1}K|
                # but have to symmetrize the second matrix because torch.eig may not be
                # completely differentiable.
                lt = self.lazy_tensor
                dlt = self.diag_tensor
                if isinstance(lt, KroneckerProductAddedDiagLazyTensor):
                    raise NotImplementedError(
                        "Log determinant for KroneckerProductAddedDiagLazyTensor + " "DiagLazyTensor not implemented."
                    )
                else:
                    _, evals_plus_i, _ = _symmetrize_kpadlt_constructor(lt, dlt)

                diag_term = self.diag_tensor.logdet()
                return diag_term + evals_plus_i.logdet()

        return super().inv_quad_logdet(logdet=True)[1]

    def _preconditioner(self):
        # solves don't use CG so don't waste time computing it
        return None, None, None

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):

        rhs_dtype = rhs.dtype

        # if the diagonal is constant, we can solve this using the Kronecker-structured eigendecomposition
        # and performing a spectral shift of its eigenvalues
        if self._diag_is_constant:
            # we perform the solve in double for numerical stability issues
            # TODO: Use fp64 registry once #1213 is addressed
            evals, q_matrix = self.lazy_tensor.to(torch.double).diagonalization()
            evals_plus_diagonal = evals + self.diag_tensor.diag().double()
            evals_root = evals_plus_diagonal.pow(0.5)
            inv_mat_sqrt = DiagLazyTensor(evals_root.reciprocal())
            res = q_matrix.transpose(-2, -1).matmul(rhs.double())
            res2 = inv_mat_sqrt.matmul(res)
            lazy_lhs = q_matrix.matmul(inv_mat_sqrt)
            return lazy_lhs.matmul(res2).type(rhs_dtype)

        # If the diagonal has the same Kronecker structure as the full matrix, we can perform the solve
        # efficiently by using the Woodbury matrix identity
        if isinstance(self.lazy_tensor, KroneckerProductAddedDiagLazyTensor):
            kron_lazy_tensors = self.lazy_tensor.lazy_tensor.lazy_tensors
        else:
            kron_lazy_tensors = self.lazy_tensor.lazy_tensors
        if (
            isinstance(self.diag_tensor, KroneckerProductDiagLazyTensor)
            and len(kron_lazy_tensors) == len(self.diag_tensor.lazy_tensors)
            and all(
                tfull.shape == tdiag.shape for tfull, tdiag in zip(kron_lazy_tensors, self.diag_tensor.lazy_tensors)
            )
        ):
            # We have
            #   (K + D)^{-1} = K^{-1} - K^{-1} (K D^{-1} + I)^{-1}
            #                = K^{-1} - K^{-1} (\kron_i{K_i D_i^{-1}} + I)^{-1}
            #
            # and so with an eigendecomposition \kron_i{K_i D_i^{-1}} = S Lambda S, we can solve (K + D) = b as
            # K^{-1}(b - S (Lambda + I)^{-1} S^T b).

            # again we perform the solve in double precision for numerical stability issues
            # TODO: Use fp64 registry once #1213 is addressed
            rhs = rhs.double()
            lt = self.lazy_tensor.to(torch.double)
            dlt = self.diag_tensor.to(torch.double)

            # If each of the diagonal factors is constant, life gets a little easier
            # as we can reuse the eigendecomposition
            # (K + D)^{-1} = D^{-1} Q(\kron d_i^{-1} \Lambda_i + I)^{-1} Q^\top
            if all(isinstance(tdiag, ConstantDiagLazyTensor) for tdiag in dlt.lazy_tensors):
                evals_p_i, evecs = _constant_kpadlt_constructor(lt, dlt)
                res1 = evals_p_i.inv_matmul(evecs._transpose_nonbatch().matmul(rhs))
                res = dlt.inv_matmul(evecs.matmul(res1))
                return res.to(rhs_dtype)

            # If the diagonals are not constant, we have to do some more work
            # since K D^{-1} is generally not symmetric. TODO: implement this solve.
            if isinstance(lt, KroneckerProductAddedDiagLazyTensor):
                raise (
                    NotImplementedError(
                        "Inverses of KroneckerProductAddedDiagonals and ConstantDiagLazyTensors are "
                        + "not implemented yet."
                    )
                )
            # in this case we can pull across the diagonals
            # (\otimes K_i + \otimes D_i) = (\otimes D_i^{1/2})
            #   (\otimes D_i^{-1/2}K_iD_i^{-1/2} + I)(\otimes D_i^{1/2})
            # so that
            # (\otimes K_i + \otimes D_i)^{-1} = (\otimes D_i^{1/2})^{-1}
            #   \tilde Q (\tilde \Lambda + I)^{-1} \tilde Q (\otimes D_i^{1/2})
            # Reference: Rakitsch, et al, 2013. "It is all in the noise,"
            # https://papers.nips.cc/paper/2013/file/59c33016884a62116be975a9bb8257e3-Paper.pdf

            dlt_inv_root, evals_p_i, evecs = _symmetrize_kpadlt_constructor(lt, dlt)

            res1 = evecs._transpose_nonbatch().matmul(dlt_inv_root.matmul(rhs))
            res2 = evals_p_i.inv_matmul(res1)
            res3 = evecs.matmul(res2)
            res = dlt_inv_root.matmul(res3)
            return res.to(rhs_dtype)

        # in all other cases we fall back to the default
        return super()._solve(rhs, preconditioner=preconditioner, num_tridiag=num_tridiag)

    def _root_decomposition(self):
        if self._diag_is_constant:
            evals, q_matrix = self.lazy_tensor.diagonalization()
            updated_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(0.5))
            return MatmulLazyTensor(q_matrix, updated_evals)

        dlt = self.diag_tensor
        lt = self.lazy_tensor
        if isinstance(self.diag_tensor, KroneckerProductDiagLazyTensor):
            if all(isinstance(tdiag, ConstantDiagLazyTensor) for tdiag in dlt.lazy_tensors):
                evals_p_i, evecs = _constant_kpadlt_constructor(lt, dlt)
                evals_p_i_root = DiagLazyTensor(evals_p_i.diag().sqrt())
                # here we need to scale the eigenvectors by the constants as
                # A = D^{1/2} Q (\kron a_i^{-1} \Lambda_i + I) Q^\top D^{1/2}
                # so that we compute
                # L = D^{1/2} Q (\kron a_i^{-1} \Lambda_i + I)^{1/2}
                #       = (\kron a_i^{1/2} Q_i)(\kron a_i^{-1} \Lambda_i + I)^{1/2}
                scaled_evecs_list = []
                for evec_, dlt_ in zip(evecs.lazy_tensors, dlt.lazy_tensors):
                    scaled_evecs_list.append(evec_ * dlt_.diag_values.sqrt())
                scaled_evecs = KroneckerProductLazyTensor(*scaled_evecs_list)
                return MatmulLazyTensor(scaled_evecs, evals_p_i_root)

            # again, we compute the root decomposition by pulling across the diagonals
            dlt_root = dlt.sqrt()
            _, evals_p_i, evecs = _symmetrize_kpadlt_constructor(lt, dlt)
            evals_p_i_root = DiagLazyTensor(evals_p_i.diag().sqrt())
            return MatmulLazyTensor(dlt_root, MatmulLazyTensor(evecs, evals_p_i_root))

        return super()._root_decomposition()

    def _root_inv_decomposition(self, initial_vectors=None):
        if self._diag_is_constant:
            evals, q_matrix = self.lazy_tensor.diagonalization()
            inv_sqrt_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(-0.5))
            return MatmulLazyTensor(q_matrix, inv_sqrt_evals)

        dlt = self.diag_tensor
        lt = self.lazy_tensor
        if isinstance(self.diag_tensor, KroneckerProductDiagLazyTensor):
            if all(isinstance(tdiag, ConstantDiagLazyTensor) for tdiag in dlt.lazy_tensors):
                evals_p_i, evecs = _constant_kpadlt_constructor(lt, dlt)
                evals_p_i_inv_root = DiagLazyTensor(evals_p_i.diag().reciprocal().sqrt())
                # here we need to scale the eigenvectors by the constants as
                # A = D^{1/2} Q (\kron a_i^{-1} \Lambda_i + I) Q^\top D^{1/2}
                # so that we compute
                # L^{-1/2} = D^{1/2} Q (\kron a_i^{-1} \Lambda_i + I)^{1/2}
                #       = (\kron a_i^{1/2} Q_i)(\kron a_i^{-1} \Lambda_i + I)^{-1/2}
                scaled_evecs_list = []
                for evec_, dlt_ in zip(evecs.lazy_tensors, dlt.lazy_tensors):
                    scaled_evecs_list.append(evec_ * dlt_.diag_values.sqrt())
                scaled_evecs = KroneckerProductLazyTensor(*scaled_evecs_list)
                return MatmulLazyTensor(scaled_evecs, evals_p_i_inv_root)

            # again, we compute the root decomposition by pulling across the diagonals
            dlt_sqrt, evals_p_i, evecs = _symmetrize_kpadlt_constructor(lt, dlt)
            dlt_inv_root = dlt_sqrt.inverse()
            evals_p_i_root = DiagLazyTensor(evals_p_i.diag().reciprocal().sqrt())
            return MatmulLazyTensor(dlt_inv_root, MatmulLazyTensor(evecs, evals_p_i_root))

        return super()._root_inv_decomposition(initial_vectors=initial_vectors)

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LazyTensor]]:
        # return_evals_as_lazy is a flag to return the eigenvalues as a lazy tensor
        # which is useful for root decompositions here (see the root_decomposition
        # method above)
        if self._diag_is_constant:
            evals, evecs = self.lazy_tensor.symeig(eigenvectors=eigenvectors)
            evals = evals + self.diag_tensor.diag_values

            return evals, evecs
        return super()._symeig(eigenvectors=eigenvectors)

    def __add__(self, other):
        if isinstance(other, ConstantDiagLazyTensor) and self._diag_is_constant:
            # the other cases have only partial implementations
            return KroneckerProductAddedDiagLazyTensor(self.lazy_tensor, self.diag_tensor + other)
        return super().__add__(other)
