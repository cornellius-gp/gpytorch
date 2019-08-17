from gpytorch.lazy import KroneckerProductLazyTensor, AddedDiagLazyTensor
from gpytorch.lazy import lazify
import torch
import warnings
from gpytorch.lazy.sum_lazy_tensor import SumLazyTensor
from gpytorch.lazy.diag_lazy_tensor import DiagLazyTensor
from gpytorch.lazy.psd_sum_lazy_tensor import PsdSumLazyTensor
from gpytorch.lazy.root_lazy_tensor import RootLazyTensor
from gpytorch.utils import broadcasting, pivoted_cholesky, woodbury
from gpytorch import settings

from spectralgp.lazy.diag_kron import DiagKron

#TODO: @greg, does this work????

class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):

    def __init__(self, *lazy_tensors):
        super(KroneckerProductAddedDiagLazyTensor, self).__init__(*lazy_tensors)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. " "Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")


    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # print("Calling the right inv quad logdet")
        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs)
        else:
            inv_quad_term = None

        if logdet is not None:
            logdet_term = self.logdet()
        else:
            logdet_term = None

        return inv_quad_term, logdet_term


    def logdet(self):
        res = 0
        noise = self._diag_tensor[0,0]
        ### THIS WORKS FOR 2 DIMENSIONS NOW ###
        sub_eigs = []
        for lt in self._lazy_tensor.lazy_tensors:
            sub_eigs.append(lt.evaluate().eig()[0][:, 0].unsqueeze(-1))

        eigs = sub_eigs[0].matmul(sub_eigs[1].t())
        return torch.log(eigs + noise).sum()

    def KronSVD(self):
        return [lt.evaluate().svd() for lt in self._lazy_tensor.lazy_tensors]

    def inv_quad(self, rhs):
        svd_list = self.KronSVD()
        noise = self._diag_tensor[0,0]
        V = DiagKron(DiagLazyTensor(svd_list[0].S),
                     DiagLazyTensor(svd_list[1].S))
        Q = KroneckerProductLazyTensor(lazify(svd_list[0].U),
                                       lazify(svd_list[1].U))
        for sub_ind in range(2, len(svd_list)):
            V = KroneckerProductLazyTensor(V, DiagLazyTensor(svd_list[sub_ind].S))
            Q = KroneckerProductLazyTensor(Q, LazyTensor(svd_list[sub_ind].S))

        ## this is a real memory hog ##
        inv_mat = DiagLazyTensor(V.get_diag() + noise)

        res = Q.t().matmul(rhs)
        res = inv_mat.inverse().matmul(res)
        res = Q.matmul(res)

        return rhs.t().matmul(res).squeeze()
