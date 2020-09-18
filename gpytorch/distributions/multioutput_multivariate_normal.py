#!/usr/bin/env python3

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.normal import Normal

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy.lazy_tensor import LazyTensor


class NormalizedOrder(Enum):
    FULL = "nmnm"
    OUTPUT_INDEPENDENT = "mnn"
    POINT_INDEPENDENT = "nmm"
    INDEPENDENT = "nm"


class MultioutputMultivariateNormal(MultivariateNormal):
    def __init__(
        self, mean: Tensor, covariance: Tensor, order: Tuple[str, ...] = ("n", "m", "n", "m")
    ) -> MultioutputMultivariateNormal:
        """Construct a MultioutputMultivariateNormal of `m` outputs across `n` datapoints from a stacked
        representation of its joint covariance matrix.

        :param torch.tensor mean: A `batch_shape x n x m` tensor of means.
        :param torch.tensor covariance: A `batch_shape x shape`-dim tensor, where `shape` has between two and
            or four elements and can be any combination of "n" and "m" with at least one and at most two
            occurrences of both "n" and "m".
        :param Tuple[str] order: A tuple of strings indicating the order of the terms in the covariance tensor

        Internally, the covariance is represented consistently a one of the following:
            - `n x m n x m` (general case)
            - `m x n x n` (case of cross-output independence)
            - `n x m x m` (case of cross-datapoint independence)
            - `n x m` (degenerate cse of a collection of Normals independent across data points and outputs)

        """
        order = "".join(order)
        if "".join(sorted(order)) not in {"".join(sorted(o.value)) for o in NormalizedOrder}:
            raise ValueError(f"Invalid order '{order}'.")
        self._mean = mean
        self._batch_shape = mean.shape[:-2]
        # check consistency of covar and mean inputs
        dims = {"n": mean.shape[-2], "m": mean.shape[-1]}
        for o, d in zip(order, covariance.shape[-len(order) :]):
            if d != dims[o]:
                raise ValueError("Supplied order inconsistent with covariance dimensions")
        b = len(self._batch_shape)
        self._covar, self._nlzd_order = _normalize_covar(covariance=covariance, order=order, b=b)

    @classmethod
    def from_independent_mvns(cls, mvns: Iterable[MultivariateNormal]) -> MultioutputMultivariateNormal:
        mean = torch.stack([mvn.mean for mvn in mvns], dim=-1)
        covariance = torch.cat([mvn.covariance_matrix.unsqueeze(-3) for mvn in mvns], dim=-3)
        return cls(mean=mean, covariance=covariance, order=NormalizedOrder.OUTPUT_INDEPENDENT.value)

    @classmethod
    def from_batch_mvn(cls, batch_mvn: MultivariateNormal, output_dim: int = -1) -> MultioutputMultivariateNormal:
        mean = batch_mvn.mean
        batch_shape = mean.shape[:-1]
        if output_dim < 0:
            output_dim += len(batch_shape)
        mean = mean.permute(*range(0, output_dim), *range(output_dim + 1, mean.ndim), output_dim)
        cov = batch_mvn.covariance_matrix
        covariance = cov.permute(*range(0, output_dim), *range(output_dim + 1, cov.ndim - 2), output_dim, -2, -1)
        return cls(mean=mean, covariance=covariance, order=NormalizedOrder.OUTPUT_INDEPENDENT.value)

    @property
    def covariance_matrix(self):
        raise NotImplementedError

    @property
    def lazy_covariance_matrix(self):
        raise NotImplementedError

    @property
    def event_shape(self) -> torch.Size:
        return self._mean.shape[-2:]

    @property
    def num_outcomes(self) -> int:
        return self.event_shape[-1]

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        if self._nlzd_order == NormalizedOrder.INDEPENDENT:
            return self._covar
        if self._nlzd_order == NormalizedOrder.OUTPUT_INDEPENDENT:
            return torch.diagonal(self._covar, dim1=-1, dim2=-1).transpose(-1, -2)
        if self._nlzd_order == NormalizedOrder.POINT_INDEPENDENT:
            return torch.diagonal(self._covar, dim1=-1, dim2=-1)
        C = torch.diagonal(self._covar, dim1=-1, dim2=-3)
        return torch.diagonal(C, dim1=-3, dim2=-2).transpose(-1, -2)

    def expand(self, batch_size: Tuple[int, ...]) -> MultioutputMultivariateNormal:
        mean = self.mean.expand(torch.Size(batch_size) + self.mean.shape[-2:])
        nldz_order_str = self._nlzd_order.value
        covariance = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-len(nldz_order_str) :])
        return self.__class__(mean=mean, covariance=covariance, order=nldz_order_str)

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        if base_samples is not None:
            raise NotImplementedError("Base samples not yet supported in MultioutputMultivariateNormal.")
        if self._nlzd_order == NormalizedOrder.INDEPENDENT:
            zero_mean_samples = self._covar.sqrt() * torch.randn(*sample_shape, *self._covar.shape)
        elif self._nlzd_order == NormalizedOrder.OUTPUT_INDEPENDENT:
            # TODO: Use lazies + caching!
            L = torch.cholesky(self._covar)
            eps = torch.randn(*sample_shape, *L.shape[:-1], 1)
            zero_mean_samples = (L @ eps).squeeze(-1).transpose(-1, -2)
        elif self._nlzd_order == NormalizedOrder.POINT_INDEPENDENT:
            # TODO: Use lazies + caching!
            L = torch.cholesky(self._covar)
            eps = torch.randn(*sample_shape, *L.shape[:-1], 1)
            zero_mean_samples = (L @ eps).squeeze(-1)
        else:
            # TODO: Use lazies + caching!
            n, m = self.event_shape
            covar = self._covar.reshape(*self._batch_shape, n * m, n * m)
            L = torch.cholesky(covar)
            eps = torch.randn(*sample_shape, *L.shape[:-1], 1)
            s = (L @ eps).squeeze(-1)
            zero_mean_samples = s.reshape(*sample_shape, *self._batch_shape, n, m)
        return self.mean + zero_mean_samples

    def log_prob(self, value: Tensor) -> Tensor:
        if self._nlzd_order == NormalizedOrder.INDEPENDENT:
            N = Normal(self.mean, self._covar.sqrt())
            return N.log_prob(value)
        elif self._nlzd_order == NormalizedOrder.OUTPUT_INDEPENDENT:
            # TODO: Use lazies and caching
            ust = torch.cholesky(self._covar)
            loc = self.mean.transpose(-1, -2)
            logprobs = mvn_log_prob(loc, ust, value.transpose(-1, -2))
            return logprobs.sum(-1)
        elif self._nlzd_order == NormalizedOrder.POINT_INDEPENDENT:
            # TODO: Use lazies and caching
            ust = torch.cholesky(self._covar)
            logprobs = mvn_log_prob(self.mean, ust, value)
            return logprobs.sum(-1)
        else:
            n, m = self.event_shape
            covariance = self._covar.reshape(*self.batch_shape, n * m, n * m)
            # TODO: Use lazies and caching
            ust = torch.cholesky(covariance)
            loc = self.mean.reshape(-1, n * m)
            return mvn_log_prob(loc, ust, value.view(*value.shape[:-2], -1))

    def __getitem__(self, index) -> MultioutputMultivariateNormal:
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), len(self._batch_shape) + 2
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims - len(index))
        new_mean = self.mean[index]
        if self._nlzd_order == NormalizedOrder.INDEPENDENT:
            new_covar = self._covar[index]
        n_idx, m_idx = index[-2:]
        if self._nlzd_order == NormalizedOrder.FULL:
            tail_idx = (n_idx, m_idx, n_idx, m_idx)
        elif self._nlzd_order == NormalizedOrder.OUTPUT_INDEPENDENT:
            tail_idx = (m_idx, n_idx, n_idx)
        elif self._nlzd_order == NormalizedOrder.POINT_INDEPENDENT:
            tail_idx = (n_idx, m_idx, m_idx)
        else:
            tail_idx = (n_idx, m_idx)
        new_covar = self._covar[(*index[:-2], *tail_idx)]
        # TODO: Consider returing MVN / Normal if indexing results in singleton dimensions
        return self.__class__(mean=new_mean, covariance=new_covar, order=self._nlzd_order.value)


class TaskIndependentLazyMTVN(MultioutputMultivariateNormal):
    def __init__(self, mean: Tensor, covariances: List[LazyTensor]) -> TaskIndependentLazyMTVN:
        self._task_indep = True
        self._point_indep = False
        self._mean = mean
        self._covariances = covariances
        self._batch_shape = mean.shape[:-2]
        n, t = mean.shape[-2:]
        if len(covariances) != t:
            raise ValueError
        if any(C.size(-1) != n for C in covariances):
            raise ValueError

    @property
    def variance(self) -> Tensor:
        return torch.stack([C.diag() for C in self._covariances], dim=-1)

    def expand(self, batch_size: Tuple[int, ...]) -> MultioutputMultivariateNormal:
        mean = self.mean.expand(torch.Size(batch_size) + self.mean.shape[-2:])
        covariances = [C.expand(torch.Size(batch_size) + C.shape[-2:]) for C in self._covariances]
        return self.__class__(mean=mean, covariances=covariances)

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        if base_samples is not None:
            raise NotImplementedError
        if len(set(C.shape for C in self._covariances)) > 1:
            # TODO: Support broadcasting across batch dimensions
            raise RuntimeError("all tasks must have the same shape")
        tkwargs = {"device": self._mean.device, "dtype": self._mean.dtype}
        batch_shape = self._covariances[0].shape[:-2]
        samples = []
        for C in self._covariances:
            # root decomps could be of different shape if low-rank
            root = C.root_decomposition().root
            eps = torch.randn(*sample_shape, *batch_shape, root.size(-1), 1, **tkwargs)
            samples.append(root @ eps)
        return self._mean + torch.cat(samples, dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        diff = value - self._mean
        inv_quads, logdets = [], []
        for i, C in enumerate(self._covariances):
            # allow broadcasting w.r.t batch dimenions of inv_quad_logdet (should clean this up / optimize it!)
            if len(diff.shape[:-1]) < len(C.batch_shape):
                diff = diff.expand(C.shape[:-1], diff.size(-1))
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - C.dim())), *C.batch_shape)
                batch_reps = (
                    diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-2], padded_batch_shape)
                )
                C = C.repeat(*batch_reps, 1, 1)
            # this should use the new triangular lazy tensor to avoid unnecessary compute here
            inv_quad, logdet = C.inv_quad_logdet(inv_quad_rhs=diff[..., i : i + 1], logdet=True)
            inv_quads.append(inv_quad)
            logdets.append(logdet)
        inv_quad = torch.stack(inv_quads, dim=-1)
        logdet = torch.stack(logdets, dim=-1)
        log_prob = -0.5 * sum([inv_quad, logdet, diff.size(-2) * math.log(2 * math.pi)])
        return log_prob.sum(-1)


def mvn_log_prob(loc: Tensor, unbroadcasted_scale_tril: Tensor, value: Tensor) -> Tensor:
    diff = value - loc
    M = _batch_mahalanobis(unbroadcasted_scale_tril, diff)
    half_log_det = unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return -0.5 * (loc.size(-1) * math.log(2 * math.pi) + M) - half_log_det


def _normalize_covar(covariance: Tensor, order: str, b: int) -> Tuple[Tensor, NormalizedOrder]:
    output_indep = order.count("m") == 1
    point_indep = order.count("n") == 1
    if point_indep:
        if output_indep:
            assert covariance.ndim == b + 2
            # normalize to `n` x m`
            nlzd_order = NormalizedOrder("nm")
            if order.find("m") == 0:
                covariance = covariance.transpose(-1, -2)
        else:
            assert covariance.ndim == b + 3
            # normalize to `n x m x m`
            nlzd_order = NormalizedOrder("nmm")
            nloc = order.find("n")
            if nloc != 0:
                covariance = covariance.transpose(-3, -3 + nloc)
    else:
        if output_indep:
            assert covariance.ndim == b + 3
            # normalize to `m x n x n`
            nlzd_order = NormalizedOrder("mnn")
            mloc = order.find("m")
            if mloc != 0:
                covariance = covariance.transpose(-3, -3 + mloc)
        else:
            assert covariance.ndim == b + 4
            # normalize to `n x m x n x m`
            nlzd_order = NormalizedOrder("nmnm")
            nlocs = [i for i, s in enumerate(order) if s == "n"]
            mlocs = [i for i, s in enumerate(order) if s == "m"]
            if nlocs != [0, 2]:
                perm_idcs = (b + nlocs[0], b + mlocs[0], b + nlocs[1], b + mlocs[1])
                covariance = covariance.permute(*range(b), *perm_idcs)
    return covariance, nlzd_order
