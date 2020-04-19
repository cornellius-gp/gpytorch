#!/usr/bin/env python3

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import _batch_mahalanobis

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy.lazy_tensor import LazyTensor

ADMISSIBLE_ORDERS = {"nnt", "ntt", "ntn", "ttn", "tnt", "tnn"}


class MultitaskMultivariateNormal(MultivariateNormal):
    def __init__(
        self, mean: Tensor, covariance: Tensor, order: Tuple[str, ...] = ("n", "t", "n", "t")
    ) -> MultitaskMultivariateNormal:
        """Construct a MultitaskMultivariateNormal from a stacked representation of its covariance matrix.

        :param torch.tensor mean: A `batch_shape x n x t` tensor of means
        :param torch.tensor covariance: A `batch_shape x shape`-dim tensor, where `shape` has either three
            or four elements and can be any combination of "n" and "t" with at least one and at most two
            occurrences of both "n" and "t" (currently only n==2 is supported)
        :param Tuple[str] order: A tuple of strings indicating the order of the terms in the covariance tensor

        Internally, the covariance is represented consistently as `n x t n x t` or `t x n x n` (case of
        cross-task independence) in order to simplify indexing / scalarization operations.

        """
        # TODO: Clean this up
        order = "".join(order)
        if len(order) not in {3, 4}:
            raise ValueError
        if order[:3] not in ADMISSIBLE_ORDERS:
            raise ValueError
        cnt = Counter(order)
        if len(cnt) > 2:
            raise ValueError
        elif max(cnt.values()) > 2:
            raise ValueError
        self._task_indep = cnt["t"] == 1
        self._point_indep = cnt["n"] == 1
        if self._point_indep:
            raise NotImplementedError("Independent points currently unsupported")

        self._mean = mean
        self._batch_shape = mean.shape[:-2]
        b = len(self._batch_shape)
        # TODO: Clean this up, avoid enumerating everything
        if order == "ntnt":
            self._covar = covariance
        elif order == "nntt":
            self._covar = covariance.permute(*range(b), -2, -3, -4, -1)
        elif order == "nttn":
            self._covar = covariance.transpose(-1, -2)
        elif order == "ttnn":
            self._covar = covariance.permute(*range(b), -4, -2, -3, -1)
        elif order == "tntn":
            self._covar = covariance.permute(*range(b), -3, -4, -1, -2)
        elif order == "tnnt":
            self._covar = covariance.transpose(-4, -3)
        # cross-task independency
        elif order == "tnn":
            self._covar = covariance
        elif order == "nnt":
            self._covar = covariance.transpose(-3, -1)
        elif order == "ntn":
            self._covar = covariance.transpose(-3, -2)
        else:
            raise ValueError(f"Unsuported order '{order}'")

    @classmethod
    def from_independent_mvns(cls, mvns: Iterable[MultivariateNormal]) -> MultitaskMultivariateNormal:
        mean = torch.stack([mvn.mean for mvn in mvns], dim=-1)
        covariance = torch.cat([mvn.covariance_matrix.unsqueeze(-3) for mvn in mvns], dim=-3)
        return cls(mean=mean, covariance=covariance, order=("t", "n", "n"))

    @classmethod
    def from_batch_mvn(cls, batch_mvn: MultivariateNormal, task_dim: int = -1) -> MultitaskMultivariateNormal:
        mean = batch_mvn.mean
        batch_shape = mean.shape[:-1]
        if task_dim < 0:
            task_dim += len(batch_shape)
        mean = mean.permute(*range(0, task_dim), *range(task_dim + 1, mean.ndim), task_dim)
        cov = batch_mvn.covariance_matrix
        covariance = cov.permute(*range(0, task_dim), *range(task_dim + 1, cov.ndim - 2), task_dim, -2, -1)
        return cls(mean=mean, covariance=covariance, order=("t", "n", "n"))

    @property
    def event_shape(self) -> torch.Size:
        return self._mean.shape[-2:]

    @property
    def num_tasks(self) -> int:
        return self.event_shape[-1]

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        if self._task_indep:
            # _covar is t x n x n
            return torch.diagonal(self._covar, dim1=-1, dim2=-2).transpose(-1, -2)
        # _covar is n x t x n x t
        C = torch.diagonal(self._covar, dim1=-1, dim2=-3)
        return torch.diagonal(C, dim1=-3, dim2=-2).transpose(-1, -2)

    def expand(self, batch_size: Tuple[int, ...]) -> MultitaskMultivariateNormal:
        mean = self.mean.expand(torch.Size(batch_size) + self.mean.shape[-2:])
        k = 3 if self._task_indep else 4
        order = ("t", "n", "n") if self._task_indep else ("n", "t", "n", "t")
        covariance = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-k:])
        return self.__class__(mean=mean, covariance=covariance, order=order)

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        if base_samples is not None:
            raise NotImplementedError
        if self._task_indep:
            # TODO: Use lazies + caching!
            L = torch.cholesky(self._covar)
            eps = torch.randn(*sample_shape, *L.shape[:-1], 1)
            zero_mean_samples = (L @ eps).squeeze(-1).transpose(-1, -2)
        else:
            n, t = self._covar.shape[-2:]
            covar = self._covar.view(*self._batch_shape, n * t, n * t)
            # TODO: Use lazies + caching!
            L = torch.cholesky(covar)
            eps = torch.randn(*sample_shape, *L.shape[:-1], 1)
            s = (L @ eps).squeeze(-1)
            zero_mean_samples = s.reshape(*sample_shape, *self._batch_shape, n, t)

        return self.mean + zero_mean_samples

    def log_prob(self, value: Tensor) -> Tensor:
        if self._task_indep:
            # TODO: Use lazies and caching
            ust = torch.cholesky(self._covar)
            loc = self.mean.transpose(-1, -2)
            logprobs = mvn_log_prob(loc, ust, value.transpose(-1, -2))
            return logprobs.sum(-1)
        else:
            n, t = self.event_shape
            covariance = self._covar.reshape(*self.batch_shape, n * t, n * t)
            # TODO: Use lazies and caching
            ust = torch.cholesky(covariance)
            loc = self.mean.reshape(-1, n * t)
            return mvn_log_prob(loc, ust, value.view(*value.shape[:-2], -1))


class TaskIndependentLazyMTVN(MultitaskMultivariateNormal):
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

    def expand(self, batch_size: Tuple[int, ...]) -> MultitaskMultivariateNormal:
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
