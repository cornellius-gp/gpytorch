#!/usr/bin/env python3

from __future__ import annotations

import warnings

import torch
from torch import Tensor


def kmeans_inducing_points(
    train_x: Tensor,
    n_inducing: int,
    max_iter: int = 100,
    batch_size: int | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""
    Select inducing point locations with K-means clustering.

    Runs K-means on ``train_x`` and returns the cluster centroids as
    inducing points. For large datasets, pass ``batch_size`` to use
    mini-batch K-means.

    :param train_x: Training inputs, shape :math:`(N, D)`.
    :type train_x: torch.Tensor
    :param n_inducing: Number of inducing points (clusters).
    :type n_inducing: int
    :param max_iter: Max K-means iterations. Default: ``100``.
    :type max_iter: int
    :param batch_size: Mini-batch size. If ``None``, runs full-batch
        K-means. Default: ``None``.
    :type batch_size: int, optional
    :param seed: Random seed. Default: ``None``.
    :type seed: int, optional
    :return: Inducing points, shape :math:`(M, D)`.
    :rtype: torch.Tensor

    Example:

        >>> train_x = torch.randn(1000, 5)
        >>> inducing_pts = kmeans_inducing_points(train_x, n_inducing=50)
        >>> inducing_pts.shape
        torch.Size([50, 5])
    """
    if train_x.ndim != 2:
        raise ValueError(f"Expected 2D train_x (N, D), got {train_x.shape}")

    n_data, d = train_x.shape

    if n_inducing <= 0:
        raise ValueError(f"n_inducing must be positive, got {n_inducing}")

    if n_inducing >= n_data:
        warnings.warn(
            f"n_inducing ({n_inducing}) >= n_data ({n_data}), returning all training points.",
            UserWarning,
        )
        return train_x.clone()

    generator = torch.Generator(device=train_x.device)
    if seed is not None:
        generator.manual_seed(seed)

    centroids = _kmeans_plusplus(train_x, n_inducing, generator)

    use_minibatch = batch_size is not None and batch_size < n_data

    for _ in range(max_iter):
        batch = train_x
        if use_minibatch:
            idx = torch.randint(n_data, (batch_size,), generator=generator, device=train_x.device)
            batch = train_x[idx]

        dists = torch.cdist(batch, centroids)
        assignments = dists.argmin(dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_inducing, device=train_x.device, dtype=train_x.dtype)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, d), batch)
        counts.scatter_add_(0, assignments, torch.ones(batch.shape[0], device=train_x.device, dtype=train_x.dtype))

        # Keep old centroid for empty clusters
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty].unsqueeze(1)
        new_centroids[~nonempty] = centroids[~nonempty]

        if not use_minibatch:
            shift = (new_centroids - centroids).norm(dim=1).max()
            centroids = new_centroids
            if shift < 1e-6:
                break
        else:
            centroids = new_centroids

    return centroids


def _kmeans_plusplus(data: Tensor, k: int, generator: torch.Generator) -> Tensor:
    """K-means++ centroid initialization."""
    n = data.shape[0]
    device = data.device

    idx = torch.randint(n, (1,), generator=generator, device=device).item()
    centroids = [data[idx]]

    for _ in range(1, k):
        stacked = torch.stack(centroids, dim=0)
        min_dists_sq = torch.cdist(data, stacked).min(dim=1).values.square()
        probs = min_dists_sq / min_dists_sq.sum()
        idx = torch.multinomial(probs, 1, generator=generator).item()
        centroids.append(data[idx])

    return torch.stack(centroids, dim=0)


def median_heuristic_lengthscale(
    train_x: Tensor,
    n_subsample: int = 1000,
    seed: int | None = None,
) -> Tensor:
    r"""
    Compute a lengthscale initialization via the median pairwise distance heuristic.

    Returns the median of pairwise Euclidean distances on a random subsample
    of ``train_x``.

    :param train_x: Training inputs, shape :math:`(N, D)`.
    :type train_x: torch.Tensor
    :param n_subsample: Subsample size for pairwise distances.
        Default: ``1000``.
    :type n_subsample: int
    :param seed: Random seed. Default: ``None``.
    :type seed: int, optional
    :return: Scalar lengthscale.
    :rtype: torch.Tensor

    Example:

        >>> train_x = torch.randn(5000, 10)
        >>> ls = median_heuristic_lengthscale(train_x)
        >>> ls.shape
        torch.Size([])
    """
    if train_x.ndim != 2:
        raise ValueError(f"Expected 2D train_x (N, D), got {train_x.shape}")

    n_data = train_x.shape[0]

    if n_subsample >= n_data:
        subsample = train_x
    else:
        generator = torch.Generator(device=train_x.device)
        if seed is not None:
            generator.manual_seed(seed)
        idx = torch.randperm(n_data, generator=generator, device=train_x.device)[:n_subsample]
        subsample = train_x[idx]

    dists = torch.pdist(subsample)

    if dists.numel() == 0:
        warnings.warn("Too few points for pairwise distances, returning lengthscale=1.0.", UserWarning)
        return torch.tensor(1.0, dtype=train_x.dtype, device=train_x.device)

    return dists.median()
