#!/usr/bin/env python3

import math

import torch
from torch import Tensor

from .rff_kernel import RFFKernel


class OrthogonalRandomFeaturesKernel(RFFKernel):
    r"""
    Orthogonal Random Features (ORF) kernel approximation.

    Improves upon :class:`RFFKernel` by using orthogonal random projections
    instead of i.i.d. Gaussian samples. The weight matrix is constructed as
    a block-diagonal of random orthogonal matrices scaled to preserve the
    spectral norm of the standard Gaussian.

    This reduces the variance of the kernel approximation while maintaining
    the same computational complexity :math:`O(nD)`. ORF strictly dominates
    standard RFF in mean squared error of the kernel approximation.

    .. note::
        The weight matrix is re-sampled every time :meth:`_init_weights` is called
        (e.g. at the first forward pass if ``num_dims`` was not provided up front).

    :param num_samples: Number of random features :math:`D`.
    :type num_samples: int
    :param num_dims: (Default ``None``.) Dimensionality of the data space :math:`d`.
        If unspecified, it will be inferred the first time :meth:`forward` is called.
    :type num_dims: int, optional

    Example:
        >>> kernel = gpytorch.kernels.OrthogonalRandomFeaturesKernel(num_samples=512)
        >>> x = torch.randn(100, 5)
        >>> covar = kernel(x, x).to_dense()

    References:
        - Yu et al., *Orthogonal Random Features*, NeurIPS 2016.
        - Choromanski et al., *The Geometry of Random Features*, AISTATS 2018.
    """

    def _get_weight_matrix(self, d: int, D: int) -> Tensor:
        """Build a block-orthogonal weight matrix of shape ``(D, d)``.

        Each block of ``d`` frequency vectors shares an orthogonal direction basis
        (Haar-distributed via QR) but has independent chi(d)-distributed norms,
        so the marginal distribution of each frequency matches ``N(0, I_d)``.
        """
        blocks = math.ceil(D / d)
        Ws = []
        for _ in range(blocks):
            G = torch.randn(d, d, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
            Q, _ = torch.linalg.qr(G)
            # chi(d) norms: norms of independent d-dim Gaussian vectors
            norms = torch.randn(d, d, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device).norm(dim=1)
            Ws.append(norms.unsqueeze(1) * Q)  # scale each row of Q by its chi(d) norm
        W = torch.cat(Ws, dim=0)[:D]  # (D, d)
        return W

    def _init_weights(
        self,
        num_dims: int | None = None,
        num_samples: int | None = None,
        randn_weights: Tensor | None = None,
    ) -> None:
        if randn_weights is None and num_dims is not None and num_samples is not None:
            d, D = num_dims, num_samples
            batch_Ws = []
            batch_shape = self._batch_shape if self._batch_shape else torch.Size([])
            n_batch = int(torch.tensor(list(batch_shape) or [1]).prod().item())
            for _ in range(n_batch):
                batch_Ws.append(self._get_weight_matrix(d, D))
            W = torch.stack(batch_Ws).view(*batch_shape, D, d) if batch_shape else batch_Ws[0]
            # RFFKernel stores weights transposed as (batch..., d, D)
            randn_weights = W.transpose(-1, -2)
        super()._init_weights(num_dims=num_dims, num_samples=num_samples, randn_weights=randn_weights)
