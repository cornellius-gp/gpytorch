import math

import numpy as np
import torch
from scipy.special import ellipj, ellipk

from .. import settings
from .lanczos import lanczos_tridiag
from .minres import minres


def contour_integral_quad(
    lazy_tensor, rhs, inverse=False, weights=None, shifts=None, max_lanczos_iter=7, num_contour_quadrature=7
):
    r"""
    Performs :math:`\mathbf K^{1/2} \mathbf b` or `\mathbf K^{-1/2} \mathbf b`
    using contour integral quadrature.

    .. note::
        Right now this only works for non-batch matrices

    :param gpytorch.lazy.LazyTensor lazy_tensor: LazyTensor representing :math:`\mathbf K`
    :param torch.Tensor rhs: Right hand side tensor :math:`\mathbf b`
    :param bool inverse: (default False) whether to compute :math:`\mathbf K^{1/2} \mathbf b` (if False)
        or `\mathbf K^{-1/2} \mathbf b` (if True)
    :param int max_lanczos_iter: (default 10) Number of Lanczos iterations to run (to estimate eigenvalues)
    :param int num_contour_quadrature: (default 15) How many quadrature samples to use for approximation
    :rtype: torch.Tensor
    :return: Approximation to :math:`\mathbf K^{1/2} \mathbf b` or :mathbf:`\mathbf K^{-1/2} \mathbf b`.
    """
    if shifts is None:
        lanczos_basis, lanczos_mat = lanczos_tridiag(
            lambda v: lazy_tensor._matmul(v),
            init_vecs=rhs[..., :, :1],
            dtype=rhs.dtype,
            device=rhs.device,
            matrix_shape=lazy_tensor.matrix_shape,
            batch_shape=lazy_tensor.batch_shape,
            max_iter=max_lanczos_iter,
        )

        """
        K^{-1/2} b = 2/pi \int_0^\infty (K - t^2 I)^{-1} dt
        We'll approximate this integral as a sum using quadrature
        We'll determine the appropriate values of t, as well as their weights using elliptical integrals
        """

        # Compute an approximate condition number
        # We'll do this with Lanczos
        try:
            approx_eigs = lanczos_mat.symeig()[0]
        except RuntimeError:
            approx_eigs = lazy_tensor.diag()
        min_eig = approx_eigs.min(dim=-1)[0]
        max_eig = approx_eigs.max(dim=-1)[0]
        k2 = min_eig / max_eig
        if settings.record_ciq_stats.on():
            settings.record_ciq_stats.condition_number = 1.0 / k2.mean().item()

        # Compute the shifts needed for the contour
        flat_shifts = torch.zeros(num_contour_quadrature + 1, k2.numel(), dtype=k2.dtype, device=k2.device)
        flat_weights = torch.zeros(num_contour_quadrature, k2.numel(), dtype=k2.dtype, device=k2.device)

        # For loop because numpy
        for i, (sub_k2, sub_min_eig) in enumerate(zip(k2.flatten().tolist(), min_eig.flatten().tolist())):
            # Compute shifts
            Kp = ellipk(1 - sub_k2)  # Elliptical integral of the first kind
            N = num_contour_quadrature
            t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
            sn, cn, dn, _ = ellipj(np.imag(t), 1 - sub_k2)  # Jacobi elliptic functions
            cn = 1.0 / cn
            dn = dn * cn
            sn = 1j * sn * cn
            w = np.sqrt(sub_min_eig) * sn
            w_pow2 = np.real(np.power(w, 2))
            sub_shifts = torch.tensor(w_pow2, dtype=rhs.dtype, device=rhs.device)

            # Compute weights
            constant = -2 * Kp * np.sqrt(sub_min_eig) / (math.pi * N)
            dzdt = torch.tensor(cn * dn, dtype=rhs.dtype, device=rhs.device)
            dzdt.mul_(constant)
            sub_weights = dzdt

            # Store results
            flat_shifts[1:, i].copy_(sub_shifts)
            flat_weights[:, i].copy_(sub_weights)

        weights = flat_weights.view(num_contour_quadrature, *k2.shape, 1, 1)
        shifts = flat_shifts.view(num_contour_quadrature + 1, *k2.shape)

    # Compute the solves at the given shifts
    # Do one more matmul if we don't want to include the inverse
    solves = minres(lambda v: lazy_tensor._matmul(v), rhs, value=-1, shifts=shifts)
    no_shift_solves = solves[0]
    solves = solves[1:]
    if not inverse:
        solves = lazy_tensor._matmul(solves)

    # Record some stats on how good the solves are
    if settings.record_ciq_stats.on():
        with torch.no_grad():
            settings.record_ciq_stats.minres_residual = (
                (lazy_tensor @ no_shift_solves + rhs)
                .div_(rhs.norm(dim=-2, keepdim=True).clamp_min_(1e-10))
                .norm(dim=-2)
                .mean()
                .item()
            )

    return solves, weights, no_shift_solves, shifts
