import math
import warnings

import torch

from .. import settings
from .broadcasting import _mul_broadcast_shape
from .linear_cg import linear_cg
from .minres import minres
from .warnings import NumericalWarning


def contour_integral_quad(
    lazy_tensor,
    rhs,
    inverse=False,
    weights=None,
    shifts=None,
    max_lanczos_iter=20,
    num_contour_quadrature=None,
    shift_offset=0,
):
    r"""
    Performs :math:`\mathbf K^{1/2} \mathbf b` or `\mathbf K^{-1/2} \mathbf b`
    using contour integral quadrature.

    :param gpytorch.lazy.LazyTensor lazy_tensor: LazyTensor representing :math:`\mathbf K`
    :param torch.Tensor rhs: Right hand side tensor :math:`\mathbf b`
    :param bool inverse: (default False) whether to compute :math:`\mathbf K^{1/2} \mathbf b` (if False)
        or `\mathbf K^{-1/2} \mathbf b` (if True)
    :param int max_lanczos_iter: (default 10) Number of Lanczos iterations to run (to estimate eigenvalues)
    :param int num_contour_quadrature: How many quadrature samples to use for approximation. Default is in settings.
    :rtype: torch.Tensor
    :return: Approximation to :math:`\mathbf K^{1/2} \mathbf b` or :math:`\mathbf K^{-1/2} \mathbf b`.
    """
    import numpy as np
    from scipy.special import ellipj, ellipk

    if num_contour_quadrature is None:
        num_contour_quadrature = settings.num_contour_quadrature.value()

    output_batch_shape = _mul_broadcast_shape(lazy_tensor.batch_shape, rhs.shape[:-2])
    preconditioner, preconditioner_lt, _ = lazy_tensor._preconditioner()

    def sqrt_precond_matmul(rhs):
        if preconditioner_lt is not None:
            solves, weights, _, _ = contour_integral_quad(preconditioner_lt, rhs, inverse=False)
            return (solves * weights).sum(0)
        else:
            return rhs

    # if not inverse:
    rhs = sqrt_precond_matmul(rhs)

    if shifts is None:
        # Determine if init_vecs has extra_dimensions
        num_extra_dims = max(0, rhs.dim() - lazy_tensor.dim())
        lanczos_init = rhs.__getitem__(
            (*([0] * num_extra_dims), Ellipsis, slice(None, None, None), slice(None, 1, None))
        ).expand(*lazy_tensor.shape[:-1], 1)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore", NumericalWarning)  # Supress CG stopping warning
            _, lanczos_mat = linear_cg(
                lambda v: lazy_tensor._matmul(v),
                rhs=lanczos_init,
                n_tridiag=1,
                max_iter=max_lanczos_iter,
                tolerance=1e-5,
                max_tridiag_iter=max_lanczos_iter,
                preconditioner=preconditioner,
            )
            lanczos_mat = lanczos_mat.squeeze(0)  # We have an extra singleton batch dimension from the Lanczos init

        """
        K^{-1/2} b = 2/pi \int_0^\infty (K - t^2 I)^{-1} dt
        We'll approximate this integral as a sum using quadrature
        We'll determine the appropriate values of t, as well as their weights using elliptical integrals
        """

        # Compute an approximate condition number
        # We'll do this with Lanczos
        try:
            if settings.verbose_linalg.on():
                settings.verbose_linalg.logger.debug(f"Running symeig on a matrix of size {lanczos_mat.shape}.")

            approx_eigs = lanczos_mat.symeig()[0]
            if approx_eigs.min() <= 0:
                raise RuntimeError
        except RuntimeError:
            approx_eigs = lazy_tensor.diag()

        max_eig = approx_eigs.max(dim=-1)[0]
        min_eig = approx_eigs.min(dim=-1)[0]
        k2 = min_eig / max_eig

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
        shifts.sub_(shift_offset)

        # Make sure we have the right shape
        if k2.shape != output_batch_shape:
            weights = torch.stack([w.expand(*output_batch_shape, 1, 1) for w in weights], 0)
            shifts = torch.stack([s.expand(output_batch_shape) for s in shifts], 0)

    # Compute the solves at the given shifts
    # Do one more matmul if we don't want to include the inverse
    with torch.no_grad():
        solves = minres(lambda v: lazy_tensor._matmul(v), rhs, value=-1, shifts=shifts, preconditioner=preconditioner)
    no_shift_solves = solves[0]
    solves = solves[1:]
    if not inverse:
        solves = lazy_tensor._matmul(solves)

    return solves, weights, no_shift_solves, shifts
