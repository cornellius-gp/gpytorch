from scipy.special import ellipk, ellipj
from .lanczos import lanczos_tridiag
import torch
import numpy as np
import math


def solve_shifted_systems(lanczos_basis, lanczos_mat, rhs, shifts):
    """
    Solves several systems of the form:

        (s_j I - A)x = rhs

    with shifts s_1,...,s_K, and where A is at least symmetric.

    To do this efficiently, we exploit the fact that any matrix (s_j I - A) has the same
    Krylov subspace as A. Therefore, we run Lanczos on A to get Q and T,
    and observe that s_j I_k - T is (s_j I - A) projected in to the Krylov subspace.

    Therefore, the jth solve is given by:

        x*_j = ||rhs||_{2}^{2} * Q(s_j I_k - T)^{-1}e_1

    Args:
        - Q (torch.Tensor): Q matrix from running Lanczos on (A, rhs)
        - T (torch.Tensor): T matrix from running Lanczos on (A, rhs)
        - rhs (torch.Tensor): rhs in system (s_j I - A)x = rhs
        - shifts (torch.Tensor): 1D tensor of shifts s_1, ..., s_K
    """
    shifts_batch = shifts.view(shifts.numel(), *([1] * (lanczos_mat.dim())))
    I_mats = shifts_batch * torch.eye(
        *lanczos_mat.shape[-2:],
        device=lanczos_mat.device,
        dtype=lanczos_mat.dtype
    ).repeat(*lanczos_mat.shape[:-2], 1, 1)

    shifted_mats = I_mats - lanczos_mat.unsqueeze(0)

    e_1 = torch.zeros(lanczos_mat.size(-1), 1, device=lanczos_mat.device, dtype=lanczos_mat.dtype)
    e_1[0] = 1

    krylov_solves = torch.gesv(e_1, shifted_mats)[0]

    norms = rhs.norm(dim=-2)
    if norms.numel() == 1:
        real_solves = norms.item() * lanczos_basis.matmul(krylov_solves)
    else:
        norms = norms.view(1, norms.numel(), 1, 1)
        real_solves = norms * lanczos_basis.matmul(krylov_solves)
    return real_solves


def sqrt_matmul(lazy_tensor, rhs, max_lanczos_iter=50, num_quad_samples=15):
    """
    Performs A^{1/2} rhs using contour integral quadrature.

    Args:
        - lazy_tensor (gpytorch.lazy.LazyTensor): LazyTensor representing A
        - rhs (torch.Tensor): rhs to multiply with
        - min_eig_est (torch.Tensor): Estimated minimum eigenvalue

    Returns:
        - A very good approximation to A^{1/2} rhs
    """
    lanczos_basis, lanczos_mat = lanczos_tridiag(
        lambda v: lazy_tensor._matmul(v),
        init_vecs=rhs,
        dtype=rhs.dtype,
        device=rhs.device,
        matrix_shape=lazy_tensor.shape,
        max_iter=max_lanczos_iter,
    )

    # We need to run Lanczos anyways
    if lanczos_mat.dim() > 2:
        ind = [0] * (lanczos_mat.dim() - 2)
        approx_eigs = lanczos_mat.__getitem__(*ind).symeig()[0]
    else:
        approx_eigs = lanczos_mat.symeig()[0]
    approx_eigs = approx_eigs[approx_eigs > 0]
    min_eig = approx_eigs.min()
    max_eig = approx_eigs.max()

    k2 = min_eig / max_eig
    Kp = ellipk(1 - k2)  # Elliptical integral of the first kind
    N = 15
    t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
    sn, cn, dn, _ = ellipj(np.imag(t), 1 - k2.item())  # Jacobi elliptic functions
    cn = 1. / cn
    dn = dn * cn
    sn = 1j * sn * cn
    w = np.sqrt(min_eig.item()) * sn
    dzdt = cn * dn
    w_pow2 = np.real(np.power(w, 2))
    solves = solve_shifted_systems(
        lanczos_basis,
        lanczos_mat,
        rhs,
        torch.from_numpy(w_pow2.astype(float)).type_as(rhs).to(rhs.device),
    )

    dzdt_th = torch.from_numpy(dzdt).type_as(solves)
    dzdt_th = dzdt_th.view(dzdt_th.numel(), *([1] * (solves.dim() - 1)))
    weighted_solves = dzdt_th * solves
    summed_solves = weighted_solves.sum(0)
    res = lazy_tensor._matmul(summed_solves)
    constant = -2 * Kp * np.sqrt(min_eig) / (math.pi * N)
    res = constant * res
    return res.squeeze(-1)


def solve_shifted_systems_fast_approx(lanczos_basis, lanczos_mat, rhs, shifts):
    """
    Same as solve_shifted_systems, but use the same Lanczos decomposition for each rhs.
    """
    shifts_batch = shifts.view(shifts.numel(), *([1] * (lanczos_mat.dim())))
    I_mats = shifts_batch * torch.eye(
        *lanczos_mat.shape[-2:],
        device=lanczos_mat.device,
        dtype=lanczos_mat.dtype
    ).repeat(*lanczos_mat.shape[:-2], 1, 1)

    shifted_mats = I_mats - lanczos_mat.unsqueeze(0)

    proj_rhs = lanczos_basis.transpose(-2, -1).matmul(rhs)

    krylov_solves = torch.gesv(proj_rhs, shifted_mats)[0]

    real_solves = lanczos_basis.matmul(krylov_solves)
    return real_solves


def sqrt_matmul_fast_approx(lazy_tensor, rhs, max_lanczos_iter=50, num_quad_samples=15):
    """
    Same as sqrt_matmul, but using only one Lanczos decomposition rather than one per rhs vector.
    """
    lanczos_vec = rhs[..., 0].unsqueeze(-1)
    lanczos_basis, lanczos_mat = lanczos_tridiag(
        lambda v: lazy_tensor._matmul(v),
        init_vecs=lanczos_vec,
        dtype=rhs.dtype,
        device=rhs.device,
        matrix_shape=lazy_tensor.shape,
        max_iter=max_lanczos_iter,
    )

    approx_eigs = lanczos_mat.symeig()[0]
    approx_eigs = approx_eigs[approx_eigs > 0]
    min_eig = approx_eigs.min()
    max_eig = approx_eigs.max()

    k2 = min_eig / max_eig
    Kp = ellipk(1 - k2)  # Elliptical integral of the first kind
    N = 15
    t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
    sn, cn, dn, _ = ellipj(np.imag(t), 1 - k2.item())  # Jacobi elliptic functions
    cn = 1. / cn
    dn = dn * cn
    sn = 1j * sn * cn
    w = np.sqrt(min_eig.item()) * sn
    dzdt = cn * dn
    w_pow2 = np.real(np.power(w, 2))
    solves = solve_shifted_systems_fast_approx(
        lanczos_basis,
        lanczos_mat,
        rhs,
        torch.from_numpy(w_pow2.astype(float)).type_as(rhs).to(rhs.device),
    )

    dzdt_th = torch.from_numpy(dzdt).type_as(solves)
    dzdt_th = dzdt_th.view(dzdt_th.numel(), *([1] * (solves.dim() - 1)))
    weighted_solves = dzdt_th * solves
    summed_solves = weighted_solves.sum(0)
    res = lazy_tensor._matmul(summed_solves)
    constant = -2 * Kp * np.sqrt(min_eig) / (math.pi * N)
    res = constant * res
    return res.squeeze(-1).transpose(-2, -1)
