#!/usr/bin/env python3

from typing import Optional, Tuple, Union

import linear_operator
import torch
from linear_operator import LinearOperator
from torch import Tensor

from . import (
    beta_features,
    distributions,
    kernels,
    lazy,
    likelihoods,
    means,
    metrics,
    mlls,
    models,
    optim,
    priors,
    settings,
    utils,
    variational,
)
from .functions import inv_matmul, log_normal_cdf, logdet, matmul  # Deprecated
from .lazy import cat, delazify, lazify
from .mlls import ExactMarginalLogLikelihood
from .module import Module

Anysor = Union[LinearOperator, Tensor]


def add_diagonal(input: Anysor, diag: Tensor) -> LinearOperator:
    r"""
    Adds an element to the diagonal of the matrix :math:`\mathbf A`.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param diag: Diagonal to add
    :return: :math:`\mathbf A + \text{diag}(\mathbf d)`, where :math:`\mathbf A` is the linear operator
        and :math:`\mathbf d` is the diagonal component
    """
    return linear_operator.add_diagonal(input=input, diag=diag)


def add_jitter(input: Anysor, jitter_val: float = 1e-3) -> Anysor:
    r"""
    Adds jitter (i.e., a small diagonal component) to the matrix this
    LinearOperator represents.
    This is equivalent to calling :meth:`~linear_operator.operators.LinearOperator.add_diagonal`
    with a scalar tensor.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param jitter_val: The diagonal component to add
    :return: :math:`\mathbf A + \alpha (\mathbf I)`, where :math:`\mathbf A` is the linear operator
        and :math:`\alpha` is :attr:`jitter_val`.
    """
    return linear_operator.add_jitter(input=input, jitter_val=jitter_val)


def diagonalization(input: Anysor, method: Optional[str] = None) -> Tuple[Tensor, Tensor]:
    r"""
    Returns a (usually partial) diagonalization of a symmetric positive definite matrix (or batch of matrices).
    :math:`\mathbf A`.
    Options are either "lanczos" or "symeig". "lanczos" runs Lanczos while
    "symeig" runs LinearOperator.symeig.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param method: Specify the method to use ("lanczos" or "symeig"). The method will be determined
        based on size if not specified.
    :return: eigenvalues and eigenvectors representing the diagonalization.
    """
    return linear_operator.diagonalization(input=input, method=method)


def dsmm(
    sparse_mat: Union[torch.sparse.HalfTensor, torch.sparse.FloatTensor, torch.sparse.DoubleTensor],
    dense_mat: Tensor,
) -> Tensor:
    r"""
    Performs the (batch) matrix multiplication :math:`\mathbf{SD}`
    where :math:`\mathbf S` is a sparse matrix and :math:`\mathbf D` is a dense matrix.

    :param sparse_mat: Sparse matrix :math:`\mathbf S` (... x M x N)
    :param dense_mat: Dense matrix :math:`\mathbf D` (... x N x O)
    :return: :math:`\mathbf S \mathbf D` (... x M x N)
    """
    return linear_operator.dsmm(sparse_mat=sparse_mat, dense_mat=dense_mat)


def inv_quad(input: Anysor, inv_quad_rhs: Tensor, reduce_inv_quad: bool = True) -> Tensor:
    r"""
    Computes an inverse quadratic form (w.r.t self) with several right hand sides, i.e:

    .. math::
       \text{tr}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right),

    where :math:`\mathbf A` is a positive definite matrix (or batch of matrices) and :math:`\mathbf R`
    represents the right hand sides (:attr:`inv_quad_rhs`).

    If :attr:`reduce_inv_quad` is set to false (and :attr:`inv_quad_rhs` is supplied),
    the function instead computes

    .. math::
       \text{diag}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right).

    :param input: :math:`\mathbf A` - the positive definite matrix (... X N X N)
    :param inv_quad_rhs: :math:`\mathbf R` - the right hand sides of the inverse quadratic term (... x N x M)
    :param reduce_inv_quad: Whether to compute
        :math:`\text{tr}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right)`
        or :math:`\text{diag}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right)`.
    :returns: The inverse quadratic term.
        If `reduce_inv_quad=True`, the inverse quadratic term is of shape (...). Otherwise, it is (... x M).
    """
    return linear_operator.inv_quad(input=input, inv_quad_rhs=inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)


def inv_quad_logdet(
    input: Anysor, inv_quad_rhs: Optional[Tensor] = None, logdet: bool = False, reduce_inv_quad: bool = True
) -> Tuple[Tensor, Tensor]:
    r"""
    Calls both :func:`inv_quad_logdet` and :func:`logdet` on a positive definite matrix (or batch) :math:`\mathbf A`.
    However, calling this method is far more efficient and stable than calling each method independently.

    :param input: :math:`\mathbf A` - the positive definite matrix (... X N X N)
    :param inv_quad_rhs: :math:`\mathbf R` - the right hand sides of the inverse quadratic term (... x N x M)
    :param logdet: Whether or not to compute the
        logdet term :math:`\log \vert \mathbf A \vert`.
    :param reduce_inv_quad: Whether to compute
        :math:`\text{tr}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right)`
        or :math:`\text{diag}\left( \mathbf R^\top \mathbf A^{-1} \mathbf R \right)`.
    :returns: The inverse quadratic term (or None), and the logdet term (or None).
        If `reduce_inv_quad=True`, the inverse quadratic term is of shape (...). Otherwise, it is (... x M).
    """
    return linear_operator.inv_quad_logdet(
        input=input, inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad
    )


def pivoted_cholesky(
    input: Anysor, rank: int, error_tol: Optional[float] = None, return_pivots: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""
    Performs a partial pivoted Cholesky factorization of a positive definite matrix (or batch of matrices).
    :math:`\mathbf L \mathbf L^\top = \mathbf A`.
    The partial pivoted Cholesky factor :math:`\mathbf L \in \mathbb R^{N \times \text{rank}}`
    forms a low rank approximation to the LinearOperator.

    The pivots are selected greedily, correspoading to the maximum diagonal element in the
    residual after each Cholesky iteration. See `Harbrecht et al., 2012`_.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param rank: The size of the partial pivoted Cholesky factor.
    :param error_tol: Defines an optional stopping criterion.
        If the residual of the factorization is less than :attr:`error_tol`, then the
        factorization will exit early. This will result in a :math:`\leq \text{ rank}` factor.
    :param return_pivots: Whether or not to return the pivots alongside
        the partial pivoted Cholesky factor.
    :return: The `... x N x rank` factor (and optionally the `... x N` pivots if :attr:`return_pivots` is True).

    .. _Harbrecht et al., 2012:
        https://www.sciencedirect.com/science/article/pii/S0168927411001814
    """
    return linear_operator.pivoted_cholesky(input=input, rank=rank, return_pivots=return_pivots)


def root_decomposition(input: Anysor, method: Optional[str] = None) -> LinearOperator:
    r"""
    Returns a (usually low-rank) root decomposition linear operator of the
    positive definite matrix (or batch of matrices) :math:`\mathbf A`.
    This can be used for sampling from a Gaussian distribution, or for obtaining a
    low-rank version of a matrix.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param method: Which method to use to perform the root decomposition. Choices are:
        "cholesky", "lanczos", "symeig", "pivoted_cholesky", or "svd".
    :return: A tensor :math:`\mathbf R` such that :math:`\mathbf R \mathbf R^\top \approx \mathbf A`.
    """
    return linear_operator.root_decomposition(input=input, method=method)


def root_inv_decomposition(
    input: Anysor,
    initial_vectors: Optional[Tensor] = None,
    test_vectors: Optional[Tensor] = None,
    method: Optional[str] = None,
) -> LinearOperator:
    r"""
    Returns a (usually low-rank) inverse root decomposition linear operator
    of the PSD LinearOperator :math:`\mathbf A`.
    This can be used for sampling from a Gaussian distribution, or for obtaining a
    low-rank version of a matrix.

    The root_inv_decomposition is performed using a partial Lanczos tridiagonalization.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param initial_vectors: Vectors used to initialize the Lanczos decomposition.
        The best initialization vector (determined by :attr:`test_vectors`) will be chosen.
    :param test_vectors: Vectors used to test the accuracy of the decomposition.
    :param method: Root decomposition method to use (symeig, diagonalization, lanczos, or cholesky).
    :return: A tensor :math:`\mathbf R` such that :math:`\mathbf R \mathbf R^\top \approx \mathbf A^{-1}`.
    """
    return linear_operator.root_inv_decomposition(
        input=input, initial_vectors=initial_vectors, test_vectors=test_vectors, method=method
    )


def solve(input: Anysor, rhs: Tensor, lhs: Optional[Tensor] = None) -> Tensor:
    r"""
    Given a positive definite matrix (or batch of matrices) :math:`\mathbf A`,
    computes a linear solve with right hand side :math:`\mathbf R`:

    .. math::
       \begin{equation}
           \mathbf A^{-1} \mathbf R,
       \end{equation}

    where :math:`\mathbf R` is :attr:`right_tensor` and :math:`\mathbf A` is the LinearOperator.

    .. note::
        Unlike :func:`torch.linalg.solve`, this function can take an optional :attr:`left_tensor` attribute.
        If this is supplied :func:`gpytorch.solve` computes

        .. math::
           \begin{equation}
               \mathbf L \mathbf A^{-1} \mathbf R,
           \end{equation}

        where :math:`\mathbf L` is :attr:`left_tensor`.
        Supplying this can reduce the number of solver calls required in the backward pass.

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param rhs: :math:`\mathbf R` - the right hand side
    :param lhs: :math:`\mathbf L` - the left hand side
    :return: :math:`\mathbf A^{-1} \mathbf R` or :math:`\mathbf L \mathbf A^{-1} \mathbf R`.
    """
    return linear_operator.solve(input=input, rhs=rhs, lhs=lhs)


def sqrt_inv_matmul(input: Anysor, rhs: Tensor, lhs: Optional[Tensor] = None) -> Tensor:
    r"""
    Given a positive definite matrix (or batch of matrices) :math:`\mathbf A`
    and a right hand size :math:`\mathbf R`,
    computes

    .. math::
       \begin{equation}
           \mathbf A^{-1/2} \mathbf R,
       \end{equation}

    If :attr:`lhs` is supplied, computes

    .. math::
       \begin{equation}
           \mathbf L \mathbf A^{-1/2} \mathbf R,
       \end{equation}

    where :math:`\mathbf L` is :attr:`lhs`.
    (Supplying :attr:`lhs` can reduce the number of solver calls required in the backward pass.)

    :param input: The matrix (or batch of matrices) :math:`\mathbf A` (... x N x N).
    :param rhs: :math:`\mathbf R` - the right hand side
    :param lhs: :math:`\mathbf L` - the left hand side
    :return: :math:`\mathbf A^{-1/2} \mathbf R` or :math:`\mathbf L \mathbf A^{-1/2} \mathbf R`.
    """
    return linear_operator.sqrt_inv_matmul(input=input, rhs=rhs, lhs=lhs)


# Read version number as written by setuptools_scm
try:
    from gpytorch.version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "Unknown"  # pragma: no cover


__all__ = [
    # Submodules
    "distributions",
    "kernels",
    "lazy",
    "likelihoods",
    "means",
    "metrics",
    "mlls",
    "models",
    "optim",
    "priors",
    "utils",
    "variational",
    # Classes
    "Module",
    "ExactMarginalLogLikelihood",
    # Functions
    "add_diagonal",
    "add_jitter",
    "dsmm",
    "inv_quad",
    "inv_quad_logdet",
    "pivoted_cholesky",
    "root_decomposition",
    "root_inv_decomposition",
    "solve",
    "sqrt_inv_matmul",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
    # Deprecated
    "add_diag",
    "cat",
    "delazify",
    "inv_matmul",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
]
