#!/usr/bin/env python3

from . import broadcasting, cholesky, grid, interpolation, lanczos, pivoted_cholesky, quadrature, sparse, warnings
from .contour_integral_quad import contour_integral_quad
from .linear_cg import linear_cg
from .memoize import cached
from .minres import minres
from .pinverse import stable_pinverse
from .qr import stable_qr
from .sparse_chol_inv import lr_plus_sparse_precond_closure, sparse_chol_inv
from .stochastic_lq import StochasticLQ


def prod(items):
    """
    """
    if len(items):
        res = items[0]
        for item in items[1:]:
            res = res * item
        return res
    else:
        return 1


__all__ = [
    "broadcasting",
    "cached",
    "contour_integral_quad",
    "linear_cg",
    "StochasticLQ",
    "cholesky",
    "grid",
    "interpolation",
    "lanczos",
    "lr_plus_sparse_precond_closure",
    "minres",
    "pinverse",
    "pivoted_cholesky",
    "prod",
    "quadrature",
    "sparse",
    "sparse_chol_inv",
    "stable_pinverse",
    "stable_qr",
    "warnings",
]
