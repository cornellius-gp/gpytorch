#!/usr/bin/env python3

from . import broadcasting, cholesky, fft, grid, interpolation, lanczos, pivoted_cholesky, quadrature, sparse, warnings
from .contour_integral_quad import contour_integral_quad
from .linear_cg import linear_cg
from .memoize import cached
from .minres import minres
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
    "fft",
    "grid",
    "interpolation",
    "lanczos",
    "minres",
    "pivoted_cholesky",
    "prod",
    "quadrature",
    "sparse",
    "warnings",
]
