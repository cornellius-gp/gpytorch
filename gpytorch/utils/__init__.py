#!/usr/bin/env python3

from .memoize import cached
from .linear_cg import linear_cg
from .stochastic_lq import StochasticLQ
from . import batch
from . import broadcasting
from . import cholesky
from . import fft
from . import eig
from . import grid
from . import interpolation
from . import lanczos
from . import pivoted_cholesky
from . import quadrature
from . import sparse


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
    "batch",
    "broadcasting",
    "cached",
    "eig",
    "cholesky",
    "fft",
    "grid",
    "interpolation",
    "lanczos",
    "linear_cg",
    "pivoted_cholesky",
    "quadrature",
    "StochasticLQ",
    "sparse",
]
