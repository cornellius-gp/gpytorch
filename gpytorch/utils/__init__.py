from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .linear_cg import linear_cg
from .stochastic_lq import StochasticLQ
from . import cholesky
from . import eig
from . import fft
from . import grid
from . import interpolation
from . import lanczos
from . import pivoted_cholesky
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
    "linear_cg",
    "StochasticLQ",
    "cholesky",
    "eig",
    "fft",
    "grid",
    "interpolation",
    "lanczos",
    "pivoted_cholesky",
    "sparse",
]
