from .add_diag import AddDiag
from .dsmm import DSMM
from .normal_cdf import NormalCDF
from .log_normal_cdf import LogNormalCDF


__all__ = [
    AddDiag,
    DSMM,
    NormalCDF,
    LogNormalCDF,
]

fastest = True
use_toeplitz = True
max_lanczos_iterations = 100
num_samples = 10
num_trace_samples = 10
max_cg_iterations = 15
