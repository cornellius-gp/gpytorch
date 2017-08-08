import torch
from torch.autograd import Function
from gpytorch.utils import fft
from gpytorch import utils


class ToeplitzMV(Function):
    """
    Performs Toeplitz matrix-vector multiplication
    Args:
        - c (vector n) - first column of Toeplitz matrix
        - r (vector n-1) - first row of Toeplitz matrix
        - v (vector n) - vector for multiplication
    Returns:
        - Vector (n)
    """
    def forward(self, c, r, v):
        self.save_for_backward(c, r, v)
        return utils.toeplitz.toeplitz_mv(c, r, v)

    def backward(self, grad_output):
        c, r, v = self.saved_tensors
        di_dc = utils.rcumsum(v * grad_output)
        di_dr = di_dc.clone()
        di_dv = utils.toeplitz.toeplitz_mv(r, c, grad_output)

        return di_dc, di_dr, di_dv