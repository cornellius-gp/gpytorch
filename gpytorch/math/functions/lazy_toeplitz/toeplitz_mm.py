from torch.autograd import Function
from gpytorch import utils


class ToeplitzMM(Function):
    """
    Performs Toeplitz matrix-matrix multiplication
    Args:
        - c (vector n) - first column of Toeplitz matrix
        - r (vector n) - first row of Toeplitz matrix
        - M (matrix nxd) - matrix for multiplication
    Returns:
        - Vector (n)
    """
    def forward(self, c, r, M):
        self.save_for_backward(c, r, M)
        return utils.toeplitz.toeplitz_mm(c, r, M)

    def backward(self, grad_output):
        c, r, M = self.saved_tensors
        di_dc = utils.rcumsum(utils.rcumsum(M * grad_output, 1)[:, 0])
        di_dr = di_dc.clone()
        di_dv = utils.toeplitz.toeplitz_mm(r, c, grad_output)

        return di_dc, di_dr, di_dv
