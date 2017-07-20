import torch
from torch.autograd import Function
from gpytorch.utils import fft
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
        if c.ndimension() != 1 or r.ndimension() != 1 or M.ndimension() != 2:
            raise RuntimeError('The first two inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                                matrix), and the last input should be a matrix.')

        if len(c) != len(r):
            raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

        if len(c) != len(M):
            raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a matrix with leading \
                                dimension {}.'.format(len(c),len(c),len(v)))

        if c[0] != r[0]:
            raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first element, \
                                otherwise the value of T[0,0] is ambiguous. Got: c[0]={} and r[0]={}'.format(c[0], r[0]))

        if type(c) != type(r) or type(c) != type(M):
            raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

        self.save_for_backward(c, r, M)
        return self._mm(c, r, M)

    def backward(self, grad_output):
        c, r, M = self.saved_tensors
        di_dc = utils.rcumsum(utils.rcumsum(M * grad_output, 1)[:, 0])
        di_dr = di_dc.clone()
        di_dv = self._mm(r, c, grad_output)

        return di_dc, di_dr, di_dv

    def _mm(self, c, r, M):
        _, num_rhs = M.size()
        orig_size = len(c)
        r_reverse = utils.reverse(r[1:])
        c.resize_(orig_size + len(r_reverse))
        c[orig_size:].copy_(r_reverse)

        M.resize_(2 * orig_size - 1, num_rhs)
        M[orig_size:, :].fill_(0)

        fft_M = fft.fft1(M.t().contiguous())
        fft_c = fft.fft1(c).expand_as(fft_M)
        fft_product = torch.zeros(fft_M.size())

        fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
        fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
        fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
        fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

        res = fft.ifft1(fft_product, (num_rhs, 2 * orig_size - 1)).t()
        c.resize_(orig_size)
        r.resize_(orig_size)
        M.resize_(orig_size, num_rhs)
        res = res[:orig_size, :]
        return res
