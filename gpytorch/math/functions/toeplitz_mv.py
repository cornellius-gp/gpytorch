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
        if c.ndimension() != 1 or r.ndimension() != 1 or v.ndimension() != 1:
            raise RuntimeError('All inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                                matrix plus the target vector v).')

        if len(c) != len(r):
            raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

        if len(c) != len(v):
            raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a length \
                                {} vector.'.format(len(c), len(c), len(v)))

        if c[0] != r[0]:
            raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first \
                                otherwise the value of T[0,0] is ambiguous. \
                                Got: c[0]={} and r[0]={}'.format(c[0], r[0]))

        if type(c) != type(r) or type(c) != type(v):
            raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

        self.save_for_backward(c, r, v)
        return self._mv(c, r, v)

    def backward(self, grad_output):
        c, r, v = self.saved_tensors
        di_dc = utils.rcumsum(v * grad_output)
        di_dr = di_dc.clone()
        di_dv = self._mv(r, c, grad_output)

        return di_dc, di_dr, di_dv

    def _mv(self, c, r, v):
        orig_size = len(c)
        r_reverse = utils.reverse(r[1:])
        c.resize_(orig_size + len(r_reverse))
        c[orig_size:].copy_(r_reverse)

        v.resize_(2 * orig_size - 1)
        v[orig_size:].fill_(0)

        fft_c = fft.fft1(c)
        fft_v = fft.fft1(v)
        fft_product = torch.zeros(fft_c.size())

        fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
        fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
        fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
        fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

        res = fft.ifft1(fft_product, c.size())
        c.resize_(orig_size)
        r.resize_(orig_size)
        v.resize_(orig_size)
        res.resize_(orig_size)
        return res
