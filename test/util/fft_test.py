import torch
import numpy as np

from gpytorch.utils import fft


def test_fft1_computes_fft_of_1d_input():
    d = 8
    input = torch.randn(d)

    res = fft.fft1(input)
    actual = np.fft.fft(input.numpy())
    assert(tuple(res.size()) == (5, 2))

    res_real = res[:, 0]
    res_imag = res[:, 1]
    actual_real = torch.from_numpy(actual.real[:5]).float()
    actual_imag = torch.from_numpy(actual.imag[:5]).float()

    assert torch.norm(res_real - actual_real) < 1e-5
    assert torch.norm(res_imag - actual_imag) < 1e-5


def test_fft1_computes_fft_of_nd_input():
    d = 8
    input = torch.randn(3, 6, d)

    res = fft.fft1(input)
    actual = np.fft.fft(input.numpy())
    assert(tuple(res.size()) == (3, 6, 5, 2))

    res_real = res[:, :, :, 0]
    res_imag = res[:, :, :, 1]
    actual_real = torch.from_numpy(actual.real[:, :, :5]).float()
    actual_imag = torch.from_numpy(actual.imag[:, :, :5]).float()

    assert torch.norm(res_real - actual_real) < 1e-5
    assert torch.norm(res_imag - actual_imag) < 1e-5


def test_fft1_returns_type_of_original_input():
    d = 8
    input = torch.randn(3, 6, d).double()

    res = fft.fft1(input)
    assert isinstance(res, torch.DoubleTensor)


def test_ifft1_computes_ifft_of_1d_input():
    d = 8
    input = torch.randn(d)

    res = fft.fft1(input)
    recon = fft.ifft1(res)
    assert input.size() == recon.size()
    assert torch.norm(input - recon) < 1e-5


def test_ifft1_computes_ifft_of_1d_input_with_odd_size():
    d = 9
    input = torch.randn(d)

    res = fft.fft1(input)
    recon = fft.ifft1(res, input.size())
    assert input.size() == recon.size()
    assert torch.norm(input - recon) < 1e-5


def test_ifft1_computes_ifft_of_2d_input():
    d = 8
    input = torch.randn(6, d)

    res = fft.fft1(input)
    recon = fft.ifft1(res)
    assert input.size() == recon.size()
    assert torch.norm(input - recon) < 1e-5


def test_ifft1_returns_type_of_original_input():
    d = 8
    input = torch.randn(6, d)
    res = fft.fft1(input).double()
    recon = fft.ifft1(res)
    assert input.size() == recon.size()
    assert torch.norm(input.double() - recon) < 1e-5
    assert isinstance(res, torch.DoubleTensor)
