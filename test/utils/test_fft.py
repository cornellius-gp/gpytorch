#!/usr/bin/env python3

import torch
import unittest
import numpy as np

from gpytorch.utils import fft


class TestFFT(unittest.TestCase):
    def test_fft1_computes_fft_of_1d_input(self):
        d = 8
        input = torch.randn(d)

        res = fft.fft1(input)
        actual = np.fft.fft(input.numpy())
        self.assertEqual(tuple(res.size()), (8, 2))

        res_real = res[:, 0]
        res_imag = res[:, 1]
        actual_real = torch.from_numpy(actual.real).float()
        actual_imag = torch.from_numpy(actual.imag).float()

        assert torch.norm(res_real - actual_real) < 1e-5
        assert torch.norm(res_imag - actual_imag) < 1e-5

    def test_fft1_computes_fft_of_nd_input(self):
        d = 8
        input = torch.randn(3, 6, d)

        res = fft.fft1(input)
        actual = np.fft.fft(input.numpy())
        self.assertEqual(tuple(res.size()), (3, 6, 8, 2))

        res_real = res[:, :, :, 0]
        res_imag = res[:, :, :, 1]
        actual_real = torch.from_numpy(actual.real[:, :, :]).float()
        actual_imag = torch.from_numpy(actual.imag[:, :, :]).float()

        self.assertLess(torch.norm(res_real - actual_real), 1e-5)
        self.assertLess(torch.norm(res_imag - actual_imag), 1e-5)

    def test_fft1_returns_type_of_original_input(self):
        d = 8
        input = torch.randn(3, 6, d).double()

        res = fft.fft1(input)
        self.assertTrue(isinstance(res, torch.DoubleTensor))

    def test_ifft1_computes_ifft_of_1d_input(self):
        d = 8
        input = torch.randn(d)

        res = fft.fft1(input)
        recon = fft.ifft1(res)
        self.assertEqual(input.size(), recon.size())
        self.assertLess(torch.norm(input - recon), 1e-5)

    def test_ifft1_computes_ifft_of_1d_input_with_odd_size(self):
        d = 9
        input = torch.randn(d)

        res = fft.fft1(input)
        recon = fft.ifft1(res)
        self.assertEqual(input.size(), recon.size())
        self.assertLess(torch.norm(input - recon), 1e-5)

    def test_ifft1_computes_ifft_of_2d_input(self):
        d = 8
        input = torch.randn(6, d)

        res = fft.fft1(input)
        recon = fft.ifft1(res)
        self.assertEqual(input.size(), recon.size())
        self.assertLess(torch.norm(input - recon), 1e-5)

    def test_ifft1_returns_type_of_original_input(self):
        d = 8
        input = torch.randn(6, d)
        res = fft.fft1(input).double()
        recon = fft.ifft1(res)
        self.assertEqual(input.size(), recon.size())
        self.assertLess(torch.norm(input.double() - recon), 1e-5)
        self.assertTrue(isinstance(res, torch.DoubleTensor))


if __name__ == "__main__":
    unittest.main()
