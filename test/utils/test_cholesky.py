#!/usr/bin/env python3

import unittest
import warnings

import torch

from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NanError
from gpytorch.utils.warnings import NumericalWarning


class TestPSDSafeCholesky(BaseTestCase, unittest.TestCase):
    seed = 0

    def _gen_test_psd(self):
        return torch.tensor([[[0.25, -0.75], [-0.75, 2.25]], [[1.0, 1.0], [1.0, 1.0]]])

    def test_psd_safe_cholesky_nan(self, cuda=False):
        A = self._gen_test_psd().sqrt()
        with self.assertRaises(NanError) as ctx:
            psd_safe_cholesky(A)
            self.assertTrue("NaN" in ctx.exception)

    def test_psd_safe_cholesky_pd(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            for batch_mode in (False, True):
                if batch_mode:
                    A = self._gen_test_psd().to(device=device, dtype=dtype)
                    D = torch.eye(2).type_as(A).unsqueeze(0).repeat(2, 1, 1)
                else:
                    A = self._gen_test_psd()[0].to(device=device, dtype=dtype)
                    D = torch.eye(2).type_as(A)
                A += D
                # basic
                L = torch.cholesky(A)
                L_safe = psd_safe_cholesky(A)
                self.assertTrue(torch.allclose(L, L_safe))
                # upper
                L = torch.cholesky(A, upper=True)
                L_safe = psd_safe_cholesky(A, upper=True)
                self.assertTrue(torch.allclose(L, L_safe))
                # output tensors
                L = torch.empty_like(A)
                L_safe = torch.empty_like(A)
                torch.cholesky(A, out=L)
                psd_safe_cholesky(A, out=L_safe)
                self.assertTrue(torch.allclose(L, L_safe))
                # output tensors, upper
                torch.cholesky(A, upper=True, out=L)
                psd_safe_cholesky(A, upper=True, out=L_safe)
                self.assertTrue(torch.allclose(L, L_safe))
                # make sure jitter doesn't do anything if p.d.
                L = torch.cholesky(A)
                L_safe = psd_safe_cholesky(A, jitter=1e-2)
                self.assertTrue(torch.allclose(L, L_safe))

    def test_psd_safe_cholesky_pd_cuda(self, cuda=False):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_psd_safe_cholesky_pd(cuda=True)

    def test_psd_safe_cholesky_psd(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            for batch_mode in (False, True):
                if batch_mode:
                    A = self._gen_test_psd().to(device=device, dtype=dtype)
                else:
                    A = self._gen_test_psd()[0].to(device=device, dtype=dtype)
                idx = torch.arange(A.shape[-1], device=A.device)
                # default values
                Aprime = A.clone()
                Aprime[..., idx, idx] += 1e-6 if A.dtype == torch.float32 else 1e-8
                L_exp = torch.cholesky(Aprime)
                with warnings.catch_warnings(record=True) as w:
                    # Makes sure warnings we catch don't cause `-w error` to fail
                    warnings.simplefilter("always", NumericalWarning)

                    L_safe = psd_safe_cholesky(A)
                    self.assertTrue(any(issubclass(w_.category, NumericalWarning) for w_ in w))
                    self.assertTrue(any("A not p.d., added jitter" in str(w_.message) for w_ in w))
                self.assertTrue(torch.allclose(L_exp, L_safe))
                # user-defined value
                Aprime = A.clone()
                Aprime[..., idx, idx] += 1e-2
                L_exp = torch.cholesky(Aprime)
                with warnings.catch_warnings(record=True) as w:
                    # Makes sure warnings we catch don't cause `-w error` to fail
                    warnings.simplefilter("always", NumericalWarning)

                    L_safe = psd_safe_cholesky(A, jitter=1e-2)
                    self.assertTrue(any(issubclass(w_.category, NumericalWarning) for w_ in w))
                    self.assertTrue(any("A not p.d., added jitter" in str(w_.message) for w_ in w))
                self.assertTrue(torch.allclose(L_exp, L_safe))

    def test_psd_safe_cholesky_psd_cuda(self, cuda=False):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_psd_safe_cholesky_psd(cuda=True)


if __name__ == "__main__":
    unittest.main()
