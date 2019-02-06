#!/usr/bin/env python3

import unittest
import warnings
from test._utils import approx_equal

import torch
from gpytorch.utils.cholesky import psd_safe_cholesky, tridiag_batch_potrf, tridiag_batch_potrs


class TestTriDiag(unittest.TestCase):
    def test_potrf(self):
        chol = torch.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 2, 0], [0, 0, 2, 3]], dtype=torch.float).unsqueeze(0)
        trid = chol.matmul(chol.transpose(-1, -2))

        self.assertTrue(torch.equal(chol, tridiag_batch_potrf(trid, upper=False)))

    def test_potrs(self):
        chol = torch.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 2, 0], [0, 0, 2, 3]], dtype=torch.float).unsqueeze(0)

        mat = torch.randn(1, 4, 3)
        self.assertTrue(
            approx_equal(torch.potrs(mat[0], chol[0], upper=False), tridiag_batch_potrs(mat, chol, upper=False)[0])
        )


class TestPSDSafeCholesky(unittest.TestCase):
    def _gen_test_psd(self):
        return torch.tensor([[[0.25, -0.75], [-0.75, 2.25]], [[1.0, 1.0], [1.0, 1.0]]])

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
                Aprime[..., idx, idx] += 1e-5 if A.dtype == torch.float32 else 1e-7
                L_exp = torch.cholesky(Aprime)
                with warnings.catch_warnings(record=True) as ws:
                    L_safe = psd_safe_cholesky(A)
                    self.assertEqual(len(ws), 1)
                    self.assertEqual(ws[-1].category, RuntimeWarning)
                self.assertTrue(torch.allclose(L_exp, L_safe))
                # user-defined value
                Aprime = A.clone()
                Aprime[..., idx, idx] += 1e-2
                L_exp = torch.cholesky(Aprime)
                with warnings.catch_warnings(record=True) as ws:
                    L_safe = psd_safe_cholesky(A, jitter=1e-2)
                    self.assertEqual(len(ws), 1)
                    self.assertEqual(ws[-1].category, RuntimeWarning)
                self.assertTrue(torch.allclose(L_exp, L_safe))

    def test_psd_safe_cholesky_psd_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_psd_safe_cholesky_psd(cuda=True)


if __name__ == "__main__":
    unittest.main()
