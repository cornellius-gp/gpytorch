#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels.keops import MaternKernel
from gpytorch.kernels import MaternKernel as GMaternKernel

try:
    import pykeops  # noqa

    class TestMaternKeOpsKernel(unittest.TestCase):
        def test_forward_nu25_x1_eq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()

            kern1 = MaternKernel(nu=2.5).cuda()
            kern2 = GMaternKernel(nu=2.5).cuda()

            k1 = kern1(x1, x1).evaluate()
            k2 = kern2(x1, x1).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

        def test_forward_nu25_x1_neq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()
            x2 = torch.randn(50, 3).cuda()

            kern1 = MaternKernel(nu=2.5).cuda()
            kern2 = GMaternKernel(nu=2.5).cuda()

            k1 = kern1(x1, x2).evaluate()
            k2 = kern2(x1, x2).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

        def test_forward_nu15_x1_eq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()

            kern1 = MaternKernel(nu=2.5).cuda()
            kern2 = GMaternKernel(nu=2.5).cuda()

            k1 = kern1(x1, x1).evaluate()
            k2 = kern2(x1, x1).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

        def test_forward_nu15_x1_neq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()
            x2 = torch.randn(50, 3).cuda()

            kern1 = MaternKernel(nu=1.5).cuda()
            kern2 = GMaternKernel(nu=1.5).cuda()

            k1 = kern1(x1, x2).evaluate()
            k2 = kern2(x1, x2).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

        def test_forward_nu05_x1_eq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()

            kern1 = MaternKernel(nu=0.5).cuda()
            kern2 = GMaternKernel(nu=0.5).cuda()

            k1 = kern1(x1, x1).evaluate()
            k2 = kern2(x1, x1).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

        def test_forward_nu05_x1_neq_x2(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()
            x2 = torch.randn(50, 3).cuda()

            kern1 = MaternKernel(nu=0.5).cuda()
            kern2 = GMaternKernel(nu=0.5).cuda()

            k1 = kern1(x1, x2).evaluate()
            k2 = kern2(x1, x2).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-5)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
