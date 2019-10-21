#!/usr/bin/env python3

import torch
import unittest
import numpy as np
import gpytorch

from gpytorch.lazy import NonLazyTensor, KroneckerProductPlusDiagLazyTensor
from gpytorch.lazy.kronecker_product_added_diag_lazy_tensor import \
                        _DiagKroneckerProdLazyTensor, _KroneckerProductLazyLogDet

# TODO: write unit test for the three classes - lazylogdet is here

class TestKroneckerLazyLogDet(unittest.TestCase):
    def test_logdet_and_solve(self):
        with torch.no_grad():
            kernel = gpytorch.kernels.RBFKernel()
            mat1 = kernel(torch.linspace(0, 1, 5)).evaluate()
            mat2 = kernel(torch.linspace(0, 3, 5)).evaluate()

        rhs = torch.randn(25)

        eval_numpy = np.kron(mat1.numpy(), mat2.numpy())

        eval_sgp = _KroneckerProductLazyLogDet(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

        #gp_lt = gpytorch.lazy.KroneckerProductLazyTensor(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

        self.assertEqual(np.linalg.norm(eval_numpy - eval_sgp.evaluate().numpy()), 0.0)
        
        solve_numpy = np.linalg.solve(eval_numpy, rhs)
        solve_sgp = eval_sgp.inv_matmul(rhs)
        #solve_gpy = torch.solve(rhs.unsqueeze(-1), gp_lt.evaluate())[0].squeeze()
        
        self.assertLess(np.linalg.norm(solve_numpy - solve_sgp.numpy()) / np.linalg.norm(solve_numpy), 0.02)

# class TestKroneckerAddedDiag(unittest.TestCase):


if __name__ == "__main__":
    unittest.main()