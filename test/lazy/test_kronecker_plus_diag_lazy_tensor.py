#!/usr/bin/env python3

import torch
import unittest
import numpy as np
import gpytorch

from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase, RectangularLazyTensorTestCase
from gpytorch.lazy.lazy_kron_add_diag import KroneckerProductPlusDiagLazyTensor, \
                        _DiagKroneckerProdLazyTensor, _KroneckerProductLazyLogDet

# TODO: write unit test for the three classes - lazylogdet is here

def __main__():
    with torch.no_grad():
        kernel = gpytorch.kernels.RBFKernel()
        mat1 = kernel(torch.linspace(0, 1, 5)).evaluate()
        mat2 = kernel(torch.linspace(0, 3, 5)).evaluate()

    rhs = torch.randn(25)

    eval_numpy = np.kron(mat1.numpy(), mat2.numpy())

    eval_sgp = _KroneckerProductLazyLogDet(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

    gp_lt = gpytorch.lazy.KroneckerProductLazyTensor(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

    print('Evaluation error: ', np.linalg.norm(eval_numpy - eval_sgp.evaluate().numpy()))

    solve_numpy = np.linalg.solve(eval_numpy, rhs)
    solve_sgp = eval_sgp.inv_matmul(rhs)
    solve_gpy = torch.solve(rhs.unsqueeze(-1), gp_lt.evaluate())[0].squeeze()
    
    #print(solve_numpy, solve_sgp.numpy())
    print('Solve error: ', np.linalg.norm(solve_numpy - solve_sgp.numpy()) / np.linalg.norm(solve_numpy))
    print('Gpytorch solve error: ', torch.norm(solve_gpy- solve_sgp) / np.linalg.norm(solve_numpy))
