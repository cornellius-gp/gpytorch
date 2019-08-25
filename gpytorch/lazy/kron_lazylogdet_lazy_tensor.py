#!/usr/bin/env python3

import torch

from . import KroneckerProductLazyTensor, DiagLazyTensor, NonLazyTensor

class KroneckerProductLazyLogDet(KroneckerProductLazyTensor):
    def __init__(self, *lazy_tensors, jitter = 0):
        super(KroneckerProductLazyLogDet, self).__init__(*lazy_tensors)
        # on initialization take the eigenvectors & eigenvalues of all of the lazy tensors
        self.eig_cache = [torch.eig(lt.evaluate(), eigenvectors = True) for lt in self.lazy_tensors]

    def inv_matmul(self, rhs):
        Vinv = KroneckerProductLazyTensor(*[DiagLazyTensor(1 / (s[0][:,0].abs()+jitter) ) for s in self.eig_cache])
        Q = KroneckerProductLazyTensor(*[NonLazyTensor(s[1]) for s in self.eig_cache])

        # first compute Q^T y
        res1 = Q.t().matmul(rhs)

        # now V^{-1} Q^T y
        res2 = Vinv.matmul(res1)
        res3 = Q.matmul(res2)

        return res3

    def logdet(self):
        lt_sizes = [lt.size(-1) for lt in self.lazy_tensors]

        # det(A \kron B) = det(A)^m det(B)^n where m,n are the sizes of A,B
        scaled_logdets = [m * s[0].sum() for m, s in zip(lt_sizes, self.eig_cache)]

        full_logdet = 0.
        for logdet in scaled_logdets:
            full_logdet = logdet + full_logdet
        
        return full_logdet

#TODO: write this as a unit test
def __main__():
    import gpytorch
    import numpy as np

    with torch.no_grad():
        kernel = gpytorch.kernels.RBFKernel()
        mat1 = kernel(torch.linspace(0, 1, 5)).evaluate()
        mat2 = kernel(torch.linspace(0, 3, 5)).evaluate()

    rhs = torch.randn(25)

    eval_numpy = np.kron(mat1.numpy(), mat2.numpy())

    eval_sgp = KronLazyLogDet(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

    gp_lt = gpytorch.lazy.KroneckerProductLazyTensor(gpytorch.lazy.NonLazyTensor(mat1), gpytorch.lazy.NonLazyTensor(mat2))

    print('Evaluation error: ', np.linalg.norm(eval_numpy - eval_sgp.evaluate().numpy()))

    solve_numpy = np.linalg.solve(eval_numpy, rhs)
    solve_sgp = eval_sgp.inv_matmul(rhs)
    solve_gpy = torch.solve(rhs.unsqueeze(-1), gp_lt.evaluate())[0].squeeze()
    
    #print(solve_numpy, solve_sgp.numpy())
    print('Solve error: ', np.linalg.norm(solve_numpy - solve_sgp.numpy()) / np.linalg.norm(solve_numpy))
    print('Gpytorch solve error: ', torch.norm(solve_gpy- solve_sgp) / np.linalg.norm(solve_numpy))
