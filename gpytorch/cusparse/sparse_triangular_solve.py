import torch
from torch.utils.cpp_extension import load

import gpytorch

import os
path_cusparse = os.path.join(os.path.dirname(gpytorch.__file__), 'cusparse')

cusparse = load(name="cusparse", sources=[os.path.join(path_cusparse, "sparse_triangular_solve.cu")])
from cusparse import sparse_triangular_solve

if __name__ == "__main__":
    indices = torch.tensor([[0, 1, 1, 2, 2, 2, 0], [0, 0, 1, 0, 1, 2, 2]], dtype=torch.long, device='cuda:0')
    values = torch.tensor([1., 2., 3., 4., 5., 6., 0.], dtype=torch.float, device='cuda:0')

    s = torch.sparse_coo_tensor(indices, values, (3, 3))
    x = torch.tensor([[1., 2.], [1., 2.], [1., 2.]], dtype=torch.float, device='cuda:0')

    b = s.matmul(x)

    print(s.to_dense())
    print(b)

    ret = sparse_triangular_solve(s, b)
    print(ret)

    # nonzero entries in the upper triangular part will be ignored
    s._values()[-1] = 100.
    print(s.to_dense())
    print(b)

    ret = sparse_triangular_solve(s, b)
    print(ret)
