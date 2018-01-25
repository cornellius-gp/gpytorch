import torch
from gpytorch.utils import approx_equal
from gpytorch.utils.lanczos import lanczos_tridiag


def test_lanczos():
    size = 100
    matrix = torch.randn(size, size)
    matrix = matrix.matmul(matrix.transpose(-1, -2))
    matrix.div_(matrix.norm())
    matrix.add_(torch.ones(matrix.size(-1)).mul(1e-6).diag())
    q_mat, t_mat = lanczos_tridiag(matrix.matmul, max_iter=size, tensor_cls=matrix.new, n_dims=matrix.size(-1))

    approx = q_mat.matmul(t_mat).matmul(q_mat.transpose(-1, -2))
    assert approx_equal(approx, matrix)
