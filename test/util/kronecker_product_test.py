import torch
from gpytorch.utils.kronecker_product import kronecker_product_toeplitz_mul, kronecker_product, \
    kronecker_product_mul
from gpytorch.utils.toeplitz import toeplitz


def test_kronecker_product():
    matrix_list = []
    matrix1 = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    matrix2 = torch.Tensor([
        [1, 2],
        [4, 3],
    ])
    matrix_list.append(matrix1)
    matrix_list.append(matrix2)
    res = kronecker_product(matrix_list)

    actual = torch.Tensor([
        [1, 2, 2, 4, 3, 6],
        [4, 3, 8, 6, 12, 9],
        [4, 8, 5, 10, 6, 12],
        [16, 12, 20, 15, 24, 18]
    ])

    assert(torch.equal(res, actual))


def test_kronecker_product_toeplitz_mul():
    toeplitz_columns = torch.randn(3, 3)
    matrix = torch.randn(27, 10)
    res = kronecker_product_toeplitz_mul(toeplitz_columns, toeplitz_columns, matrix)

    toeplitz_matrices = torch.zeros(3, 3, 3)
    for i in range(3):
        toeplitz_matrices[i] = toeplitz(toeplitz_columns[i], toeplitz_columns[i])

    kronecker_product_matrix = kronecker_product(toeplitz_matrices)
    actual = kronecker_product_matrix.mm(matrix)

    assert(torch.norm(res - actual) < 1e-4)


def test_kronecker_product_mul():
    kronecker_matrices = []
    kronecker_matrices.append(torch.randn(3, 3))
    kronecker_matrices.append(torch.randn(2, 2))
    kronecker_matrices.append(torch.randn(3, 3))

    matrix = torch.randn(3 * 2 * 3, 9)
    res = kronecker_product_mul(kronecker_matrices, matrix)

    kronecker_product_matrix = kronecker_product(kronecker_matrices)
    actual = kronecker_product_matrix.mm(matrix)
    assert(torch.norm(res - actual) < 1e-4)
