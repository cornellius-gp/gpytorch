import torch
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable


toeplitz_column = torch.Tensor([2, 0, 4, 1])
batch_toeplitz_column = torch.Tensor([
    [2, 0, 4, 1],
    [1, 1, -1, 3],
])


def test_evaluate():
    lazy_toeplitz_var = ToeplitzLazyVariable(Variable(toeplitz_column))
    res = lazy_toeplitz_var.evaluate()
    actual = torch.Tensor([
        [2, 0, 4, 1],
        [0, 2, 0, 4],
        [4, 0, 2, 0],
        [1, 4, 0, 2],
    ])
    assert utils.approx_equal(res, actual)

    lazy_toeplitz_var = ToeplitzLazyVariable(Variable(batch_toeplitz_column))
    res = lazy_toeplitz_var.evaluate()
    actual = torch.Tensor([
        [
            [2, 0, 4, 1],
            [0, 2, 0, 4],
            [4, 0, 2, 0],
            [1, 4, 0, 2],
        ],
        [
            [1, 1, -1, 3],
            [1, 1, 1, -1],
            [-1, 1, 1, 1],
            [3, -1, 1, 1],
        ],
    ])
    assert utils.approx_equal(res, actual)


def test_get_item_square_on_variable():
    toeplitz_var = ToeplitzLazyVariable(Variable(torch.Tensor([1, 2, 3, 4])))
    evaluated = toeplitz_var.evaluate().data

    assert utils.approx_equal(toeplitz_var[2:4, 2:4].evaluate().data, evaluated[2:4, 2:4])


def test_get_item_on_batch():
    toeplitz_var = ToeplitzLazyVariable(Variable(batch_toeplitz_column))
    evaluated = toeplitz_var.evaluate().data
    assert utils.approx_equal(toeplitz_var[0, 1:3].evaluate().data, evaluated[0, 1:3])


def test_get_item_scalar_on_batch():
    toeplitz_var = ToeplitzLazyVariable(Variable(torch.Tensor([[1, 2, 3, 4]])))
    evaluated = toeplitz_var.evaluate().data
    assert utils.approx_equal(toeplitz_var[0].evaluate().data, evaluated[0])
