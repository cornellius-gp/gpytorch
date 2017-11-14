import torch
from torch.autograd import Variable
from gpytorch.lazy import NonLazyVariable, SumInterpolatedLazyVariable, ToeplitzLazyVariable
from gpytorch.utils import approx_equal


def test_matmul_batch():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]])).repeat(5, 3, 1)
    right_interp_values = Variable(torch.Tensor([[1, 2], [2, 0.5], [1, 3]])).repeat(5, 3, 1)

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
    test_matrix = Variable(torch.randn(9, 4))

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)
    res = interp_lazy_var.matmul(test_matrix)

    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ]).repeat(5, 1, 1)

    right_matrix = torch.Tensor([
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
    ]).repeat(5, 1, 1)

    actual = left_matrix.matmul(base_lazy_variable_mat).matmul(right_matrix.transpose(-1, -2))
    actual = actual.matmul(test_matrix.data).sum(0)
    assert approx_equal(res.data, actual, epsilon=1e-3)


def test_derivatives():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)
    right_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    right_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
    test_matrix = Variable(torch.randn(9, 4))

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)
    res = interp_lazy_var.matmul(test_matrix)
    res.sum().backward()

    base_lazy_variable2 = Variable(base_lazy_variable_mat, requires_grad=True)
    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0], [0, 0, 0, 0, 1, 3],
    ]).repeat(5, 1, 1)

    actual = Variable(left_matrix).matmul(base_lazy_variable2).matmul(Variable(left_matrix).transpose(-1, -2))
    actual = actual.matmul(test_matrix).sum(0)
    actual.sum().backward()

    assert approx_equal(base_lazy_variable.var.grad.data, base_lazy_variable2.grad.data)


def test_getitem_batch():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1))
    left_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1))
    right_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)

    actual = (base_lazy_variable[:, 2:5, 0:3] + base_lazy_variable[:, 2:5, 1:4] +
              base_lazy_variable[:, 3:6, 0:3] + base_lazy_variable[:, 3:6, 1:4]).evaluate()

    assert approx_equal(interp_lazy_var[:1, :2].evaluate().data, actual[:, :1, :2].data.sum(0))
    assert approx_equal(interp_lazy_var[:1, 2].data, actual[:, :1, 2].data.sum(0))


def test_diag():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1))
    left_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1))
    right_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))

    # Non-lazy variable
    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)

    res = interp_lazy_var.diag()
    actual = interp_lazy_var.evaluate().diag()
    assert approx_equal(res, actual)

    # Toeplitz
    base_lazy_variable = ToeplitzLazyVariable(Variable(torch.randn(5, 6)))
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)

    res = interp_lazy_var.diag()
    actual = interp_lazy_var.evaluate().diag()
    assert approx_equal(res, actual)

    # Constant mul
    base_lazy_variable = base_lazy_variable * Variable(torch.ones(1) * 1.3)
    interp_lazy_var = SumInterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                                  right_interp_indices, right_interp_values)

    res = interp_lazy_var.diag()
    actual = interp_lazy_var.evaluate().diag()
    assert approx_equal(res, actual)
