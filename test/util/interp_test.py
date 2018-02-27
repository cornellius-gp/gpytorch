import torch
from torch.autograd import Variable
from gpytorch.utils import left_interp, left_t_interp, approx_equal

interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(3, 1)
interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(3, 1)

interp_indices_2 = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]])).repeat(3, 1)
interp_values_2 = Variable(torch.Tensor([[1, 2], [2, 0.5], [1, 3]])).repeat(3, 1)
batch_interp_indices = torch.cat([interp_indices.unsqueeze(0), interp_indices_2.unsqueeze(0)], 0)
batch_interp_values = torch.cat([interp_values.unsqueeze(0), interp_values_2.unsqueeze(0)], 0)

interp_matrix = torch.Tensor([
    [0, 0, 1, 2, 0, 0],
    [0, 0, 0, 0.5, 1, 0],
    [0, 0, 0, 0, 1, 3],
    [0, 0, 1, 2, 0, 0],
    [0, 0, 0, 0.5, 1, 0],
    [0, 0, 0, 0, 1, 3],
    [0, 0, 1, 2, 0, 0],
    [0, 0, 0, 0.5, 1, 0],
    [0, 0, 0, 0, 1, 3],
])

batch_interp_matrix = torch.Tensor([
    [
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ], [
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
    ]
])


def test_left_interp_on_a_vector():
    vector = torch.randn(6)

    res = left_interp(interp_indices, interp_values, Variable(vector)).data
    actual = torch.matmul(interp_matrix, vector)
    assert approx_equal(res, actual)


def test_left_t_interp_on_a_vector():
    vector = torch.randn(9)

    res = left_t_interp(interp_indices, interp_values, Variable(vector), 6).data
    actual = torch.matmul(interp_matrix.transpose(-1, -2), vector)
    assert approx_equal(res, actual)


def test_batch_left_interp_on_a_vector():
    vector = torch.randn(6)

    actual = torch.matmul(batch_interp_matrix, vector.unsqueeze(-1).unsqueeze(0)).squeeze(0)
    res = left_interp(batch_interp_indices, batch_interp_values, Variable(vector)).data
    assert approx_equal(res, actual)


def test_batch_left_t_interp_on_a_vector():
    vector = torch.randn(9)

    actual = torch.matmul(batch_interp_matrix.transpose(-1, -2), vector.unsqueeze(-1).unsqueeze(0)).squeeze(0)
    res = left_t_interp(batch_interp_indices, batch_interp_values, Variable(vector), 6).data
    assert approx_equal(res, actual)


def test_left_interp_on_a_matrix():
    matrix = torch.randn(6, 3)

    res = left_interp(interp_indices, interp_values, Variable(matrix)).data
    actual = torch.matmul(interp_matrix, matrix)
    assert approx_equal(res, actual)


def test_left_t_interp_on_a_matrix():
    matrix = torch.randn(9, 3)

    res = left_t_interp(interp_indices, interp_values, Variable(matrix), 6).data
    actual = torch.matmul(interp_matrix.transpose(-1, -2), matrix)
    assert approx_equal(res, actual)


def test_batch_left_interp_on_a_matrix():
    batch_matrix = torch.randn(6, 3)

    res = left_interp(batch_interp_indices, batch_interp_values, Variable(batch_matrix)).data
    actual = torch.matmul(batch_interp_matrix, batch_matrix.unsqueeze(0))
    assert approx_equal(res, actual)


def test_batch_left_t_interp_on_a_matrix():
    batch_matrix = torch.randn(9, 3)

    res = left_t_interp(batch_interp_indices, batch_interp_values, Variable(batch_matrix), 6).data
    actual = torch.matmul(batch_interp_matrix.transpose(-1, -2), batch_matrix.unsqueeze(0))
    assert approx_equal(res, actual)


def test_batch_left_interp_on_a_batch_matrix():
    batch_matrix = torch.randn(2, 6, 3)

    res = left_interp(batch_interp_indices, batch_interp_values, Variable(batch_matrix)).data
    actual = torch.matmul(batch_interp_matrix, batch_matrix)
    assert approx_equal(res, actual)


def test_batch_left_t_interp_on_a_batch_matrix():
    batch_matrix = torch.randn(2, 9, 3)

    res = left_t_interp(batch_interp_indices, batch_interp_values, Variable(batch_matrix), 6).data
    actual = torch.matmul(batch_interp_matrix.transpose(-1, -2), batch_matrix)
    assert approx_equal(res, actual)
