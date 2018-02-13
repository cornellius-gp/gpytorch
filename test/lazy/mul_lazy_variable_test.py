import torch
from torch.autograd import Variable
from gpytorch.lazy import MulLazyVariable, RootLazyVariable
from gpytorch.utils import prod


def make_random_mat(size, rank, batch_size=None):
    if batch_size is None:
        res = torch.randn(size, rank)
    else:
        res = torch.randn(batch_size, size, rank)
    return Variable(res, requires_grad=True)


def test_matmul_vec_with_two_matrices():
    mat1 = make_random_mat(20, 5)
    mat2 = make_random_mat(20, 5)
    vec = Variable(torch.randn(20), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_matmul_vec_with_five_matrices():
    mat1 = make_random_mat(20, 5)
    mat2 = make_random_mat(20, 5)
    mat3 = make_random_mat(20, 5)
    mat4 = make_random_mat(20, 5)
    mat5 = make_random_mat(20, 5)
    vec = Variable(torch.randn(20), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)
    mat4_copy = Variable(mat4.data, requires_grad=True)
    mat5_copy = Variable(mat5.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3),
                          RootLazyVariable(mat4), RootLazyVariable(mat5)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
        mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
        mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_matmul_mat_with_two_matrices():
    mat1 = make_random_mat(20, 5)
    mat2 = make_random_mat(20, 5)
    vec = Variable(torch.randn(20, 7), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_matmul_mat_with_five_matrices():
    mat1 = make_random_mat(20, 5)
    mat2 = make_random_mat(20, 5)
    mat3 = make_random_mat(20, 5)
    mat4 = make_random_mat(20, 5)
    mat5 = make_random_mat(20, 5)
    vec = Variable(torch.eye(20), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)
    mat4_copy = Variable(mat4.data, requires_grad=True)
    mat5_copy = Variable(mat5.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3),
                          RootLazyVariable(mat4), RootLazyVariable(mat5)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
        mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
        mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_batch_matmul_mat_with_two_matrices():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    vec = Variable(torch.randn(5, 20, 7), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_batch_matmul_mat_with_five_matrices():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    mat3 = make_random_mat(20, rank=4, batch_size=5)
    mat4 = make_random_mat(20, rank=4, batch_size=5)
    mat5 = make_random_mat(20, rank=4, batch_size=5)
    vec = Variable(torch.randn(5, 20, 7), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)
    mat4_copy = Variable(mat4.data, requires_grad=True)
    mat5_copy = Variable(mat5.data, requires_grad=True)
    vec_copy = Variable(vec.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3),
                          RootLazyVariable(mat4), RootLazyVariable(mat5)).matmul(vec)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
        mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
        mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
    ]).matmul(vec_copy)
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    # Backward
    res.sum().backward()
    actual.sum().backward()
    assert torch.max(((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()) < 0.01
    assert torch.max(((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()) < 0.01
    assert torch.max(((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()) < 0.01


def test_mul_adding_another_variable():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    mat3 = make_random_mat(20, rank=4, batch_size=5)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2))
    res = res * RootLazyVariable(mat3)
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ])
    assert torch.max(((res.evaluate().data - actual.data) / actual.data).abs()) < 0.01


def test_mul_adding_constant_mul():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    mat3 = make_random_mat(20, rank=4, batch_size=5)
    const = Variable(torch.ones(1), requires_grad=True)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)
    const_copy = Variable(const.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3))
    res = res * const
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ]) * const_copy
    assert torch.max(((res.evaluate().data - actual.data) / actual.data).abs()) < 0.01

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3))
    res = res * 2.5
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ]) * 2.5
    assert torch.max(((res.evaluate().data - actual.data) / actual.data).abs()) < 0.01


def test_diag():
    mat1 = make_random_mat(20, rank=4)
    mat2 = make_random_mat(20, rank=4)
    mat3 = make_random_mat(20, rank=4)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)).diag()
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ]).diag()
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01


def test_batch_diag():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    mat3 = make_random_mat(20, rank=4, batch_size=5)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)).diag()
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ])
    actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(5)])
    assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01


def test_getitem():
    mat1 = make_random_mat(20, rank=4)
    mat2 = make_random_mat(20, rank=4)
    mat3 = make_random_mat(20, rank=4)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3))
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ])

    assert torch.max(((res[5, 3:5].data - actual[5, 3:5].data) / actual[5, 3:5].data).abs()) < 0.01
    assert torch.max(((res[3:5, 2:].evaluate().data - actual[3:5, 2:].data) / actual[3:5, 2:].data).abs()) < 0.01
    assert torch.max(((res[2:, 3:5].evaluate().data - actual[2:, 3:5].data) / actual[2:, 3:5].data).abs()) < 0.01


def test_batch_getitem():
    mat1 = make_random_mat(20, rank=4, batch_size=5)
    mat2 = make_random_mat(20, rank=4, batch_size=5)
    mat3 = make_random_mat(20, rank=4, batch_size=5)

    mat1_copy = Variable(mat1.data, requires_grad=True)
    mat2_copy = Variable(mat2.data, requires_grad=True)
    mat3_copy = Variable(mat3.data, requires_grad=True)

    # Forward
    res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3))
    actual = prod([
        mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
        mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
        mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
    ])

    assert torch.max(((res[0].evaluate().data - actual[0].data) / actual[0].data).abs()) < 0.01
    assert torch.max(((res[0:2, 5, 3:5].data - actual[0:2, 5, 3:5].data) / actual[0:2, 5, 3:5].data).abs()) < 0.01
    assert torch.max(((res[:, 3:5, 2:].evaluate().data - actual[:, 3:5, 2:].data) /
                      actual[:, 3:5, 2:].data).abs()) < 0.01
    assert torch.max(((res[:, 2:, 3:5].evaluate().data - actual[:, 2:, 3:5].data) /
                      actual[:, 2:, 3:5].data).abs()) < 0.01
