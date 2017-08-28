import math
import torch
from gpytorch.utils import circulant
from gpytorch import utils


def test_rotate_vector_forward():
    a = torch.randn(5)
    Q0 = torch.zeros(5, 5)
    Q0[0, 4] = 1
    Q0[1:, :-1] = torch.eye(4)

    Q = Q0.clone()
    for i in range(1, 5):
        a_rotated_result = circulant.rotate(a, i)
        a_rotated_actual = Q.matmul(a)

        assert(utils.approx_equal(a_rotated_actual, a_rotated_result))
        Q = Q.matmul(Q0)


def test_rotate_vector_reverse():
    a = torch.randn(5)
    Q0 = torch.zeros(5, 5)
    Q0[0, 4] = 1
    Q0[1:, :-1] = torch.eye(4)

    Q = Q0.clone()
    for i in range(1, 5):
        a_rotated_result = circulant.rotate(a, -i)
        a_rotated_actual = Q.inverse().matmul(a)

        assert(utils.approx_equal(a_rotated_actual, a_rotated_result))
        Q = Q.matmul(Q0)


def test_rotate_matrix_forward():
    a = torch.randn(5, 5)
    Q0 = torch.zeros(5, 5)
    Q0[0, 4] = 1
    Q0[1:, :-1] = torch.eye(4)

    Q = Q0.clone()
    for i in range(1, 5):
        a_rotated_result = circulant.rotate(a, i)
        a_rotated_actual = Q.matmul(a)

        assert(utils.approx_equal(a_rotated_actual, a_rotated_result))
        Q = Q.matmul(Q0)


def test_rotate_matrix_reverse():
    a = torch.randn(5, 5)
    Q0 = torch.zeros(5, 5)
    Q0[0, 4] = 1
    Q0[1:, :-1] = torch.eye(4)

    Q = Q0.clone()
    for i in range(1, 5):
        a_rotated_result = circulant.rotate(a, -i)
        a_rotated_actual = Q.inverse().matmul(a)

        assert(utils.approx_equal(a_rotated_actual, a_rotated_result))
        Q = Q.matmul(Q0)


def test_left_rotate_trace():
    a = torch.randn(5, 5)

    for i in range(1, 5):
        actual = circulant.rotate(a, i).trace()
        result = circulant.left_rotate_trace(a, i)

        assert(math.fabs(actual - result) < 1e-5)


def test_right_rotate_trace():
    a = torch.randn(5, 5)

    for i in range(1, 5):
        actual = circulant.rotate(a, -i).trace()
        result = circulant.left_rotate_trace(a, -i)

        assert(math.fabs(actual - result) < 1e-5)


def test_circulant_transpose():
    a = torch.randn(5)

    C = circulant.circulant(a)
    C_T_actual = C.t()
    C_T_result = circulant.circulant(circulant.circulant_transpose(a))

    assert(utils.approx_equal(C_T_actual, C_T_result))


def test_circulant_mv():
    a = torch.randn(5)
    v = torch.randn(5)

    av_result = circulant.circulant_mv(a, v)
    C = circulant.circulant(a)
    av_actual = C.mv(v)

    assert(utils.approx_equal(av_result, av_actual))


def test_circulant_mm():
    a = torch.randn(5)
    M = torch.randn(5, 5)

    aM_result = circulant.circulant_mm(a, M)
    C = circulant.circulant(a)
    aM_actual = C.mm(M)

    assert(utils.approx_equal(aM_result, aM_actual))


def test_circulant_invmv():
    a = torch.randn(5)
    v = torch.randn(5)

    av_result = circulant.circulant_invmv(a, v)
    C = circulant.circulant(a)
    av_actual = C.inverse().mv(v)

    assert(utils.approx_equal(av_result, av_actual))


def test_circulant_invmm():
    a = torch.randn(5)
    M = torch.randn(5, 5)

    aM_result = circulant.circulant_invmm(a, M)
    C = circulant.circulant(a)
    aM_actual = C.inverse().mm(M)

    assert(utils.approx_equal(aM_result, aM_actual))


def test_frobenius_circulant_approximation():
    A = torch.randn(5, 5)

    C1 = circulant.frobenius_circulant_approximation(A)
    C2 = circulant.frobenius_circulant_approximation(circulant.circulant(C1))

    assert(utils.approx_equal(C1, C2))


def test_frobenius_circulant_approximation_toeplitz():
    toeplitz_column = torch.randn(5)

    C1 = circulant.frobenius_circulant_approximation_toeplitz(toeplitz_column)
    C2 = circulant.frobenius_circulant_approximation_toeplitz(C1)

    assert(torch.norm(C1 - C2) < 1e-3)
