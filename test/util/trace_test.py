import torch
import math
from gpytorch.utils import trace_components


def test_trace_components_normal_matrices():
    a_mat = torch.randn(3, 4)
    b_mat = torch.randn(3, 4)

    a_res, b_res = trace_components(a_mat, b_mat)
    assert torch.equal(a_res, a_mat)
    assert torch.equal(b_res, b_mat)


def test_trace_components_implicit_matrices_mubs():
    a_mat = torch.randn(500, 500)
    b_mat = torch.randn(500, 500)

    # Ensure positive definite
    a_mat = a_mat.t().matmul(a_mat) + torch.eye(500)
    b_mat = b_mat.t().matmul(b_mat) + torch.eye(500)

    a_res, b_res = trace_components(a_mat.matmul, b_mat.matmul, size=500,
                                    num_samples=10)

    actual_trace = (a_mat * b_mat).sum()
    stochastic_trace = (a_res * b_res).sum()
    assert math.fabs(actual_trace - stochastic_trace) / actual_trace < 1e-1


def test_trace_components_implicit_matrices_hutchinsom():
    a_mat = torch.randn(500, 500)
    b_mat = torch.randn(500, 500)

    # Ensure positive definite
    a_mat = a_mat.t().matmul(a_mat) + torch.eye(500)
    b_mat = b_mat.t().matmul(b_mat) + torch.eye(500)

    a_res, b_res = trace_components(a_mat.matmul, b_mat.matmul, size=500,
                                    num_samples=25, estimator_type='hutchinson')

    actual_trace = (a_mat * b_mat).sum()
    stochastic_trace = (a_res * b_res).sum()
    assert math.fabs(actual_trace - stochastic_trace) / actual_trace < 1e-1
