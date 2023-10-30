#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import (
    LinearMean,
    LinearMeanGrad,
    LinearMeanGradGrad,
    PositiveQuadraticMean,
    PositiveQuadraticMeanGrad,
    PositiveQuadraticMeanGradGrad,
)

from gpytorch.test.base_test_case import BaseTestCase

w = torch.tensor([[1.0], [2.0]])
b = torch.tensor([1.0])
L = torch.tensor(
    [
        [1.0, 0.0],
        [2.0, 1.0],
    ]
)


class TestAdditiveAndSubstractiveMeans(BaseTestCase, unittest.TestCase):
    def test_no_grad(self):
        mean_mod1 = LinearMean(2).initialize(weights=w, bias=b)
        mean_mod2 = PositiveQuadraticMean(2).initialize(cholesky=torch.tensor([L[0, 0], L[1, 0], L[1, 1]]))
        mean_module = mean_mod1 + mean_mod2
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.assertAllClose(
            mean_module(x),
            b + x.matmul(w).squeeze() + x.matmul(L).pow(2).sum(-1).div(2),
        )
        mean_module_sub = mean_mod1 - mean_mod2
        self.assertAllClose(
            mean_module_sub(x),
            b + x.matmul(w).squeeze() - x.matmul(L).pow(2).sum(-1).div(2),
        )

    def test_batch_no_grad(self):
        batch_shape = torch.Size([3])
        mean_mod1 = LinearMean(2, batch_shape).initialize(weights=w, bias=b)
        mean_mod2 = PositiveQuadraticMean(2, batch_shape).initialize(cholesky=torch.tensor([L[0, 0], L[1, 0], L[1, 1]]))
        mean_module = mean_mod1 + mean_mod2
        x = torch.randn(*batch_shape, 5, 2)
        self.assertAllClose(
            mean_module(x),
            b + x.matmul(w).squeeze(-1) + x.matmul(L).pow(2).sum(-1).div(2),
        )

    def test_grad(self):
        mean_mod1 = LinearMeanGrad(2).initialize(weights=w, bias=b)
        mean_mod2 = PositiveQuadraticMeanGrad(2).initialize(cholesky=torch.tensor([L[0, 0], L[1, 0], L[1, 1]]))
        mean_module = mean_mod1 + mean_mod2
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.assertAllClose(
            mean_module(x),
            torch.cat(
                [
                    (b + x.matmul(w).squeeze() + x.matmul(L).pow(2).sum(-1).div(2)).unsqueeze(-1),
                    w.squeeze().expand_as(x) + x.matmul(L.matmul(L.T)),
                ],
                -1,
            ),
        )
        mean_module_sub = mean_mod1 - mean_mod2
        self.assertAllClose(
            mean_module_sub(x),
            torch.cat(
                [
                    (b + x.matmul(w).squeeze() - x.matmul(L).pow(2).sum(-1).div(2)).unsqueeze(-1),
                    w.squeeze().expand_as(x) - x.matmul(L.matmul(L.T)),
                ],
                -1,
            ),
        )

    def test_gradgrad(self):
        mean_mod1 = LinearMeanGradGrad(2).initialize(weights=w, bias=b)
        mean_mod2 = PositiveQuadraticMeanGradGrad(2).initialize(cholesky=torch.tensor([L[0, 0], L[1, 0], L[1, 1]]))
        mean_module = mean_mod1 + 2 * mean_mod2
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        A = L.matmul(L.T)
        self.assertAllClose(
            mean_module(x),
            torch.cat(
                [
                    torch.cat(
                        [
                            (b + x.matmul(w).squeeze() + x.matmul(L).pow(2).sum(-1)).unsqueeze(-1),
                            w.squeeze().expand_as(x) + 2 * x.matmul(A),
                            2 * A.diag().expand_as(x),
                        ],
                        -1,
                    ),
                ],
                -1,
            ),
        )
        mean_module_sub = mean_mod1 - mean_mod2
        self.assertAllClose(
            mean_module_sub(x),
            torch.cat(
                [
                    torch.cat(
                        [
                            (b + x.matmul(w).squeeze() - x.matmul(L).pow(2).sum(-1).div(2)).unsqueeze(-1),
                            w.squeeze().expand_as(x) - x.matmul(A),
                            -A.diag().expand_as(x),
                        ],
                        -1,
                    ),
                ],
                -1,
            ),
        )
        mean_module_neg = -mean_mod1
        self.assertAllClose(
            mean_module_neg(x),
            torch.cat(
                [
                    torch.cat(
                        [
                            -(b + x.matmul(w).squeeze()).unsqueeze(-1),
                            -w.squeeze().expand_as(x),
                            torch.zeros_like(x),
                        ],
                        -1,
                    ),
                ],
                -1,
            ),
        )

    def test_error_grad(self):
        mean_mod1 = LinearMean(2).initialize(weights=w, bias=b)
        mean_mod2 = PositiveQuadraticMeanGrad(2).initialize(cholesky=torch.tensor([L[0, 0], L[1, 0], L[1, 1]]))
        mean_module = mean_mod1 - mean_mod2
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        with self.assertRaises(RuntimeError):
            mean_module(x)
