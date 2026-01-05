import unittest
from unittest.mock import patch

import torch

from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.variational.large_batch_variational_strategy import LargeBatchVariationalStrategy, QuadFormDiagonal
from gpytorch.variational.variational_strategy import VariationalStrategy

from .test_variational_strategy import TestVariationalGP


class TestQuadFormDiagonal(BaseTestCase, unittest.TestCase):
    def create_inputs(self):
        m = 2
        n = 3

        A = torch.rand(m, m)
        A = A + A.mT
        A.requires_grad_(True)

        B = torch.rand(m, n).requires_grad_(True)

        return A, B

    def test_forward_backward(self):
        A, B = self.create_inputs()

        # custom autograd function
        diag = QuadFormDiagonal.apply(A, B)
        loss = diag.sum()
        loss.backward()

        # ground truth
        A_copy = A.clone().detach().requires_grad_(True)
        B_copy = B.clone().detach().requires_grad_(True)
        expected_diag = torch.diagonal(B_copy.mT @ A_copy @ B_copy)
        expected_loss = expected_diag.sum()
        expected_loss.backward()

        # test forward
        self.assertAllClose(diag, expected_diag)

        # test backward
        self.assertAllClose(A.grad, A_copy.grad)
        self.assertAllClose(B.grad, B_copy.grad)


class TestLargeBatchVariationalGP(TestVariationalGP):
    @property
    def strategy_cls(self):
        return LargeBatchVariationalStrategy

    def test_forward_train_eval(self, *args, **kwargs):
        model, _ = self._make_model_and_likelihood(
            batch_shape=self.batch_shape,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls,
        )

        train_x = torch.rand(*self.batch_shape, 4, 2)

        # In train mode, the custom autograd function is called
        model.train()
        with patch.object(QuadFormDiagonal, "forward", wraps=QuadFormDiagonal.forward) as mock_forward:
            predictive_dist1 = model(train_x)
            mock_forward.assert_called()

        # In eval mode, the custom autograd function should not be called
        model.eval()
        with patch.object(QuadFormDiagonal, "forward", wraps=QuadFormDiagonal.forward) as mock_forward:
            predictive_dist2 = model(train_x)
            mock_forward.assert_not_called()

        # The train mode and eval mode should produce the same predictive mean and variance
        self.assertAllClose(predictive_dist1.mean, predictive_dist2.mean)
        self.assertAllClose(predictive_dist1.variance, predictive_dist2.variance)

    def test_against_variational_strategy(self, train: bool = True):
        train_x = torch.rand(5, 2)

        torch.manual_seed(42)
        model1, _ = self._make_model_and_likelihood(strategy_cls=LargeBatchVariationalStrategy)
        model1.train(mode=train)
        output1 = model1(train_x)
        loss1 = output1.mean.mean() + output1.variance.mean()
        loss1.backward()

        torch.manual_seed(42)
        model2, _ = self._make_model_and_likelihood(strategy_cls=VariationalStrategy)
        model2.train(mode=train)
        output2 = model2(train_x)
        loss2 = output2.mean.mean() + output2.variance.mean()
        loss2.backward()

        # NOTE: Make sure the two models are the same before running the actual tests. The forward pass changes the
        # model parameters non-deterministically. Because the variational distribution gets re-initialized the first
        # time the model is called.
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertAllClose(p1, p2)

        self.assertAllClose(output1.mean, output2.mean)
        self.assertAllClose(output1.variance, output2.variance)

        # NOTE: `LargeBatchVariationalStrategy` always does computation in FP64, while `VariationalStrategy` by default
        # does most computation in FP32. The tests seem fine despite this difference in precision.
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertIsNotNone(p1.grad)
            self.assertIsNotNone(p2.grad)
            self.assertAllClose(p1.grad, p2.grad)

    def test_against_variational_strategy_eval(self):
        self.test_against_variational_strategy(train=False)
