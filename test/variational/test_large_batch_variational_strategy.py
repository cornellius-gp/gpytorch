import torch

from gpytorch.variational.large_batch_variational_strategy import LargeBatchVariationalStrategy
from gpytorch.variational.variational_strategy import VariationalStrategy

from test.variational.test_variational_strategy import TestVariationalGP


class TestLargeBatchVariationalGP(TestVariationalGP):
    @property
    def strategy_cls(self):
        return LargeBatchVariationalStrategy

    def test_forward_backward(self, mode: bool = True):
        train_x = torch.rand(5, 2)

        torch.manual_seed(42)
        model1, _ = self._make_model_and_likelihood(strategy_cls=LargeBatchVariationalStrategy)
        model1.train(mode=mode)
        output1 = model1(train_x)
        loss1 = output1.mean.mean() + output1.variance.mean()
        loss1.backward()

        torch.manual_seed(42)
        model2, _ = self._make_model_and_likelihood(strategy_cls=VariationalStrategy)
        model2.train(mode=mode)
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

    def test_forward_backward_eval(self):
        self.test_forward_backward(mode=False)
