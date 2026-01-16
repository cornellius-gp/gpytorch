import unittest
from unittest.mock import patch

import torch

from gpytorch.mlls import VariationalELBO
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

    def test_inference_caching(self):
        """Test that inference caching works correctly."""
        torch.manual_seed(42)

        model, likelihood = self._make_model_and_likelihood(
            batch_shape=self.batch_shape,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls,
        )

        # Training - caches should not exist
        model.train()
        likelihood.train()

        train_x = torch.rand(32, 2)
        _ = model(train_x)

        # Check that caches don't exist in training mode
        self.assertFalse(hasattr(model.variational_strategy, "_cached_inv_chol_t_inducing_values"))
        self.assertFalse(hasattr(model.variational_strategy, "_cached_middle_term"))

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        # First inference call should create caches
        test_x = torch.rand(5, 2)
        with torch.no_grad():
            _ = model(test_x)

        # Check caches exist
        self.assertTrue(hasattr(model.variational_strategy, "_cached_inv_chol_t_inducing_values"))
        self.assertTrue(hasattr(model.variational_strategy, "_cached_middle_term"))

        # Second inference call should use cached values
        cached_inv_chol = model.variational_strategy._cached_inv_chol_t_inducing_values
        cached_middle = model.variational_strategy._cached_middle_term

        with torch.no_grad():
            _ = model(test_x)

        # Verify caches are the same objects (not recomputed)
        self.assertTrue(model.variational_strategy._cached_inv_chol_t_inducing_values is cached_inv_chol)
        self.assertTrue(model.variational_strategy._cached_middle_term is cached_middle)

        # Switching back to train mode should clear caches
        model.train()
        _ = model(test_x)
        self.assertFalse(hasattr(model.variational_strategy, "_cached_inv_chol_t_inducing_values"))
        self.assertFalse(hasattr(model.variational_strategy, "_cached_middle_term"))

    def test_onnx_export(self):
        """Test that a trained SVGP with LargeBatchVariationalStrategy can be exported to ONNX in fp64."""
        import os
        import tempfile

        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime required for this test")

        from torch.onnx import register_custom_op_symbolic

        # Create and train model in fp64
        torch.manual_seed(42)
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=self.batch_shape,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls,
        )
        model = model.double()
        likelihood = likelihood.double()

        # Quick training on toy data (fp64)
        train_x = torch.rand(32, 2, dtype=torch.float64)
        train_y = torch.sin(train_x[:, 0]) + 0.1 * torch.randn(32, dtype=torch.float64)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        mll = VariationalELBO(likelihood, model, num_data=train_x.size(0))

        for _ in range(5):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        # Create a wrapper that returns the mean as a tensor (required for ONNX)
        class MeanPredictionWrapper(torch.nn.Module):
            def __init__(self, gp_model):
                super().__init__()
                self.gp_model = gp_model

            def forward(self, x):
                output = self.gp_model(x)
                return output.mean

        wrapper = MeanPredictionWrapper(model)
        wrapper.eval()

        # Do a dummy forward to populate the inference caches
        test_x = torch.rand(5, 2, dtype=torch.float64)
        with torch.no_grad():
            _ = wrapper(test_x)

        # Verify caches are populated and are in fp64
        self.assertTrue(hasattr(model.variational_strategy, "_cached_inv_chol_t_inducing_values"))
        self.assertTrue(hasattr(model.variational_strategy, "_cached_middle_term"))
        self.assertEqual(model.variational_strategy._cached_inv_chol_t_inducing_values.dtype, torch.float64)
        self.assertEqual(model.variational_strategy._cached_middle_term.dtype, torch.float64)

        # Get reference output for comparison
        with torch.no_grad():
            mean_pred = wrapper(test_x)
        self.assertEqual(mean_pred.shape, torch.Size([5]))
        self.assertEqual(mean_pred.dtype, torch.float64)

        # Register custom ONNX symbolics for ops not in default registry
        def mT_symbolic(g, self):
            # mT swaps the last two dimensions
            tensor_type = self.type()
            if tensor_type is not None and tensor_type.dim() is not None:
                rank = tensor_type.dim()
                perm = list(range(rank - 2)) + [rank - 1, rank - 2]
                return g.op("Transpose", self, perm_i=perm)
            return g.op("Transpose", self, perm_i=[1, 0])

        def softplus_symbolic(g, self, beta, threshold):
            # Numerically stable Softplus using Where:
            # softplus(x) = x + log(1 + exp(-x)) for x > 0
            #             = log(1 + exp(x)) for x <= 0
            scaled = g.op("Mul", self, beta)
            zero = g.op("Constant", value_t=torch.tensor([0.0], dtype=torch.float64))
            one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float64))
            condition = g.op("Greater", scaled, zero)

            neg_scaled = g.op("Neg", scaled)
            exp_neg = g.op("Exp", neg_scaled)
            one_plus_exp_neg = g.op("Add", one, exp_neg)
            log_pos = g.op("Log", one_plus_exp_neg)
            result_pos = g.op("Add", scaled, log_pos)

            exp_scaled = g.op("Exp", scaled)
            one_plus_exp = g.op("Add", one, exp_scaled)
            result_neg = g.op("Log", one_plus_exp)

            stable_result = g.op("Where", condition, result_pos, result_neg)
            return g.op("Div", stable_result, beta)

        try:
            register_custom_op_symbolic("aten::mT", mT_symbolic, 17)
        except Exception:
            pass
        try:
            register_custom_op_symbolic("aten::softplus", softplus_symbolic, 17)
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")

            # Build export kwargs - use legacy exporter if available (PyTorch 2.9+)
            export_kwargs = dict(
                input_names=["input"],
                output_names=["mean"],
                opset_version=17,
            )
            # dynamo=False forces the legacy TorchScript-based exporter
            if hasattr(torch.onnx.export, "__wrapped__") or torch.__version__ >= "2.9":
                export_kwargs["dynamo"] = False

            torch.onnx.export(
                wrapper,
                test_x,
                onnx_path,
                **export_kwargs,
            )

            self.assertTrue(os.path.exists(onnx_path))
            self.assertGreater(os.path.getsize(onnx_path), 0)

            # Verify the ONNX model input is fp64
            model_onnx = onnx.load(onnx_path)
            input_type = model_onnx.graph.input[0].type.tensor_type.elem_type
            self.assertEqual(input_type, 11)  # ONNX TensorProto.DOUBLE = 11

            # Verify with onnxruntime
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(onnx_path, sess_options)
            onnx_output = session.run(None, {"input": test_x.numpy()})[0]

            # Compare ONNX output with PyTorch output
            self.assertAllClose(torch.from_numpy(onnx_output), mean_pred, rtol=1e-5, atol=1e-5)
