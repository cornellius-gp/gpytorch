import io
import unittest

import torch
from linear_operator import operators

import gpytorch

from gpytorch.models.zoo import CholeskyGP
from gpytorch.test.model_test_case import BaseModelTestCase

N_PTS = 50


class TestCholeskyGP(BaseModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = CholeskyGP(
            mean=gpytorch.means.ConstantMean(),
            kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=likelihood,
        )
        return model

    def create_test_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.randn(N_PTS) + 2
        return likelihood, labels

    def create_batch_test_data(self, batch_shape=torch.Size([3])):
        return torch.randn(*batch_shape, N_PTS, 1)

    def create_batch_likelihood_and_labels(self, batch_shape=torch.Size([3])):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        labels = torch.randn(*batch_shape, N_PTS) + 2
        return likelihood, labels

    def test_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_forward_eval()

    def test_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_batch_forward_eval()

    def test_multi_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_multi_batch_forward_eval()

    def test_batch_forward_then_nonbatch_forward_eval(self):
        batch_data = self.create_batch_test_data()
        likelihood, labels = self.create_batch_likelihood_and_labels()
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)

        # Smoke test derivatives working
        output.mean.sum().backward()

        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Create non-batch data
        data = self.create_test_data()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

        # Smoke test derivatives working
        output.mean.sum().backward()

    def test_batch_forward_then_different_batch_forward_eval(self):
        non_batch_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(non_batch_data, labels, likelihood)
        model.eval()

        # Batch size 3
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now Batch size 2
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 3 again
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 1
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([1]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_prior_mode(self):
        train_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        prior_model = self.create_model(None, None, likelihood)
        model = self.create_model(train_data, labels, likelihood)
        prior_model.eval()
        model.eval()

        test_data = self.create_test_data()
        prior_out = prior_model(test_data)
        with gpytorch.settings.prior_mode(True):
            prior_out_cm = model(test_data)
        self.assertTrue(torch.allclose(prior_out.mean, prior_out_cm.mean))
        self.assertTrue(torch.allclose(prior_out.covariance_matrix, prior_out_cm.covariance_matrix))

    # TODO ------ consider moving to test_approximation_strategy or model_test_case.py--------#
    def test_train_data_is_updated_consistently(self):
        """Test whether updating the training data is done consistently in the model and the approximation strategy."""
        x_train = torch.randn(N_PTS, 1)
        likelihood, y_train = self.create_likelihood_and_labels()
        model = self.create_model(x_train, y_train, likelihood)
        model(x_train)

        x_train_new = torch.zeros_like(x_train)
        model.train_inputs = x_train_new
        self.assertTrue(torch.equal(x_train_new, model.train_inputs))
        self.assertTrue(torch.equal(x_train_new, model.approximation_strategy.model.train_inputs))

        y_train_new = torch.zeros_like(y_train)
        model.train_targets = y_train_new
        self.assertTrue(torch.equal(y_train_new, model.train_targets))
        self.assertTrue(torch.equal(y_train_new, model.approximation_strategy.model.train_targets))

    def test_save_and_load_model_preserves_cached_quantities(self):
        """Test whether saving and loading a model preserves caches."""

        x_train = torch.randn(N_PTS, 1)
        likelihood, y_train = self.create_likelihood_and_labels()
        model = self.create_model(x_train, y_train, likelihood)

        # Predict with model to ensure cached quantities are computed
        model.eval()
        model(x_train)

        # Save to IO buffer (to mock saving to file)
        buffer = io.BytesIO()
        torch.save(model, buffer)

        # Load from IO buffer
        buffer.seek(0)
        model_loaded = torch.load(buffer)

        # Compare cached quantities between original and loaded model
        for cached_quantity_name, cached_quantity in model.approximation_strategy.named_buffers():
            if isinstance(cached_quantity, torch.Tensor):
                loaded_cached_quantity = model_loaded.approximation_strategy.__getattr__(cached_quantity_name)
                self.assertTrue(torch.equal(cached_quantity, loaded_cached_quantity))
                self.assertTrue(cached_quantity.requires_grad == loaded_cached_quantity.requires_grad)
            else:
                for i, cached_quantity_representation_tensor in enumerate(cached_quantity.representation()):
                    loaded_cached_quantity_representation_tensor = model_loaded.approximation_strategy.__getattr__(
                        cached_quantity_name
                    ).representation()[i]
                    self.assertTrue(
                        torch.equal(
                            cached_quantity_representation_tensor,
                            loaded_cached_quantity_representation_tensor,
                        )
                    )
                    self.assertTrue(
                        (
                            cached_quantity_representation_tensor.requires_grad
                            == loaded_cached_quantity_representation_tensor.requires_grad
                        )
                    )

        self.assertTrue(
            model.approximation_strategy._cache_initialized == model_loaded.approximation_strategy._cache_initialized
        )

    def test_linear_operators_as_cached_quantities(self):
        """Test whether one can cache linear operators."""
        x_train = torch.randn(N_PTS, 1)
        likelihood, y_train = self.create_likelihood_and_labels()
        model = self.create_model(x_train, y_train, likelihood)

        model.approximation_strategy.init_cache(model)
        model.approximation_strategy.register_cached_quantity(
            "cached_quantity",
            quantity=operators.MatmulLinearOperator(
                operators.DenseLinearOperator(torch.eye(5)), operators.DenseLinearOperator(torch.ones(5, 3))
            ),
        )
        self.assertTrue(isinstance(model.approximation_strategy.cached_quantity, operators.LinearOperator))

    def test_cached_quantities_require_grad_if_assigned_quantity_requires_grad(self):
        """Test whether cached quantities require gradients if the assigned tensor / linear operator does."""

        x_train = torch.randn(N_PTS, 1)
        likelihood, y_train = self.create_likelihood_and_labels()
        model = self.create_model(x_train, y_train, likelihood)

        # Predict with model to ensure cached quantities are computed
        model.eval()
        model(x_train)

        for buffer_name, buffer in model.approximation_strategy.named_buffers():
            for req_grad in [True, False]:
                model.approximation_strategy.__setattr__(buffer_name, buffer.requires_grad_(req_grad))
                self.assertTrue(model.approximation_strategy.__getattr__(buffer_name).requires_grad == req_grad)

    def test_updating_training_data_clears_caches(self):
        # TODO
        pass

    def test_overwriting_cache_without_overwrite_option_throws_error(self):
        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
