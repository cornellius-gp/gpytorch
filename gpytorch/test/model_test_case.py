#!/usr/bin/env python3

from abc import abstractmethod

import torch

import gpytorch
from gpytorch import likelihoods


N_TRAIN = 50
N_TEST = 20


class BaseModelTestCase(object):
    @abstractmethod
    def create_model(self, train_inputs, train_targets, likelihood) -> gpytorch.Module:
        raise NotImplementedError()

    @abstractmethod
    def create_train_data(self, batch_shape=()):
        raise NotImplementedError()

    @abstractmethod
    def create_test_data(self, batch_shape=()):
        raise NotImplementedError()

    @abstractmethod
    def create_likelihood(self) -> likelihoods.Likelihood:
        raise NotImplementedError()

    def test_forward_train(self):
        train_inputs, train_targets = self.create_train_data()
        test_inputs, _ = self.create_test_data()
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.train()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))

    def test_batch_forward_train(self):
        batch_shape = torch.Size((5,))
        train_inputs, train_targets = self.create_train_data(batch_shape=batch_shape)
        test_inputs, _ = self.create_test_data(batch_shape=batch_shape)
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.train()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))

    def test_multi_batch_forward_train(self):
        batch_shape = torch.Size((2, 3))
        train_inputs, train_targets = self.create_train_data(batch_shape=batch_shape)
        test_inputs, _ = self.create_test_data(batch_shape=batch_shape)
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.train()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 4)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))

    def test_forward_eval(self):
        train_inputs, train_targets = self.create_train_data()
        test_inputs, _ = self.create_test_data()
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.eval()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))

    def test_batch_forward_eval(self):
        batch_shape = torch.Size((5,))
        train_inputs, train_targets = self.create_train_data(batch_shape=batch_shape)
        test_inputs, _ = self.create_test_data(batch_shape=batch_shape)
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.eval()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))

    def test_multi_batch_forward_eval(self):
        batch_shape = torch.Size((2, 3))
        train_inputs, train_targets = self.create_train_data(batch_shape=batch_shape)
        test_inputs, _ = self.create_test_data(batch_shape=batch_shape)
        model = self.create_model(
            train_inputs=train_inputs, train_targets=train_targets, likelihood=self.create_likelihood()
        )
        model.eval()
        output = model(test_inputs)

        self.assertTrue(output.lazy_covariance_matrix.dim() == 4)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == test_inputs.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == test_inputs.size(-2))


class VariationalModelTestCase(BaseModelTestCase):
    def test_backward_train(self):
        data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(data, labels, likelihood)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.size(-1))
        model.train()
        likelihood.train()

        # We'll just do one step of gradient descent to mix up the params a bit
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        output = model(data)
        loss = -mll(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, labels)
        loss.backward()

        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

    def test_batch_backward_train(self, batch_shape=torch.Size([3])):
        data = self.create_batch_test_data(batch_shape)
        likelihood, labels = self.create_batch_likelihood_and_labels(batch_shape)
        model = self.create_model(data, labels, likelihood)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.size(-1))
        model.train()
        likelihood.train()

        # We'll just do one step of gradient descent to mix up the params a bit
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        output = model(data)
        loss = -mll(output, labels).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, labels).sum()
        loss.backward()

        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

    def test_multi_batch_backward_train(self, batch_shape=torch.Size([2, 3])):
        return self.test_batch_backward_train(batch_shape=batch_shape)
