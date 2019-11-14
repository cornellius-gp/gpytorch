#!/usr/bin/env python3

from abc import abstractmethod

import torch

import gpytorch


class BaseModelTestCase(object):
    @abstractmethod
    def create_model(self, train_x, train_y, likelihood):
        raise NotImplementedError()

    @abstractmethod
    def create_test_data(self):
        raise NotImplementedError()

    @abstractmethod
    def create_likelihood_and_labels(self):
        raise NotImplementedError()

    @abstractmethod
    def create_batch_test_data(self, batch_shape=torch.Size([3])):
        raise NotImplementedError()

    @abstractmethod
    def create_batch_likelihood_and_labels(self, batch_shape=torch.Size([3])):
        raise NotImplementedError()

    def test_forward_train(self):
        data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(data, labels, likelihood)
        model.train()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_train(self):
        batch_data = self.create_batch_test_data()
        likelihood, labels = self.create_batch_likelihood_and_labels()
        model = self.create_model(batch_data, labels, likelihood)
        model.train()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_multi_batch_forward_train(self):
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2, 3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(batch_shape=torch.Size([2, 3]))
        model = self.create_model(batch_data, labels, likelihood)
        model.train()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 4)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_forward_eval(self):
        data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(data, labels, likelihood)
        model.eval()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_eval(self):
        batch_data = self.create_batch_test_data()
        likelihood, labels = self.create_batch_likelihood_and_labels()
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_multi_batch_forward_eval(self):
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2, 3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(batch_shape=torch.Size([2, 3]))
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 4)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))


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
