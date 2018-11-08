from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod


class ModelTestCase(object):
    @abstractmethod
    def create_model(self, train_data):
        raise NotImplementedError()

    @abstractmethod
    def create_test_data(self):
        raise NotImplementedError()

    @abstractmethod
    def create_batch_test_data(self):
        raise NotImplementedError()

    def test_forward_train(self):
        data = self.create_test_data()
        model = self.create_model(data)
        model.train()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_train(self):
        batch_data = self.create_batch_test_data()
        model = self.create_model(batch_data)
        model.train()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_forward_eval(self):
        data = self.create_test_data()
        model = self.create_model(data)
        model.eval()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 2)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_eval(self):
        batch_data = self.create_batch_test_data()
        model = self.create_model(batch_data)
        model.eval()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))
