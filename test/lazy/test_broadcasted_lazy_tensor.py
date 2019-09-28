#!/usr/bin/env python3

import torch
import unittest
from gpytorch import lazify
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase

class TestSumLazyTensorBroadcasting(unittest.TestCase):
    def test_broadcast_same_shape(self):
        test1 = lazify(torch.randn(30, 30))

        test2 = torch.randn(30,30)
        res = test1 + test2
        final_res = res + test2 

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_tensor_shape(self):
        test1 = lazify(torch.randn(30, 30))

        test2 = torch.randn(30,1)
        res = test1 + test2
        final_res = res + test2 

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_lazy_shape(self):
        test1 = lazify(torch.randn(30, 1))

        test2 = torch.randn(30,30)
        res = test1 + test2
        final_res = res + test2 

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

if __name__ == "__main__":
    unittest.main()
