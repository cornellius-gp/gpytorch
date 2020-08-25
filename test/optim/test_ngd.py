#!/usr/bin/env python3

import unittest

import torch

from gpytorch.optim import NGD
from gpytorch.test.base_test_case import BaseTestCase


class TestNGD(unittest.TestCase, BaseTestCase):
    def test_ngd_step_no_groups(self):
        parameters = [
            torch.nn.Parameter(torch.tensor(2.5)),
            torch.nn.Parameter(torch.tensor([1.0, -0.5])),
        ]
        # parameters[0].grad = torch.tensor(1.)
        # parameters[1].grad = torch.tensor([2., -1.])

        optimizer = NGD(parameters, num_data=5, lr=0.1)
        optimizer.zero_grad()
        loss = parameters[0] + torch.dot(parameters[1], torch.tensor([2.0, -1]))
        loss.backward()
        optimizer.step()

        self.assertAllClose(parameters[0], torch.tensor(2.0))
        self.assertAllClose(parameters[1], torch.tensor([0.0, 0.0]))

    def test_ngd_step_groups(self):
        parameters = [
            {"params": [torch.nn.Parameter(torch.tensor(2.5))], "lr": 0.2},
            {"params": [torch.nn.Parameter(torch.tensor([1.0, -0.5]))]},
        ]

        optimizer = NGD(parameters, num_data=5, lr=0.1)
        optimizer.zero_grad()
        loss = parameters[0]["params"][0] + torch.dot(parameters[1]["params"][0], torch.tensor([2.0, -1]))
        loss.backward()
        optimizer.step()

        self.assertAllClose(parameters[0]["params"][0], torch.tensor(1.5))
        self.assertAllClose(parameters[1]["params"][0], torch.tensor([0.0, 0.0]))
