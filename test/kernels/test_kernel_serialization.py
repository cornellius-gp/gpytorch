#!/usr/bin/env python3

import copy
import pickle
import unittest

import torch

from gpytorch.kernels import RBFKernel, ScaleKernel


class TestKernelSerialization(unittest.TestCase):
    def test_shallow_copy_owns_python_state_and_shares_parameters(self):
        kernel = ScaleKernel(RBFKernel()).double()
        copied = copy.copy(kernel)
        self.assertIsNot(copied.__dict__, kernel.__dict__)
        copied.marker = "copy"
        self.assertFalse(hasattr(kernel, "marker"))
        self.assertIs(copied.base_kernel, kernel.base_kernel)
        self.assertIs(copied.raw_outputscale, kernel.raw_outputscale)
        self.assertIs(copied.raw_outputscale_constraint, kernel.raw_outputscale_constraint)

    def test_serializing_does_not_clear_live_distance_module(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for serialize in (copy.copy, copy.deepcopy, lambda k: pickle.loads(pickle.dumps(k))):
                with self.subTest(device=device, serialize=serialize):
                    kernel = RBFKernel().to(device=device, dtype=torch.double)
                    distance = torch.jit.trace(
                        torch.nn.Linear(1, 1).to(device=device, dtype=torch.double),
                        torch.ones(1, 1, device=device, dtype=torch.double),
                    )
                    kernel.distance_module = distance
                    x = torch.tensor([[0.1], [0.4], [0.8]], device=device, dtype=torch.double)
                    expected = kernel(x).to_dense()
                    restored = serialize(kernel)
                    self.assertIs(kernel.distance_module, distance)
                    self.assertIsNone(restored.distance_module)
                    torch.testing.assert_close(restored(x).to_dense(), expected)


if __name__ == "__main__":
    unittest.main()
