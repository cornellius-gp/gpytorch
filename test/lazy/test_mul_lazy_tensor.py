from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
from gpytorch.lazy import MulLazyTensor, RootLazyTensor
from gpytorch.utils import prod


def make_random_mat(size, rank, batch_size=None):
    if batch_size is None:
        return torch.randn(size, rank, requires_grad=True)
    else:
        return torch.randn(batch_size, size, rank, requires_grad=True)


class TestMulLazyTensor(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(2)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_vec_with_two_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        vec = torch.randn(20, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2)).matmul(vec)
        actual = prod(
            [mat1_copy.matmul(mat1_copy.transpose(-1, -2)), mat2_copy.matmul(mat2_copy.transpose(-1, -2))]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_matmul_vec_with_five_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        mat3 = make_random_mat(20, 5)
        mat4 = make_random_mat(20, 5)
        mat5 = make_random_mat(20, 5)
        vec = torch.randn(20, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)
        mat4_copy = mat4.clone().detach().requires_grad_(True)
        mat5_copy = mat5.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(
            RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3), RootLazyTensor(mat4), RootLazyTensor(mat5)
        ).matmul(vec)
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat3.grad - mat3_copy.grad) / mat3_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat4.grad - mat4_copy.grad) / mat4_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat5.grad - mat5_copy.grad) / mat5_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_matmul_mat_with_two_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        vec = torch.randn(20, 7, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2)).matmul(vec)
        actual = prod(
            [mat1_copy.matmul(mat1_copy.transpose(-1, -2)), mat2_copy.matmul(mat2_copy.transpose(-1, -2))]
        ).matmul(vec_copy)
        assert torch.max(((res - actual) / actual).abs()) < 0.01

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_matmul_mat_with_five_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        mat3 = make_random_mat(20, 5)
        mat4 = make_random_mat(20, 5)
        mat5 = make_random_mat(20, 5)
        vec = torch.eye(20, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)
        mat4_copy = mat4.clone().detach().requires_grad_(True)
        mat5_copy = mat5.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(
            RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3), RootLazyTensor(mat4), RootLazyTensor(mat5)
        ).matmul(vec)
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat3.grad - mat3_copy.grad) / mat3_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat4.grad - mat4_copy.grad) / mat4_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat5.grad - mat5_copy.grad) / mat5_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_batch_matmul_mat_with_two_matrices(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        vec = torch.randn(5, 20, 7, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2)).matmul(vec)
        actual = prod(
            [mat1_copy.matmul(mat1_copy.transpose(-1, -2)), mat2_copy.matmul(mat2_copy.transpose(-1, -2))]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_batch_matmul_mat_with_five_matrices(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)
        mat4 = make_random_mat(20, rank=4, batch_size=5)
        mat5 = make_random_mat(20, rank=4, batch_size=5)
        vec = torch.randn(5, 20, 7, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)
        mat4_copy = mat4.clone().detach().requires_grad_(True)
        mat5_copy = mat5.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(
            RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3), RootLazyTensor(mat4), RootLazyTensor(mat5)
        ).matmul(vec)
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mat1.grad - mat1_copy.grad) / mat1_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat2.grad - mat2_copy.grad) / mat2_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat3.grad - mat3_copy.grad) / mat3_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat4.grad - mat4_copy.grad) / mat4_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((mat5.grad - mat5_copy.grad) / mat5_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_mul_adding_another_variable(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2))
        res = res * RootLazyTensor(mat3)
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )
        self.assertLess(torch.max(((res.evaluate() - actual) / actual).abs()), 0.01)

    def test_mul_adding_constant_mul(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)
        const = torch.ones(1, requires_grad=True)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)
        const_copy = const.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3))
        res = res * const
        actual = (
            prod(
                [
                    mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                    mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                    mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                ]
            )
            * const_copy
        )
        self.assertLess(torch.max(((res.evaluate() - actual) / actual).abs()), 0.01)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3))
        res = res * 2.5
        actual = (
            prod(
                [
                    mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                    mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                    mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                ]
            )
            * 2.5
        )
        self.assertLess(torch.max(((res.evaluate() - actual) / actual).abs()), 0.01)

    def test_diag(self):
        mat1 = make_random_mat(20, rank=4)
        mat2 = make_random_mat(20, rank=4)
        mat3 = make_random_mat(20, rank=4)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3)).diag()
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        ).diag()
        assert torch.max(((res - actual) / actual).abs()) < 0.01

    def test_batch_diag(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3)).diag()
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(5)])
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

    def test_getitem(self):
        mat1 = make_random_mat(20, rank=4)
        mat2 = make_random_mat(20, rank=4)
        mat3 = make_random_mat(20, rank=4)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3))
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )

        self.assertLess(torch.max(((res[5, 3:5] - actual[5, 3:5]) / actual[5, 3:5]).abs()), 0.01)
        self.assertLess(torch.max(((res[3:5, 2:].evaluate() - actual[3:5, 2:]) / actual[3:5, 2:]).abs()), 0.01)
        self.assertLess(torch.max(((res[2:, 3:5].evaluate() - actual[2:, 3:5]) / actual[2:, 3:5]).abs()), 0.01)

    def test_batch_getitem(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = mat1.clone().detach().requires_grad_(True)
        mat2_copy = mat2.clone().detach().requires_grad_(True)
        mat3_copy = mat3.clone().detach().requires_grad_(True)

        # Forward
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3))
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )

        self.assertLess(torch.max(((res[0].evaluate() - actual[0]) / actual[0]).abs()), 0.01)
        self.assertLess(torch.max(((res[0:2, 5, 3:5] - actual[0:2, 5, 3:5]) / actual[0:2, 5, 3:5]).abs()), 0.01)
        self.assertLess(torch.max(((res[:, 3:5, 2:].evaluate() - actual[:, 3:5, 2:]) / actual[:, 3:5, 2:]).abs()), 0.01)
        self.assertLess(torch.max(((res[:, 2:, 3:5].evaluate() - actual[:, 2:, 3:5]) / actual[:, 2:, 3:5]).abs()), 0.01)

    def test_batch_mode_matmul_mat_with_five_matrices(self):
        mats = make_random_mat(6, rank=4, batch_size=6)
        vec = torch.randn(6, 7, requires_grad=True)
        mats_copy = mats.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = RootLazyTensor(mats).mul_batch().matmul(vec)
        actual = prod(
            [
                mats_copy[0].matmul(mats_copy[0].transpose(-1, -2)),
                mats_copy[1].matmul(mats_copy[1].transpose(-1, -2)),
                mats_copy[2].matmul(mats_copy[2].transpose(-1, -2)),
                mats_copy[3].matmul(mats_copy[3].transpose(-1, -2)),
                mats_copy[4].matmul(mats_copy[4].transpose(-1, -2)),
                mats_copy[5].matmul(mats_copy[5].transpose(-1, -2)),
            ]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mats.grad - mats_copy.grad) / mats_copy.grad).abs()), 0.01)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.01)

    def test_batch_mode_matmul_batch_mat_with_five_matrices(self):
        mats = make_random_mat(6, rank=4, batch_size=30)
        vec = torch.randn(5, 6, 7, requires_grad=True)
        mats_copy = mats.clone().detach().requires_grad_(True)
        vec_copy = vec.clone().detach().requires_grad_(True)

        # Forward
        res = RootLazyTensor(mats).mul_batch(mul_batch_size=6).matmul(vec)
        reshaped_mats_copy = mats_copy.view(5, 6, 6, 4)
        actual = prod(
            [
                (reshaped_mats_copy[:, 0].matmul(reshaped_mats_copy[:, 0].transpose(-1, -2)).view(5, 6, 6)),
                (reshaped_mats_copy[:, 1].matmul(reshaped_mats_copy[:, 1].transpose(-1, -2)).view(5, 6, 6)),
                (reshaped_mats_copy[:, 2].matmul(reshaped_mats_copy[:, 2].transpose(-1, -2)).view(5, 6, 6)),
                (reshaped_mats_copy[:, 3].matmul(reshaped_mats_copy[:, 3].transpose(-1, -2)).view(5, 6, 6)),
                (reshaped_mats_copy[:, 4].matmul(reshaped_mats_copy[:, 4].transpose(-1, -2)).view(5, 6, 6)),
                (reshaped_mats_copy[:, 5].matmul(reshaped_mats_copy[:, 5].transpose(-1, -2)).view(5, 6, 6)),
            ]
        ).matmul(vec_copy)
        self.assertLess(torch.max(((res - actual) / actual).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.max(((mats.grad - mats_copy.grad) / mats_copy.grad).abs()), 0.05)
        self.assertLess(torch.max(((vec.grad - vec_copy.grad) / vec_copy.grad).abs()), 0.05)


if __name__ == "__main__":
    unittest.main()
