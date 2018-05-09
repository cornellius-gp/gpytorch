from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import MulLazyVariable, RootLazyVariable
from gpytorch.utils import prod


def make_random_mat(size, rank, batch_size=None):
    if batch_size is None:
        res = torch.randn(size, rank)
    else:
        res = torch.randn(batch_size, size, rank)
    return Variable(res, requires_grad=True)


class TestMulLazyVariable(unittest.TestCase):

    def setUp(self):
        if (
            os.getenv("UNLOCK_SEED") is None
            or os.getenv("UNLOCK_SEED").lower() == "false"
        ):
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(2)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_vec_with_two_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        vec = Variable(torch.randn(20), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_matmul_vec_with_five_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        mat3 = make_random_mat(20, 5)
        mat4 = make_random_mat(20, 5)
        mat5 = make_random_mat(20, 5)
        vec = Variable(torch.randn(20), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)
        mat4_copy = Variable(mat4.data, requires_grad=True)
        mat5_copy = Variable(mat5.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1),
            RootLazyVariable(mat2),
            RootLazyVariable(mat3),
            RootLazyVariable(mat4),
            RootLazyVariable(mat5),
        ).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_matmul_mat_with_two_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        vec = Variable(torch.randn(20, 7), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_matmul_mat_with_five_matrices(self):
        mat1 = make_random_mat(20, 5)
        mat2 = make_random_mat(20, 5)
        mat3 = make_random_mat(20, 5)
        mat4 = make_random_mat(20, 5)
        mat5 = make_random_mat(20, 5)
        vec = Variable(torch.eye(20), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)
        mat4_copy = Variable(mat4.data, requires_grad=True)
        mat5_copy = Variable(mat5.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1),
            RootLazyVariable(mat2),
            RootLazyVariable(mat3),
            RootLazyVariable(mat4),
            RootLazyVariable(mat5),
        ).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_batch_matmul_mat_with_two_matrices(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        vec = Variable(torch.randn(5, 20, 7), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2)).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_batch_matmul_mat_with_five_matrices(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)
        mat4 = make_random_mat(20, rank=4, batch_size=5)
        mat5 = make_random_mat(20, rank=4, batch_size=5)
        vec = Variable(torch.randn(5, 20, 7), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)
        mat4_copy = Variable(mat4.data, requires_grad=True)
        mat5_copy = Variable(mat5.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1),
            RootLazyVariable(mat2),
            RootLazyVariable(mat3),
            RootLazyVariable(mat4),
            RootLazyVariable(mat5),
        ).matmul(
            vec
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
                mat4_copy.matmul(mat4_copy.transpose(-1, -2)),
                mat5_copy.matmul(mat5_copy.transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mat1.grad.data - mat1_copy.grad.data) / mat1_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat2.grad.data - mat2_copy.grad.data) / mat2_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat3.grad.data - mat3_copy.grad.data) / mat3_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat4.grad.data - mat4_copy.grad.data) / mat4_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((mat5.grad.data - mat5_copy.grad.data) / mat5_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_mul_adding_another_variable(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(RootLazyVariable(mat1), RootLazyVariable(mat2))
        res = res * RootLazyVariable(mat3)
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )
        self.assertLess(
            torch.max(((res.evaluate().data - actual.data) / actual.data).abs()), 0.01
        )

    def test_mul_adding_constant_mul(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)
        const = Variable(torch.ones(1), requires_grad=True)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)
        const_copy = Variable(const.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        )
        res = res * const
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        ) * const_copy
        self.assertLess(
            torch.max(((res.evaluate().data - actual.data) / actual.data).abs()), 0.01
        )

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        )
        res = res * 2.5
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        ) * 2.5
        self.assertLess(
            torch.max(((res.evaluate().data - actual.data) / actual.data).abs()), 0.01
        )

    def test_diag(self):
        mat1 = make_random_mat(20, rank=4)
        mat2 = make_random_mat(20, rank=4)
        mat3 = make_random_mat(20, rank=4)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        ).diag()
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        ).diag()
        assert torch.max(((res.data - actual.data) / actual.data).abs()) < 0.01

    def test_batch_diag(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        ).diag()
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(5)])
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

    def test_getitem(self):
        mat1 = make_random_mat(20, rank=4)
        mat2 = make_random_mat(20, rank=4)
        mat3 = make_random_mat(20, rank=4)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )

        self.assertLess(
            torch.max(
                ((res[5, 3:5].data - actual[5, 3:5].data) / actual[5, 3:5].data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                (
                    (res[3:5, 2:].evaluate().data - actual[3:5, 2:].data)
                    / actual[3:5, 2:].data
                ).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                (
                    (res[2:, 3:5].evaluate().data - actual[2:, 3:5].data)
                    / actual[2:, 3:5].data
                ).abs()
            ),
            0.01,
        )

    def test_batch_getitem(self):
        mat1 = make_random_mat(20, rank=4, batch_size=5)
        mat2 = make_random_mat(20, rank=4, batch_size=5)
        mat3 = make_random_mat(20, rank=4, batch_size=5)

        mat1_copy = Variable(mat1.data, requires_grad=True)
        mat2_copy = Variable(mat2.data, requires_grad=True)
        mat3_copy = Variable(mat3.data, requires_grad=True)

        # Forward
        res = MulLazyVariable(
            RootLazyVariable(mat1), RootLazyVariable(mat2), RootLazyVariable(mat3)
        )
        actual = prod(
            [
                mat1_copy.matmul(mat1_copy.transpose(-1, -2)),
                mat2_copy.matmul(mat2_copy.transpose(-1, -2)),
                mat3_copy.matmul(mat3_copy.transpose(-1, -2)),
            ]
        )

        self.assertLess(
            torch.max(
                ((res[0].evaluate().data - actual[0].data) / actual[0].data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                (
                    (res[0:2, 5, 3:5].data - actual[0:2, 5, 3:5].data)
                    / actual[0:2, 5, 3:5].data
                ).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                (
                    (res[:, 3:5, 2:].evaluate().data - actual[:, 3:5, 2:].data)
                    / actual[:, 3:5, 2:].data
                ).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                (
                    (res[:, 2:, 3:5].evaluate().data - actual[:, 2:, 3:5].data)
                    / actual[:, 2:, 3:5].data
                ).abs()
            ),
            0.01,
        )

    def test_batch_mode_matmul_mat_with_five_matrices(self):
        mats = make_random_mat(6, rank=4, batch_size=6)
        vec = Variable(torch.randn(6, 7), requires_grad=True)

        mats_copy = Variable(mats.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = RootLazyVariable(mats).mul_batch().matmul(vec)
        actual = prod(
            [
                mats_copy[0].matmul(mats_copy[0].transpose(-1, -2)),
                mats_copy[1].matmul(mats_copy[1].transpose(-1, -2)),
                mats_copy[2].matmul(mats_copy[2].transpose(-1, -2)),
                mats_copy[3].matmul(mats_copy[3].transpose(-1, -2)),
                mats_copy[4].matmul(mats_copy[4].transpose(-1, -2)),
                mats_copy[5].matmul(mats_copy[5].transpose(-1, -2)),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mats.grad.data - mats_copy.grad.data) / mats_copy.grad.data).abs()
            ),
            0.01,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.01,
        )

    def test_batch_mode_matmul_batch_mat_with_five_matrices(self):
        mats = make_random_mat(6, rank=4, batch_size=30)
        vec = Variable(torch.randn(5, 6, 7), requires_grad=True)

        mats_copy = Variable(mats.data, requires_grad=True)
        vec_copy = Variable(vec.data, requires_grad=True)

        # Forward
        res = RootLazyVariable(mats).mul_batch(mul_batch_size=6).matmul(vec)
        reshaped_mats_copy = mats_copy.view(5, 6, 6, 4)
        actual = prod(
            [
                (
                    reshaped_mats_copy[:, 0].matmul(
                        reshaped_mats_copy[:, 0].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
                (
                    reshaped_mats_copy[:, 1].matmul(
                        reshaped_mats_copy[:, 1].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
                (
                    reshaped_mats_copy[:, 2].matmul(
                        reshaped_mats_copy[:, 2].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
                (
                    reshaped_mats_copy[:, 3].matmul(
                        reshaped_mats_copy[:, 3].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
                (
                    reshaped_mats_copy[:, 4].matmul(
                        reshaped_mats_copy[:, 4].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
                (
                    reshaped_mats_copy[:, 5].matmul(
                        reshaped_mats_copy[:, 5].transpose(-1, -2)
                    ).view(
                        5, 6, 6
                    )
                ),
            ]
        ).matmul(
            vec_copy
        )
        self.assertLess(torch.max(((res.data - actual.data) / actual.data).abs()), 0.01)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(
            torch.max(
                ((mats.grad.data - mats_copy.grad.data) / mats_copy.grad.data).abs()
            ),
            0.05,
        )
        self.assertLess(
            torch.max(
                ((vec.grad.data - vec_copy.grad.data) / vec_copy.grad.data).abs()
            ),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
