#!/usr/bin/env python3

import unittest

import torch

from gpytorch.test.utils import approx_equal
from gpytorch.utils.interpolation import Interpolation, left_interp, left_t_interp


class TestCubicInterpolation(unittest.TestCase):
    def test_interpolation(self):
        x = torch.linspace(0.01, 1, 100).unsqueeze(1)
        grid = torch.linspace(-0.05, 1.05, 50).unsqueeze(1)
        indices, values = Interpolation().interpolate(grid, x)
        indices = indices.squeeze_(0)
        values = values.squeeze_(0)
        test_func_grid = grid.squeeze(1).pow(2)
        test_func_x = x.pow(2).squeeze(-1)

        interp_func_x = left_interp(indices, values, test_func_grid.unsqueeze(1)).squeeze()

        self.assertTrue(approx_equal(interp_func_x, test_func_x))

    def test_multidim_interpolation(self):
        x = torch.tensor([[0.25, 0.45, 0.65, 0.85], [0.35, 0.375, 0.4, 0.425], [0.45, 0.5, 0.55, 0.6]]).t().contiguous()
        grid = torch.linspace(0.0, 1.0, 11).unsqueeze(1).repeat(1, 3)

        indices, values = Interpolation().interpolate(grid, x)

        actual_indices = torch.cat(
            [
                torch.tensor(
                    [
                        [146, 147, 148, 149, 157, 158, 159, 160, 168, 169, 170, 171, 179],
                        [389, 390, 391, 392, 400, 401, 402, 403, 411, 412, 413, 414, 422],
                        [642, 643, 644, 645, 653, 654, 655, 656, 664, 665, 666, 667, 675],
                        [885, 886, 887, 888, 896, 897, 898, 899, 907, 908, 909, 910, 918],
                    ],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [
                        [180, 181, 182, 267, 268, 269, 270, 278, 279, 280, 281, 289, 290],
                        [423, 424, 425, 510, 511, 512, 513, 521, 522, 523, 524, 532, 533],
                        [676, 677, 678, 763, 764, 765, 766, 774, 775, 776, 777, 785, 786],
                        [919, 920, 921, 1006, 1007, 1008, 1009, 1017, 1018, 1019, 1020, 1028, 1029],
                    ],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [
                        [291, 292, 300, 301, 302, 303, 388, 389, 390, 391, 399, 400, 401],
                        [534, 535, 543, 544, 545, 546, 631, 632, 633, 634, 642, 643, 644],
                        [787, 788, 796, 797, 798, 799, 884, 885, 886, 887, 895, 896, 897],
                        [1030, 1031, 1039, 1040, 1041, 1042, 1127, 1128, 1129, 1130, 1138, 1139, 1140],
                    ],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [
                        [402, 410, 411, 412, 413, 421, 422, 423, 424, 509, 510, 511, 512],
                        [645, 653, 654, 655, 656, 664, 665, 666, 667, 752, 753, 754, 755],
                        [898, 906, 907, 908, 909, 917, 918, 919, 920, 1005, 1006, 1007, 1008],
                        [1141, 1149, 1150, 1151, 1152, 1160, 1161, 1162, 1163, 1248, 1249, 1250, 1251],
                    ],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [
                        [520, 521, 522, 523, 531, 532, 533, 534, 542, 543, 544, 545],
                        [763, 764, 765, 766, 774, 775, 776, 777, 785, 786, 787, 788],
                        [1016, 1017, 1018, 1019, 1027, 1028, 1029, 1030, 1038, 1039, 1040, 1041],
                        [1259, 1260, 1261, 1262, 1270, 1271, 1272, 1273, 1281, 1282, 1283, 1284],
                    ],
                    dtype=torch.long,
                ),
            ],
            1,
        )
        self.assertTrue(approx_equal(indices, actual_indices))

        actual_values = torch.cat(
            [
                torch.tensor(
                    [
                        [-0.0002, 0.0022, 0.0022, -0.0002, 0.0022, -0.0198, -0.0198, 0.0022, 0.0022, -0.0198],
                        [0.0000, 0.0015, 0.0000, 0.0000, -0.0000, -0.0142, -0.0000, -0.0000, -0.0000, -0.0542],
                        [0.0000, -0.0000, -0.0000, 0.0000, 0.0039, -0.0352, -0.0352, 0.0039, 0.0000, -0.0000],
                        [0.0000, 0.0044, 0.0000, 0.0000, -0.0000, -0.0542, -0.0000, -0.0000, -0.0000, -0.0142],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0198, 0.0022, -0.0002, 0.0022, 0.0022, -0.0002, 0.0022, -0.0198, -0.0198, 0.0022],
                        [-0.0000, -0.0000, 0.0000, 0.0044, 0.0000, 0.0000, -0.0000, -0.0132, -0.0000, -0.0000],
                        [-0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0000, 0.0000, -0.0000],
                        [-0.0000, -0.0000, 0.0000, 0.0015, 0.0000, 0.0000, -0.0000, -0.0396, -0.0000, -0.0000],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0198, 0.1780, 0.1780, -0.0198, -0.0198, 0.1780, 0.1780, -0.0198, 0.0022, -0.0198],
                        [0.0000, 0.1274, 0.0000, 0.0000, 0.0000, 0.4878, 0.0000, 0.0000, -0.0000, -0.0396],
                        [-0.0352, 0.3164, 0.3164, -0.0352, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000],
                        [0.0000, 0.4878, 0.0000, 0.0000, 0.0000, 0.1274, 0.0000, 0.0000, -0.0000, -0.0132],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0198, 0.0022, 0.0022, -0.0198, -0.0198, 0.0022, -0.0198, 0.1780, 0.1780, -0.0198],
                        [-0.0000, -0.0000, -0.0000, -0.0132, -0.0000, -0.0000, 0.0000, 0.1274, 0.0000, 0.0000],
                        [0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0352, 0.3164, 0.3164, -0.0352],
                        [-0.0000, -0.0000, -0.0000, -0.0396, -0.0000, -0.0000, 0.0000, 0.4878, 0.0000, 0.0000],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0198, 0.1780, 0.1780, -0.0198, 0.0022, -0.0198, -0.0198, 0.0022, -0.0002, 0.0022],
                        [0.0000, 0.4878, 0.0000, 0.0000, -0.0000, -0.0396, -0.0000, -0.0000, 0.0000, 0.0015],
                        [-0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000],
                        [0.0000, 0.1274, 0.0000, 0.0000, -0.0000, -0.0132, -0.0000, -0.0000, 0.0000, 0.0044],
                    ]
                ),
                torch.tensor(
                    [
                        [0.0022, -0.0002, 0.0022, -0.0198, -0.0198, 0.0022, 0.0022, -0.0198, -0.0198, 0.0022],
                        [0.0000, 0.0000, -0.0000, -0.0142, -0.0000, -0.0000, -0.0000, -0.0542, -0.0000, -0.0000],
                        [-0.0000, 0.0000, 0.0039, -0.0352, -0.0352, 0.0039, 0.0000, -0.0000, -0.0000, 0.0000],
                        [0.0000, 0.0000, -0.0000, -0.0542, -0.0000, -0.0000, -0.0000, -0.0142, -0.0000, -0.0000],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0002, 0.0022, 0.0022, -0.0002],
                        [0.0000, 0.0044, 0.0000, 0.0000],
                        [0.0000, -0.0000, -0.0000, 0.0000],
                        [0.0000, 0.0015, 0.0000, 0.0000],
                    ]
                ),
            ],
            1,
        )
        self.assertTrue(approx_equal(values, actual_values))


class TestInterp(unittest.TestCase):
    def setUp(self):
        self.interp_indices = torch.tensor([[2, 3], [3, 4], [4, 5]], dtype=torch.long).repeat(3, 1)
        self.interp_values = torch.tensor([[1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(3, 1)
        self.interp_indices_2 = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).repeat(3, 1)
        self.interp_values_2 = torch.tensor([[1, 2], [2, 0.5], [1, 3]], dtype=torch.float).repeat(3, 1)
        self.batch_interp_indices = torch.cat([self.interp_indices.unsqueeze(0), self.interp_indices_2.unsqueeze(0)], 0)
        self.batch_interp_values = torch.cat([self.interp_values.unsqueeze(0), self.interp_values_2.unsqueeze(0)], 0)
        self.interp_matrix = torch.tensor(
            [
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
            ],
            dtype=torch.float,
        )

        self.batch_interp_matrix = torch.tensor(
            [
                [
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                ],
                [
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                ],
            ],
            dtype=torch.float,
        )

    def test_left_interp_on_a_vector(self):
        vector = torch.randn(6)

        res = left_interp(self.interp_indices, self.interp_values, vector)
        actual = torch.matmul(self.interp_matrix, vector)
        self.assertTrue(approx_equal(res, actual))

    def test_left_t_interp_on_a_vector(self):
        vector = torch.randn(9)

        res = left_t_interp(self.interp_indices, self.interp_values, vector, 6)
        actual = torch.matmul(self.interp_matrix.transpose(-1, -2), vector)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_vector(self):
        vector = torch.randn(6)

        actual = torch.matmul(self.batch_interp_matrix, vector.unsqueeze(-1).unsqueeze(0)).squeeze(-1)
        res = left_interp(self.batch_interp_indices, self.batch_interp_values, vector)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_vector(self):
        vector = torch.randn(9)

        actual = torch.matmul(self.batch_interp_matrix.transpose(-1, -2), vector.unsqueeze(-1).unsqueeze(0)).squeeze(-1)
        res = left_t_interp(self.batch_interp_indices, self.batch_interp_values, vector, 6)
        self.assertTrue(approx_equal(res, actual))

    def test_left_interp_on_a_matrix(self):
        matrix = torch.randn(6, 3)

        res = left_interp(self.interp_indices, self.interp_values, matrix)
        actual = torch.matmul(self.interp_matrix, matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_left_t_interp_on_a_matrix(self):
        matrix = torch.randn(9, 3)

        res = left_t_interp(self.interp_indices, self.interp_values, matrix, 6)
        actual = torch.matmul(self.interp_matrix.transpose(-1, -2), matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_matrix(self):
        batch_matrix = torch.randn(6, 3)

        res = left_interp(self.batch_interp_indices, self.batch_interp_values, batch_matrix)
        actual = torch.matmul(self.batch_interp_matrix, batch_matrix.unsqueeze(0))
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_matrix(self):
        batch_matrix = torch.randn(9, 3)

        res = left_t_interp(self.batch_interp_indices, self.batch_interp_values, batch_matrix, 6)
        actual = torch.matmul(self.batch_interp_matrix.transpose(-1, -2), batch_matrix.unsqueeze(0))
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_batch_matrix(self):
        batch_matrix = torch.randn(2, 6, 3)

        res = left_interp(self.batch_interp_indices, self.batch_interp_values, batch_matrix)
        actual = torch.matmul(self.batch_interp_matrix, batch_matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_batch_matrix(self):
        batch_matrix = torch.randn(2, 9, 3)

        res = left_t_interp(self.batch_interp_indices, self.batch_interp_values, batch_matrix, 6)
        actual = torch.matmul(self.batch_interp_matrix.transpose(-1, -2), batch_matrix)
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
