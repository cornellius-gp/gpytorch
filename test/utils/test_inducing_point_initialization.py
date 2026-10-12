#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import unittest

import torch

from gpytorch.utils.inducing_point_initialization import kmeans_inducing_points, median_heuristic_lengthscale


class TestKmeansInducingPoints(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_basic(self):
        train_x = torch.randn(200, 5)
        inducing = kmeans_inducing_points(train_x, n_inducing=20, seed=42)
        self.assertEqual(inducing.shape, torch.Size([20, 5]))
        self.assertEqual(inducing.dtype, train_x.dtype)
        self.assertEqual(inducing.device, train_x.device)

    def test_finds_clusters(self):
        cluster1 = torch.randn(200, 2) + torch.tensor([10.0, 10.0])
        cluster2 = torch.randn(200, 2) + torch.tensor([-10.0, -10.0])
        train_x = torch.cat([cluster1, cluster2], dim=0)

        inducing = kmeans_inducing_points(train_x, n_inducing=2, seed=42)

        center1_dists = (inducing - torch.tensor([10.0, 10.0])).norm(dim=1)
        center2_dists = (inducing - torch.tensor([-10.0, -10.0])).norm(dim=1)
        self.assertLess(center1_dists.min().item(), 2.0)
        self.assertLess(center2_dists.min().item(), 2.0)

    def test_minibatch(self):
        train_x = torch.randn(500, 3)
        inducing = kmeans_inducing_points(train_x, n_inducing=20, batch_size=100, seed=42)
        self.assertEqual(inducing.shape, torch.Size([20, 3]))

    def test_n_inducing_exceeds_data(self):
        train_x = torch.randn(10, 3)
        with self.assertWarns(UserWarning):
            inducing = kmeans_inducing_points(train_x, n_inducing=20, seed=42)
        self.assertEqual(inducing.shape, train_x.shape)

    def test_reproducibility(self):
        train_x = torch.randn(200, 5)
        inducing1 = kmeans_inducing_points(train_x, n_inducing=20, seed=42)
        inducing2 = kmeans_inducing_points(train_x, n_inducing=20, seed=42)
        self.assertTrue(torch.allclose(inducing1, inducing2))

    def test_float64(self):
        train_x = torch.randn(100, 3, dtype=torch.float64)
        inducing = kmeans_inducing_points(train_x, n_inducing=10, seed=42)
        self.assertEqual(inducing.dtype, torch.float64)


class TestMedianHeuristicLengthscale(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_basic(self):
        train_x = torch.randn(200, 3)
        ls = median_heuristic_lengthscale(train_x)
        self.assertEqual(ls.ndim, 0)
        self.assertGreater(ls.item(), 0.0)
        self.assertEqual(ls.dtype, train_x.dtype)

    def test_scales_with_spread(self):
        train_narrow = torch.randn(500, 3) * 0.1
        train_wide = torch.randn(500, 3) * 10.0
        ls_narrow = median_heuristic_lengthscale(train_narrow, seed=42)
        ls_wide = median_heuristic_lengthscale(train_wide, seed=42)
        self.assertGreater(ls_wide.item(), ls_narrow.item())

    def test_subsample(self):
        train_x = torch.randn(5000, 3)
        ls = median_heuristic_lengthscale(train_x, n_subsample=100, seed=42)
        self.assertGreater(ls.item(), 0.0)

    def test_single_point(self):
        train_x = torch.randn(1, 3)
        with self.assertWarns(UserWarning):
            ls = median_heuristic_lengthscale(train_x)
        self.assertAlmostEqual(ls.item(), 1.0)

    def test_reproducibility(self):
        train_x = torch.randn(5000, 5)
        ls1 = median_heuristic_lengthscale(train_x, n_subsample=200, seed=42)
        ls2 = median_heuristic_lengthscale(train_x, n_subsample=200, seed=42)
        self.assertEqual(ls1.item(), ls2.item())

    def test_known_value(self):
        # For [0, 1, ..., 99], median pairwise distance should match torch.pdist
        train_x = torch.arange(100, dtype=torch.float32).unsqueeze(1)
        ls = median_heuristic_lengthscale(train_x)
        expected = torch.pdist(train_x).median().item()
        self.assertAlmostEqual(ls.item(), expected, places=4)


if __name__ == "__main__":
    unittest.main()
