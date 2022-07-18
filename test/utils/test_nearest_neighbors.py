#!/usr/bin/env python3

import unittest

import torch

from gpytorch.utils.nearest_neighbors import NNUtil


class TestNNUtil(unittest.TestCase):
    def _setup(self, k, dim, batch_shape, train_n, test_n, preferred_nnlib, seq_train_n=None):
        # initialize
        nn_util = NNUtil(k=k, dim=dim, batch_shape=batch_shape, preferred_nnlib=preferred_nnlib)

        self._run(nn_util, dim, batch_shape, train_n, test_n, seq_train_n)
        return nn_util

    def _run(self, nn_util, dim, batch_shape, train_n, test_n, seq_train_n=None):
        train_x = torch.randn(*batch_shape, train_n, dim)
        test_x = torch.randn(*batch_shape, test_n, dim)
        k = nn_util.k

        # set training data
        nn_util.set_nn_idx(train_x)

        # test nn search
        test_nn_indices = nn_util.find_nn_idx(test_x)
        self.assertEqual(test_nn_indices.shape, (*batch_shape, test_n, k))

        # build sequential nn structure
        if seq_train_n is None:
            seq_train_x = train_x
            seq_train_n = train_n
        else:
            seq_train_x = torch.randn(*batch_shape, seq_train_n, dim)
        sequential_nn_idx = nn_util.build_sequential_nn_idx(seq_train_x)
        self.assertEqual(sequential_nn_idx.shape, (*batch_shape, seq_train_n - k, k))

    def test_setup(self):
        train_n = 10
        test_n = 5
        dim = 2
        k = 3

        new_train_n = 20
        new_test_n = 15

        for preferred_nnlib in ["sklearn", "faiss"]:
            nn_util = self._setup(
                k=k,
                dim=dim,
                batch_shape=torch.Size([]),
                train_n=train_n,
                test_n=test_n,
                preferred_nnlib=preferred_nnlib,
            )
            self._run(nn_util, dim=dim, batch_shape=nn_util.batch_shape, train_n=new_train_n, test_n=new_test_n)

            self._setup(
                k=k,
                dim=dim,
                batch_shape=torch.Size([2]),
                train_n=train_n,
                test_n=test_n,
                preferred_nnlib=preferred_nnlib,
            )
            self._run(nn_util, dim=dim, batch_shape=nn_util.batch_shape, train_n=new_train_n, test_n=new_test_n)

            self._setup(
                k=k,
                dim=dim,
                batch_shape=torch.Size([2, 5, 10]),
                train_n=train_n,
                test_n=test_n,
                seq_train_n=25,
                preferred_nnlib=preferred_nnlib,
            )
            self._run(
                nn_util,
                dim=dim,
                batch_shape=nn_util.batch_shape,
                train_n=new_train_n,
                test_n=new_test_n,
                seq_train_n=30,
            )

    def test_1Dtoycase(self):
        for preferred_nnlib in ["sklearn", "faiss"]:
            train_x = torch.arange(5, dtype=torch.float32).unsqueeze(-1)

            nn_util = NNUtil(k=3, dim=train_x.size(-1), preferred_nnlib=preferred_nnlib)

            # set train data
            nn_util.set_nn_idx(train_x)
            nn_idx = nn_util.find_nn_idx(train_x)
            nn_idx = torch.sort(nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 4], [2, 3, 4]], dtype=torch.int64)
            assert torch.equal(
                nn_idx, expected_idx
            ), f"Preferred_nnlib = {preferred_nnlib}, nn_idx={nn_idx}, expected_idx={expected_idx}"

            # single point test
            test_x = torch.tensor([20], dtype=torch.float32)  # shape would be automatically expanded
            nn_idx = nn_util.find_nn_idx(test_x)
            nn_idx = torch.sort(nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[2, 3, 4]], dtype=torch.int64)
            assert torch.equal(nn_idx, expected_idx)

            # multiple point test
            test_x = torch.tensor([1.6, 3], dtype=torch.float32)  # shape would be automatically expanded
            nn_idx = nn_util.find_nn_idx(test_x)
            nn_idx = torch.sort(nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int64)
            assert torch.equal(nn_idx, expected_idx)

            # with less k
            nn_idx = nn_util.find_nn_idx(test_x, k=1)
            expected_idx = torch.tensor([[2], [3]], dtype=torch.int64)
            assert torch.equal(nn_idx, expected_idx)

            # build sequential nn idx
            seq_nn_idx = nn_util.build_sequential_nn_idx(train_x)
            seq_nn_idx = torch.sort(seq_nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
            assert torch.equal(expected_idx, seq_nn_idx)

            # reset
            # set new train data
            train_x = torch.tensor([5.0, 3, 1, 4]).unsqueeze(-1)
            nn_util.set_nn_idx(train_x)

            test_x = torch.tensor([2.0, 10])
            nn_idx = nn_util.find_nn_idx(test_x)
            nn_idx = torch.sort(nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[1, 2, 3], [0, 1, 3]], dtype=torch.int64)
            assert torch.equal(nn_idx, expected_idx)

            seq_train_x = torch.tensor([10.0, 2, 5, 6, 9]).unsqueeze(-1)
            seq_nn_idx = nn_util.build_sequential_nn_idx(seq_train_x)
            seq_nn_idx = torch.sort(seq_nn_idx, dim=-1)[0]
            expected_idx = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
            assert torch.equal(seq_nn_idx, expected_idx)


if __name__ == "__main__":
    unittest.main()
