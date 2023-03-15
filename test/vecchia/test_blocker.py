import unittest

import torch

import pdb

import gpytorch
from gpytorch.vecchia import KMeansBlocker, VoronoiBlocker, DistanceMetrics


class TestBaseBlocker(unittest.TestCase):

    def test_blocks_template(self):

        for test_name, test_input, expected in zip(test_names, test_inputs, expected_outputs):
            with self.subTest(test_name):

                # correct number of blocks
                self.assertEqual(len(test_input._block_observations), expected["n_blocks"])

                # no blocks are empty
                self.assertFalse(any([block.nelement() == 0 for block in test_input._block_observations]))

                # every element in each block is unique
                self.assertTrue(all([len(block.unique()) == len(block) for block in test_input._block_observations]))

                # all blocks elements are of correct dimensions
                self.assertTrue(all([block.dim() == 1 for block in test_input._block_observations]))

                # no blocks overlap
                non_overlapping = []
                i = 1
                for block1 in test_input._block_observations:
                    for block2 in test_input._block_observations[i:len(test_input._block_observations)]:
                        non_overlapping.append(
                            len(torch.cat((block1, block2), -1).unique()) == len(torch.cat((block1, block2), -1))
                        )
                    i += 1
                self.assertTrue(all(non_overlapping))

    def test_create_ordered_neighbors(self):

        for test_name, test_input, expected in zip(test_names, test_inputs, expected_outputs):
            with self.subTest(test_name):

                # correct number of overall neighboring sets (should be equal to number of blocks, 1 set per block)
                self.assertEqual(len(test_input._exclusive_neighboring_observations), expected["n_blocks"])

                # correct number of neighbors per block
                correct_num_neighbors = torch.cat(
                    (torch.arange(0, expected["n_neighbors"]),
                     torch.tensor(expected["n_neighbors"]).repeat_interleave(
                         expected["n_blocks"] - expected["n_neighbors"]))
                )
                self.assertTrue(all([len(test_input._neighboring_blocks[i]) == correct_num_neighbors[i]
                                     for i in range(expected["n_blocks"])]))

                # all elements in neighboring sets are unique
                self.assertTrue(all([len(neighbor.unique()) == len(neighbor)
                                     for neighbor in test_input._exclusive_neighboring_observations]))

                # first neighboring set is empty, all others are empty if n_neighbors == 0
                if expected["n_neighbors"] == 0:
                    self.assertTrue(all([len(neighbor) == 0
                                         for neighbor in test_input._exclusive_neighboring_observations]))
                else:
                    self.assertTrue(len(test_input._exclusive_neighboring_observations[0]) == 0)

                # exclusive neighboring sets are identical to concatenated neighboring blocks
                if expected["n_neighbors"] != 0:
                    reconstructed_sorted_neighboring_sets = [
                        torch.sort(torch.cat([test_input._block_observations[block] for block in neighboring_set]))[0]
                        for neighboring_set in test_input._neighboring_blocks[1:]
                    ]
                    self.assertTrue(all(
                        [torch.equal(torch.sort(test_input._exclusive_neighboring_observations[i])[0].long(),
                                     reconstructed_sorted_neighboring_sets[i-1])
                         for i in range(1, expected["n_blocks"])]))

                # correct overall number of inclusive neighboring sets
                self.assertEqual(len(test_input._inclusive_neighboring_observations), expected["n_blocks"])

                # assert blocks._inclusive_neighboring_observations elements are unique
                self.assertTrue(all([len(neighbor.unique()) == len(neighbor)
                                     for neighbor in test_input._inclusive_neighboring_observations]))

                # inclusive neighboring sets are identical to blocks joined with their concatenated neighboring blocks
                reconstructed_sorted_neighboring_sets = [
                    torch.sort(torch.cat((*[test_input._block_observations[block] for block in neighboring_set],
                                            test_input._block_observations[i])))[0]
                    for neighboring_set, i in zip(test_input._neighboring_blocks, range(expected["n_blocks"]))
                ]
                self.assertTrue(all(
                    [torch.equal(torch.sort(test_input._inclusive_neighboring_observations[i])[0].long(),
                                 reconstructed_sorted_neighboring_sets[i])
                     for i in range(expected["n_blocks"])]))

    def test_reorder(self):
        # assert blocks._block_observations was reordered properly
        # retest self.test_create_ordered_neighbors
        ...


# class TestKMeansBlocker(TestBaseBlocker):
#    ...


# class TestVoronoiBlocker(TestBaseBlocker):
#    ...

data = torch.tensor([[x1, x2] for x1, x2 in zip(
    torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([5.0])).rsample(torch.tensor([100])),
    torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([5.0])).rsample(torch.tensor([100])))])
distance_metric = DistanceMetrics.euclidean_distance()

test_names = ["10 blocks, 0 neighbors",
              "10 blocks, 3 neighbors",
              "10 blocks, 9 neighbors",
              "100 blocks, 0 neighbors",
              "100 blocks, 20 neighbors",
              "100 blocks, 99 neighbors"]

test_inputs = [KMeansBlocker(data, n_blocks=10, n_neighbors=0, distance_metric=distance_metric),
               KMeansBlocker(data, n_blocks=10, n_neighbors=3, distance_metric=distance_metric),
               KMeansBlocker(data, n_blocks=10, n_neighbors=9, distance_metric=distance_metric),
               KMeansBlocker(data, n_blocks=100, n_neighbors=0, distance_metric=distance_metric),
               KMeansBlocker(data, n_blocks=100, n_neighbors=20, distance_metric=distance_metric),
               KMeansBlocker(data, n_blocks=100, n_neighbors=99, distance_metric=distance_metric)]

expected_outputs = [{"n_blocks": 10, "n_neighbors": 0},
                    {"n_blocks": 10, "n_neighbors": 3},
                    {"n_blocks": 10, "n_neighbors": 9},
                    {"n_blocks": 100, "n_neighbors": 0},
                    {"n_blocks": 100, "n_neighbors": 20},
                    {"n_blocks": 100, "n_neighbors": 99}]


if __name__ == "__main__":
    unittest.main()
