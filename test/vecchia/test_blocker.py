import copy
import unittest

import torch

from gpytorch.vecchia import DistanceMetrics, KMeansBlocker


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
                    for block2 in test_input._block_observations[i : len(test_input._block_observations)]:
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
                    (
                        torch.arange(0, expected["n_neighbors"]),
                        torch.tensor(expected["n_neighbors"]).repeat_interleave(
                            expected["n_blocks"] - expected["n_neighbors"]
                        ),
                    )
                )
                self.assertTrue(
                    all(
                        [
                            len(test_input._neighboring_blocks[i]) == correct_num_neighbors[i]
                            for i in range(expected["n_blocks"])
                        ]
                    )
                )

                # all elements in neighboring sets are unique
                self.assertTrue(
                    all(
                        [
                            len(neighbor.unique()) == len(neighbor)
                            for neighbor in test_input._exclusive_neighboring_observations
                        ]
                    )
                )

                # first neighboring set is empty, all others are empty if n_neighbors == 0
                if expected["n_neighbors"] == 0:
                    self.assertTrue(
                        all([len(neighbor) == 0 for neighbor in test_input._exclusive_neighboring_observations])
                    )
                else:
                    self.assertTrue(len(test_input._exclusive_neighboring_observations[0]) == 0)

                # exclusive neighboring sets are identical to concatenated neighboring blocks
                if expected["n_neighbors"] != 0:
                    reconstructed_sorted_neighboring_sets = [
                        torch.sort(torch.cat([test_input._block_observations[block] for block in neighboring_set]))[0]
                        for neighboring_set in test_input._neighboring_blocks[1:]
                    ]
                    self.assertTrue(
                        all(
                            [
                                torch.equal(
                                    torch.sort(test_input._exclusive_neighboring_observations[i])[0].long(),
                                    reconstructed_sorted_neighboring_sets[i - 1],
                                )
                                for i in range(1, expected["n_blocks"])
                            ]
                        )
                    )

                # correct overall number of inclusive neighboring sets
                self.assertEqual(len(test_input._inclusive_neighboring_observations), expected["n_blocks"])

                # assert blocks._inclusive_neighboring_observations elements are unique
                self.assertTrue(
                    all(
                        [
                            len(neighbor.unique()) == len(neighbor)
                            for neighbor in test_input._inclusive_neighboring_observations
                        ]
                    )
                )

                # inclusive neighboring sets are identical to blocks joined with their concatenated neighboring blocks
                reconstructed_sorted_neighboring_sets = [
                    torch.sort(
                        torch.cat(
                            (
                                *[test_input._block_observations[block] for block in neighboring_set],
                                test_input._block_observations[i],
                            )
                        )
                    )[0]
                    for neighboring_set, i in zip(test_input._neighboring_blocks, range(expected["n_blocks"]))
                ]
                self.assertTrue(
                    all(
                        [
                            torch.equal(
                                torch.sort(test_input._inclusive_neighboring_observations[i])[0].long(),
                                reconstructed_sorted_neighboring_sets[i],
                            )
                            for i in range(expected["n_blocks"])
                        ]
                    )
                )

    def test_reorder(self):
        def new_ordering_strategy(order):
            return lambda x: order

        for test_name, test_input, expected, new_order in zip(test_names, test_inputs, expected_outputs, new_orders):
            with self.subTest(test_name):

                # copy original block observations for basis of comparison
                block_observations_1 = copy.copy(test_input._block_observations)
                # reorder blocks
                test_input.reorder(new_ordering_strategy(new_order[0]))

                # ensure new blocks are ordered correctly and rerun all other tests
                self.assertTrue(
                    all(
                        [
                            torch.equal(test_input._block_observations[i], block_observations_1[j])
                            for i, j in zip(range(expected["n_blocks"]), new_order[0])
                        ]
                    )
                )
                self.test_blocks_template()
                self.test_create_ordered_neighbors()

                # do it all again with another order
                block_observations_2 = copy.copy(test_input._block_observations)
                # reorder blocks
                test_input.reorder(new_ordering_strategy(new_order[1]))

                # ensure new blocks are ordered correctly and rerun all other tests
                self.assertTrue(
                    all(
                        [
                            torch.equal(test_input._block_observations[i], block_observations_2[j])
                            for i, j in zip(range(expected["n_blocks"]), new_order[1])
                        ]
                    )
                )
                self.test_blocks_template()
                self.test_create_ordered_neighbors()


# class TestKMeansBlocker(TestBaseBlocker):
#    ...


# class TestVoronoiBlocker(TestBaseBlocker):
#    ...

data = torch.tensor(
    [
        [x1, x2]
        for x1, x2 in zip(
            torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([5.0])).rsample(torch.tensor([100])),
            torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([5.0])).rsample(torch.tensor([100])),
        )
    ]
)
distance_metric = DistanceMetrics.euclidean_distance()

test_names = [
    "10 blocks, 0 neighbors",
    "10 blocks, 3 neighbors",
    "10 blocks, 9 neighbors",
    "100 blocks, 0 neighbors",
    "100 blocks, 20 neighbors",
    "100 blocks, 99 neighbors",
]

test_inputs = [
    KMeansBlocker(data, n_blocks=10, n_neighbors=0, distance_metric=distance_metric),
    KMeansBlocker(data, n_blocks=10, n_neighbors=3, distance_metric=distance_metric),
    KMeansBlocker(data, n_blocks=10, n_neighbors=9, distance_metric=distance_metric),
    KMeansBlocker(data, n_blocks=100, n_neighbors=0, distance_metric=distance_metric),
    KMeansBlocker(data, n_blocks=100, n_neighbors=20, distance_metric=distance_metric),
    KMeansBlocker(data, n_blocks=100, n_neighbors=99, distance_metric=distance_metric),
]

new_orders = [
    (torch.LongTensor([2, 5, 7, 4, 9, 0, 1, 6, 8, 3]), torch.LongTensor([8, 2, 3, 1, 0, 7, 6, 9, 5, 4])),
    (torch.LongTensor([4, 1, 7, 5, 3, 9, 0, 8, 6, 2]), torch.LongTensor([3, 9, 4, 2, 7, 8, 6, 0, 5, 1])),
    (torch.LongTensor([4, 0, 8, 5, 9, 1, 6, 3, 7, 2]), torch.LongTensor([9, 7, 2, 4, 3, 8, 6, 5, 0, 1])),
    (
        torch.LongTensor(
            [
                22,
                9,
                59,
                93,
                54,
                73,
                98,
                8,
                71,
                20,
                23,
                94,
                42,
                41,
                82,
                79,
                33,
                36,
                95,
                87,
                83,
                51,
                84,
                53,
                39,
                26,
                63,
                17,
                2,
                88,
                58,
                50,
                0,
                68,
                24,
                45,
                74,
                89,
                29,
                62,
                15,
                85,
                21,
                57,
                4,
                27,
                96,
                46,
                37,
                25,
                28,
                70,
                7,
                34,
                18,
                16,
                78,
                30,
                77,
                32,
                66,
                86,
                48,
                5,
                69,
                91,
                76,
                14,
                72,
                97,
                38,
                49,
                44,
                90,
                19,
                31,
                47,
                11,
                56,
                64,
                99,
                75,
                1,
                80,
                65,
                6,
                35,
                61,
                10,
                12,
                67,
                92,
                3,
                43,
                13,
                60,
                40,
                81,
                55,
                52,
            ]
        ),
        torch.LongTensor(
            [
                1,
                60,
                64,
                31,
                20,
                24,
                99,
                76,
                86,
                75,
                74,
                11,
                17,
                68,
                95,
                87,
                26,
                91,
                89,
                36,
                6,
                18,
                51,
                97,
                25,
                80,
                10,
                78,
                66,
                28,
                22,
                49,
                54,
                53,
                38,
                65,
                8,
                84,
                98,
                55,
                58,
                39,
                46,
                71,
                90,
                93,
                82,
                23,
                40,
                94,
                0,
                42,
                44,
                29,
                50,
                52,
                27,
                59,
                79,
                48,
                2,
                9,
                61,
                81,
                96,
                69,
                56,
                88,
                34,
                73,
                45,
                5,
                72,
                62,
                33,
                41,
                4,
                70,
                16,
                3,
                37,
                57,
                12,
                85,
                47,
                67,
                32,
                30,
                15,
                43,
                77,
                35,
                13,
                92,
                14,
                19,
                7,
                83,
                21,
                63,
            ]
        ),
    ),
    (
        torch.LongTensor(
            [
                92,
                4,
                49,
                25,
                93,
                58,
                96,
                65,
                66,
                61,
                55,
                42,
                21,
                27,
                38,
                57,
                67,
                84,
                79,
                2,
                73,
                35,
                71,
                59,
                6,
                45,
                19,
                63,
                14,
                75,
                64,
                80,
                83,
                77,
                99,
                40,
                1,
                85,
                34,
                5,
                31,
                91,
                47,
                17,
                10,
                22,
                11,
                32,
                86,
                54,
                74,
                33,
                39,
                3,
                69,
                18,
                16,
                68,
                70,
                20,
                0,
                76,
                94,
                51,
                53,
                50,
                37,
                26,
                46,
                8,
                9,
                36,
                95,
                82,
                97,
                72,
                87,
                12,
                62,
                60,
                56,
                7,
                52,
                88,
                30,
                90,
                28,
                24,
                41,
                98,
                13,
                89,
                23,
                44,
                81,
                29,
                15,
                48,
                78,
                43,
            ]
        ),
        torch.LongTensor(
            [
                27,
                3,
                33,
                85,
                37,
                97,
                78,
                6,
                92,
                99,
                89,
                31,
                0,
                62,
                60,
                70,
                46,
                69,
                56,
                7,
                34,
                74,
                42,
                32,
                1,
                4,
                57,
                59,
                47,
                81,
                55,
                61,
                36,
                67,
                68,
                80,
                39,
                66,
                82,
                18,
                15,
                38,
                29,
                23,
                53,
                10,
                73,
                5,
                48,
                12,
                40,
                41,
                26,
                45,
                58,
                75,
                65,
                14,
                9,
                87,
                43,
                24,
                16,
                86,
                51,
                79,
                11,
                93,
                30,
                20,
                84,
                91,
                49,
                77,
                72,
                83,
                98,
                54,
                25,
                8,
                90,
                52,
                17,
                94,
                96,
                22,
                2,
                19,
                76,
                21,
                64,
                35,
                44,
                50,
                63,
                95,
                71,
                88,
                13,
                28,
            ]
        ),
    ),
    (
        torch.LongTensor(
            [
                71,
                28,
                2,
                16,
                31,
                34,
                65,
                12,
                35,
                45,
                55,
                18,
                67,
                41,
                97,
                79,
                13,
                23,
                96,
                38,
                68,
                17,
                62,
                48,
                49,
                77,
                88,
                26,
                42,
                61,
                76,
                14,
                44,
                37,
                87,
                22,
                10,
                60,
                32,
                74,
                29,
                95,
                11,
                83,
                59,
                1,
                91,
                93,
                92,
                63,
                84,
                24,
                8,
                52,
                58,
                56,
                69,
                90,
                0,
                89,
                85,
                30,
                73,
                51,
                54,
                98,
                99,
                25,
                15,
                53,
                86,
                4,
                78,
                64,
                7,
                6,
                75,
                72,
                5,
                3,
                70,
                50,
                81,
                40,
                9,
                33,
                21,
                19,
                43,
                39,
                80,
                46,
                47,
                20,
                27,
                66,
                82,
                36,
                94,
                57,
            ]
        ),
        torch.LongTensor(
            [
                27,
                25,
                98,
                36,
                60,
                1,
                35,
                22,
                79,
                77,
                53,
                24,
                32,
                16,
                58,
                55,
                92,
                94,
                62,
                38,
                91,
                52,
                74,
                2,
                76,
                90,
                57,
                23,
                34,
                78,
                44,
                99,
                37,
                0,
                39,
                42,
                65,
                11,
                87,
                54,
                75,
                17,
                14,
                86,
                59,
                18,
                93,
                45,
                7,
                97,
                73,
                19,
                72,
                50,
                84,
                43,
                21,
                85,
                56,
                64,
                26,
                28,
                33,
                31,
                80,
                6,
                49,
                96,
                81,
                5,
                70,
                13,
                51,
                46,
                3,
                29,
                47,
                88,
                9,
                10,
                95,
                82,
                48,
                20,
                15,
                61,
                67,
                30,
                66,
                89,
                63,
                69,
                4,
                83,
                8,
                40,
                71,
                41,
                12,
                68,
            ]
        ),
    ),
]

expected_outputs = [
    {"n_blocks": 10, "n_neighbors": 0},
    {"n_blocks": 10, "n_neighbors": 3},
    {"n_blocks": 10, "n_neighbors": 9},
    {"n_blocks": 100, "n_neighbors": 0},
    {"n_blocks": 100, "n_neighbors": 20},
    {"n_blocks": 100, "n_neighbors": 99},
]
