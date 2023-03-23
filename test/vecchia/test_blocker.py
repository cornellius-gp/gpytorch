import abc
import copy

import torch


class TestBaseBlocker(object):
    """
    This base testing class for every blocker object provides some basic sanity checks on the parent class that all
    blockers must inherit from. It includes abstract methods for testing child-specific behavior that child test cases
    must implement.
    """

    test_names = None
    test_inputs = None
    expected_outputs = None
    new_orders = None

    def test_blocks_template(self):

        for test_name, test_input, expected in zip(self.test_names, self.test_inputs, self.expected_outputs):
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

        for test_name, test_input, expected in zip(self.test_names, self.test_inputs, self.expected_outputs):
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

    def test_parent_reorder(self):
        def new_ordering_strategy(order):
            return lambda x: order

        for test_name, test_input, expected, new_order in zip(
            self.test_names, self.test_inputs, self.expected_outputs, self.new_orders
        ):
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

                # finally, create an ordering based on composing previous two to get us back to where we started
                final_order = torch.argsort(torch.stack([new_order[0][i] for i in new_order[1]]))
                # reorder blocks
                test_input.reorder(new_ordering_strategy(final_order))
                # ensure new blocks are ordered correctly and rerun all other tests
                self.assertTrue(
                    all(
                        [
                            torch.equal(test_input._block_observations[i], block_observations_1[i])
                            for i in range(len(final_order))
                        ]
                    )
                )
                self.test_blocks_template()
                self.test_create_ordered_neighbors()

    @abc.abstractmethod
    def test_set_blocks(self):
        ...

    @abc.abstractmethod
    def test_set_neighbors(self):
        ...

    @abc.abstractmethod
    def test_set_test_blocks(self):
        ...

    @abc.abstractmethod
    def test_reorder(self):
        ...
