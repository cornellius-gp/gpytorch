#!/usr/bin/env python3

import torch
import abc

from typing import List

import math
import numpy as np
from matplotlib import pyplot as plt


class BaseBlocker(abc.ABC):
    """
    Provides a base interface for blocking data and establishing neighbor relationships.
    Cannot be directly instantiated and must be subclassed before use.

    Subclasses must implement the set_blocks, set_neighbors, and set_test_blocks methods. Use help() to learn more
    about what these methods must return.

    :param set_blocks_kwargs: Dict of keyword arguments to be passed to child's set_blocks implementation.
    :param set_neighbors_kwargs: Dict of keyword arguments to be passed to child's set_neighbors implementation.
    """
    def __init__(self, set_blocks_kwargs: dict = None, set_neighbors_kwargs: dict = None):

        self._block_observations = None
        self._neighboring_blocks = None
        self._exclusive_neighboring_observations = None
        self._inclusive_neighboring_observations = None
        self._test_block_observations = None

        self.set_blocks_kwargs = set_blocks_kwargs
        self.set_neighbors_kwargs = set_neighbors_kwargs

        self._blocks_template()

    def _blocks_template(self):
        """
        Template that allows children to specify block membership and neighboring structure, then uses that information
        to compute all remaining dependent quantities.
        """
        if self.set_blocks_kwargs is None:
            self._block_observations = self.set_blocks()
        else:
            self._block_observations = self.set_blocks(**self.set_blocks_kwargs)

        self._create_ordered_neighbors(self.set_neighbors_kwargs)

    def _create_ordered_neighbors(self, set_neighbors_kwargs: dict):
        """
        Calculates neighboring relationships based on the order defined by self._block_observations, using the algorithm
        defined by self.set_neighbors(). This is the meat of the template method defined above. Since the results of
        these calculations implicitly depend on the order of self._block_observations, we wrap these steps in a separate
        function, so we can recalculate if the order of self._block_observations changes via a call to self.reorder().

        :param set_neighbors_kwargs: Dict of keyword arguments to be passed to child's set_neighbors implementation.
        """
        if set_neighbors_kwargs is None:
            self._neighboring_blocks = self.set_neighbors()
        else:
            self._neighboring_blocks = self.set_neighbors(**set_neighbors_kwargs)

        exclusive_neighboring_observations = []
        inclusive_neighboring_observations = []

        for i in range(0, len(self._neighboring_blocks)):
            if len(self._neighboring_blocks[i]) == 0:
                exclusive_neighboring_observations.append(torch.tensor([]))
                inclusive_neighboring_observations.append(self._block_observations[i])
            else:
                exclusive_neighboring_observations.append(
                    torch.cat([self._block_observations[block] for block in self._neighboring_blocks[i]]))
                inclusive_neighboring_observations.append(
                    torch.cat([self._block_observations[i], exclusive_neighboring_observations[i]]))

        self._exclusive_neighboring_observations = exclusive_neighboring_observations
        self._inclusive_neighboring_observations = inclusive_neighboring_observations

    @abc.abstractmethod
    def set_blocks(self, **kwargs) -> List[torch.LongTensor]:
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the training set that belong to the ith block.

        :param kwargs: Keyword arguments to be passed to child's set_blocks implementation.
        """
        ...

    @abc.abstractmethod
    def set_neighbors(self, **kwargs) -> List[torch.LongTensor]:
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the blocks that neighbor the ith block. Importantly, the ordering structure of the blocks is
        defined here, and cannot be modified after the object is instantiated.

        :param kwargs: Keyword arguments to be passed to child's set_neighbors implementation.
        """
        ...

    @abc.abstractmethod
    def set_test_blocks(self, **kwargs) -> List[torch.LongTensor]:
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the testing set that belong to the ith block.

        :param kwargs: Keyword arguments to be passed to child's set_test_blocks implementation.
        """
        ...

    @property
    def blocks(self):
        """
        List of tensors where the ith element contains the indices of the training set points belonging to block i.
        """
        return self._block_observations

    @property
    def neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the neighbor
        set of block i.
        """
        return self._exclusive_neighboring_observations

    @property
    def test_blocks(self):
        """
        List of tensors where the ith element contains the indices of the testing set points belonging to block i.
        Only defined after set_test_blocks has been called.
        """
        if self._test_block_observations is None:
            raise RuntimeError(
                "Blocks of testing data do not exist, as the 'block_new_data' "
                "method has not been called on testing data."
            )
        return self._test_block_observations

    @property
    def test_neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the
        neighbor set of the ith test block, where the blocks are ordered by self.block_order. Importantly, the neighbor
        sets of test blocks only consist of training points. Only defined after set_test_blocks has been called.
        """
        if self._test_block_observations is None:
            raise RuntimeError(
                "Neighboring sets of testing blocks do not exist, as the 'set_test_blocks' "
                "method has not been called on testing data."
            )
        return self._inclusive_neighboring_observations

    def reorder(self, new_order: torch.LongTensor):
        """
        Reorders self._block_observations to the order specified by new_order. The ordered neighbors are recalculated,
        and all the relevant lists are modified in place.

        :param new_order: Tensor where the ith element contains the index of the block to be moved to index i.
        """
        # blocks get reordered here directly
        self._block_observations = [self._block_observations[idx] for idx in new_order]
        # neighboring blocks get recomputed here based on new ordering, so we do not have to explicitly reorder them
        self._create_ordered_neighbors(self.set_neighbors_kwargs)

    def block_new_data(self, **kwargs):
        """
        Calls the set_test_blocks method defined by the child class.
        """
        self._test_block_observations = self.set_test_blocks(**kwargs)

    def plot(self, x: torch.tensor, y: torch.tensor, n_blocks: int = None, seed: int = 0):
        """
        Useful visualization for this object and the ordering of the blocks, only implemented for 2D features.

        :param x: Spatial coordinates to plot. This must be the same tensor that was used to construct the blocks.
        :param y: Response values corresponding to each spatial coordinate in x.
        :param n_blocks: Number of blocks to sample for the plot.
        :param seed: RNG seed to change which blocks get randomly sampled.
        """

        np.random.seed(seed)

        if n_blocks is not None:
            if n_blocks > len(self._block_observations):
                raise ValueError("Number of blocks to plot must not exceed total number of blocks.")
        else:
            n_blocks = math.ceil(math.log(len(self._block_observations)))

        ordered_x = torch.cat([x[self._block_observations[i], :] for i in range(len(self._block_observations))]).numpy()
        ordered_y = torch.cat([y[self._block_observations[i]] for i in range(len(self._block_observations))]).numpy()

        unique_colors = np.linspace(0, 1, len(self._block_observations))
        colors = np.concatenate(
            [[(unique_colors[j], 0, unique_colors[j])
              for _ in range(len(self._block_observations[j]))]
              for j in range(len(self._block_observations))])

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(ordered_x[:, 0], ordered_x[:, 1], c=colors)
        plt.title("Ordered Features")

        plt.subplot(2, 2, 2)
        plt.scatter(ordered_x[:, 0], ordered_x[:, 1], c=ordered_y)
        plt.title("Response Values")

        # for a fixed sample of indices, this will always yield the same sampled_blocks regardless of reordering
        invariant_block_idx = torch.argsort(torch.stack([torch.max(block) for block in self._block_observations]))
        sampled_blocks = invariant_block_idx[np.random.permutation(len(self._block_observations))[:n_blocks]]

        plt.subplot(2, 2, 3)
        plt.scatter(ordered_x[:, 0], ordered_x[:, 1], c="grey", alpha=0.25)
        for sampled_block in sampled_blocks:
            plt.scatter(x[self._block_observations[sampled_block], 0].numpy(),
                        x[self._block_observations[sampled_block], 1].numpy(), c="red", s=50)
            plt.scatter(x[self._exclusive_neighboring_observations[sampled_block], 0].numpy(),
                        x[self._exclusive_neighboring_observations[sampled_block], 1].numpy(), c="red", alpha=0.25)
        plt.title("Ordered Neighbors")

        plt.subplot(2, 2, 4)
        plt.scatter(ordered_x[:, 0], ordered_x[:, 1], c="grey", alpha=0.25)
        for sampled_block in sampled_blocks:
            plt.scatter(x[self._block_observations[sampled_block], 0].numpy(),
                        x[self._block_observations[sampled_block], 1].numpy(),
                        c=y[self._block_observations[sampled_block]].numpy(),
                        vmin=torch.min(y), vmax=torch.max(y))
            plt.scatter(x[self._exclusive_neighboring_observations[sampled_block], 0].numpy(),
                        x[self._exclusive_neighboring_observations[sampled_block], 1].numpy(),
                        c=y[self._exclusive_neighboring_observations[sampled_block]].numpy(),
                        vmin=torch.min(y), vmax=torch.max(y))
        plt.title("Corresponding Response Values")
