#!/usr/bin/env python3

import torch
import abc


class BaseBlocker(abc.ABC):
    """
    Provides a base interface for blocking data and establishing neighbor relationships.
    Cannot be directly instantiated and must be subclassed before use.

    Subclasses must implement the set_blocks, set_neighbors, and set_test_blocks methods. Use help() to learn more
    about what these methods must return.
    """
    def __init__(self, set_blocks_kwargs=None, set_neighbors_kwargs=None):

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

    def _create_ordered_neighbors(self, set_neighbors_kwargs):
        """
        Calculates neighboring relationships based on the order defined by self._block_observations, using the algorithm
        defined by self.set_neighbors(). This is the meat of the template method defined above. Since the results of
        these calculations implicitly depend on the order of self._block_observations, we wrap these steps in a separate
        function, so we can recalculate if the order of self._block_observations changes via a call to self.reorder().
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
    def set_blocks(self, **kwargs):
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the training set that belong to the ith block.
        """
        ...

    @abc.abstractmethod
    def set_neighbors(self, **kwargs):
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the blocks that neighbor the ith block. Importantly, the ordering structure of the blocks is
        defined here, and cannot be modified after the object is instantiated.
        """
        ...

    @abc.abstractmethod
    def set_test_blocks(self, **kwargs):
        """
        Returns a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the testing set that belong to the ith block.
        """
        ...

    @property
    def blocks(self):
        """
        List of tensors where the ith element contains the indices of the training set points belonging to the
        ith block, where the blocks are ordered by self.block_order.
        """
        return self._block_observations

    @property
    def neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the neighbor
        set of the ith block, where the blocks are ordered by self.block_order.
        """
        return self._exclusive_neighboring_observations

    @property
    def test_blocks(self):
        """
        List of tensors where the ith element contains the indices of the testing set points belonging to the ith block,
        where the blocks are ordered by self.block_order. Only defined after set_test_blocks has been called.
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

    # TODO: Document reorder and how it should be implemented
    def reorder(self, new_order):
        self._block_observations = [self._block_observations[idx] for idx in new_order]
        self._create_ordered_neighbors(self.set_neighbors_kwargs)

    def block_new_data(self, **kwargs):
        self._test_block_observations = self.set_test_blocks(**kwargs)
