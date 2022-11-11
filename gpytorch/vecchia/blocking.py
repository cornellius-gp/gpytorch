#!/usr/bin/env python3

import torch
import faiss
import numpy as np

import abc


class AbstractBlocks(abc.ABC):
    """
    Provides a base interface for blocking data, establishing neighbor relationships, and reordering blocks.
    Cannot be directly instantiated and must be subclassed before use.

    Subclasses must implement the set_blocks method, which returns a list of length equal to the number of blocks,
    where the ith element is a tensor containing the indices of the training set that belong to the ith block.

    Subclasses must also implement the set_neighbors method, which returns a list of length equal to the number of
    blocks, where the ith element is a tensor containing the indices of the blocks that neighbor the ith block.
    """
    def __init__(self):
        # properties of the blocks of data
        self._block_observations = None

        # properties of the neighbors of blocks
        self._neighboring_blocks = None
        self._exclusive_neighboring_observations = None
        self._inclusive_neighboring_observations = None

        # use template to define block_observations and neighboring_blocks, then compute remaining dependent quantities
        self._blocks_template()

    def _blocks_template(self):
        self._block_observations = self.set_blocks()
        self._neighboring_blocks = self.set_neighbors()

        self._exclusive_neighboring_observations = [torch.tensor([]), *[torch.cat([self._block_observations[block]
                                                    for block in self._neighboring_blocks[i]])
                                                    for i in range(1, len(self._neighboring_blocks))]]

        self._inclusive_neighboring_observations = [self._block_observations[0],
                                                    *[torch.cat(self._block_observations[i],
                                                                self._exclusive_neighboring_observations[i])
                                                    for i in range(1, len(self._neighboring_blocks))]]

    @abc.abstractmethod
    def set_blocks(self):
        ...

    @abc.abstractmethod
    def set_neighbors(self):
        ...

    @property
    def block_order(self):
        """Tensor containing the current order of the blocks, relative to their original ordering. """
        raise NotImplementedError

    @property
    def centroids(self):
        """Tensor containing the centroids of each block, returned in the order given by self.block_order. """
        raise NotImplementedError

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
        where the blocks are ordered by self.block_order. Only defined after block_new_data has been called.
        """
        raise NotImplementedError

    @property
    def test_neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the
        neighbor set of the ith test block, where the blocks are ordered by self.block_order. Importantly, the neighbor
        sets of test blocks only consist of training points. Only defined after block_new_data has been called.
        """
        raise NotImplementedError


# TODO: complete this reimplementation of the Block class in terms of AbstractBlocks.
class Blocks(AbstractBlocks):
    def __init__(self, data, n_blocks, n_neighbors):
        super().__init__()
        self.data = data
        self.n_blocks = n_blocks
        self.n_neighbors = n_neighbors

    def set_blocks(self):
        test_blocks = self.n_blocks
        return test_blocks

    def set_neighbors(self):
        test_neighbors = self.n_neighbors
        return test_neighbors


# TODO: Reimplement this in terms of the above classes.
class Block:
    """
    Groups datasets into spatial blocks, determines which blocks are neighbors, and enables reordering of the blocks,
    as Vecchia's approximation depends on the order of the conditioning sets. Once a dataset has been blocked, this
    class groups new testing datasets into blocks based on the training data.

    :param data:
    :param n_blocks:
    :param n_neighbors:
    """

    def __init__(self, data, n_blocks, n_neighbors):

        self._n_neighbors = n_neighbors
        self._n_blocks = n_blocks

        # original block order by index, is constant and represents whatever ordering kmeans imposes on our blocks
        self._original_block_order = None
        # block order by index, this gets updated when an "order" method is called
        self._current_block_order = None
        # keeps track of whether the blocks have been reordered from their original order
        self._reordered = False

        # object to save FAISS kmeans object for getting block memnbership of new test points
        self._k_means = None
        # numeric values of block centroids after training
        self._block_centroids = None

        # list of length n_blocks, where the ith entry contains the indices of training data that belong to block i
        self._train_blocks = None
        # list of length n_blocks, where the ith entry contains the indices of testing data that belong to block i
        self._test_blocks = None

        # boolean matrix indicating whether block i is a neighbor of block j
        self._is_neighbors = None
        # list of length n_blocks, where the ith element contains the indices of blocks that neighbor block i
        self._neighbor_block_idx = None
        # list of length n_blocks, where the ith element contains the indices of training data that neighbor block i
        self._neighbor_block_obs = None
        # list of length n_blocks, where the ith element contains the indices of testing data that neighbor block i
        self._test_neighbor_block_obs = None

        self._block(data, n_blocks)
        self._create_neighbors(n_neighbors)

    @property
    def block_order(self):
        """Tensor containing the current order of the blocks, relative to their original ordering after kmeans. """
        return self._current_block_order

    @property
    def centroids(self):
        """Tensor containing the centroids of each block, returned in the order given by self.block_order. """
        if self._reordered:
            return self._block_centroids[self._current_block_order]
        else:
            return self._block_centroids

    @property
    def blocks(self):
        """
        List of tensors where the ith element contains the indices of the training set points belonging to the
        ith block, where the blocks are ordered by self.block_order.
        """
        if self._reordered:
            return [self._train_blocks[i] for i in self._current_block_order]
        else:
            return self._train_blocks

    @property
    def neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the neighbor
        set of the ith block, where the blocks are ordered by self.block_order.
        """
        return self._neighbor_block_obs

    @property
    def test_blocks(self):
        """
        List of tensors where the ith element contains the indices of the testing set points belonging to the ith block,
        where the blocks are ordered by self.block_order. Only defined after block_new_data has been called.
        """
        if self._test_blocks is None:
            raise RuntimeError(
                "Blocks of testing data do not exist, as the 'block_new_data' "
                "method has not been called on testing data."
            )
        if self._reordered:
            return [self._test_blocks[i] for i in self._current_block_order]
        else:
            return self._test_blocks

    @property
    def test_neighbors(self):
        """
        List of tensors, where the ith element contains the indices of the training set points belonging to the
        neighbor set of the ith test block, where the blocks are ordered by self.block_order. Importantly, the neighbor
        sets of test blocks only consist of training points. Only defined after block_new_data has been called.
        """
        if self._test_blocks is None:
            raise RuntimeError(
                "Neighboring sets of testing blocks do not exist, as the 'block_new_data' "
                "method has not been called on testing data."
            )
        return self._test_neighbor_block_obs

    @property
    def block_adj_mat(self):
        """
        Tensor of the adjacency matrix indicating block neighbor relationships,
        where the blocks are ordered by self.block_order
        """
        return self._is_neighbors

    def _block(self, data, n_blocks):
        # use FAISS k-means to block data
        kmeans = faiss.Kmeans(data.shape[1], n_blocks, niter=10)
        kmeans.train(np.array(data.float()))

        # store kmeans for finding block membership of test points
        self._k_means = kmeans
        # k-means gives centroids directly, so save centroids
        self._block_centroids = torch.tensor(kmeans.centroids)
        # create vectors of order of blocks, one is constant for reference, one represents new orderings of blocks
        self._original_block_order = torch.tensor(range(0, len(self._block_centroids)))
        self._current_block_order = torch.tensor(range(0, len(self._block_centroids)))

        # get list of len(data) where the ith element indicates which block the ith element of data belongs to
        block_membership = kmeans.index.search(np.array(data.float()), 1)[1].squeeze()
        # create array where the ith element contains the set of indices of data points corresponding to the ith block
        blocking_indices = [[] for _ in range(n_blocks)]
        argsorted = block_membership.argsort()
        for i in range(0, len(block_membership)):
            blocking_indices[block_membership[argsorted[i]]].append(argsorted[i])

        self._train_blocks = [torch.tensor(blocking_index) for blocking_index in blocking_indices]
        self._trained = True

    def _create_neighbors(self, n_neighbors):
        if n_neighbors == 0:
            self._is_neighbors = torch.zeros((self._n_blocks, self._n_blocks))
            self._neighbor_block_idx = [torch.tensor([]) for _ in range(0, self._n_blocks)]
            self._neighbor_block_obs = [torch.tensor([]) for _ in range(0, self._n_blocks)]
            self._test_neighbor_block_obs = self.blocks

        else:
            # euclidean distance matrix
            dist_matrix = torch.cdist(self.centroids, self.centroids)
            # sort by distances
            sorter = dist_matrix.argsort()
            # create empty matrix to indicate neighbor relationship
            neighbor_mask = torch.zeros((self._n_blocks, self._n_blocks))

            for i in range(len(dist_matrix)):
                # this is from the probability chain rule and ensures a valid density function
                if i < n_neighbors + 1:
                    neighbor_mask[0:i, i] = True
                else:
                    neighbor_mask[sorter[i][sorter[i] < i][0:n_neighbors], i] = True

            self._is_neighbors = neighbor_mask.transpose(0, 1)
            self._neighbor_block_idx = [sorter[i][sorter[i] < i][0:n_neighbors] for i in range(0, len(sorter))]
            self._neighbor_block_obs = [torch.tensor([]), *[torch.cat([self.blocks[block]
                                                                       for block in self._neighbor_block_idx[i]])
                                                            for i in range(1, self._n_blocks)]]

            # because only training points are considered neighbors of any future testing data, we can calculate testing
            # neighbors before calling 'block_new_data'
            self._test_neighbor_block_obs = [self.blocks[0],
                                             *[torch.cat([self.blocks[i], self.neighbors[i]])
                                               for i in range(1, self._n_blocks)]]

    def block_new_data(self, new_data):
        # get list of len(data) where the ith element indicates which block the ith element of data belongs to
        block_membership = self._k_means.index.search(np.array(new_data.float()), 1)[1].squeeze()
        # create array where the ith element contains the set of indices of data corresponding to the ith block
        blocking_indices = [[] for _ in range(self._n_blocks)]

        argsorted = block_membership.argsort()
        for i in range(0, len(block_membership)):
            blocking_indices[block_membership[argsorted[i]]].append(argsorted[i])

        self._test_blocks = [torch.tensor(blocking_index) for blocking_index in blocking_indices]

    def reorder(self, new_order):
        self._reordered = True
        # this is where the reordering happens
        self._current_block_order = new_order
        # recompute neighbors
        self._create_neighbors(self._n_neighbors)

    def compute_mean_covar(self, x1, x2, y, mean_module, covar_module, training):
        # create empty lists to hold block means and covariances
        mean_list = []
        cov_list = []

        if training:
            # append mean function applied to first block in first spot
            mean_list.append(mean_module(x1[self.blocks[0]]))
            # append within covariance block to first spot
            cov_list.append(covar_module(x1[self.blocks[0]], x2[self.blocks[0]]))

            if self._n_neighbors == 0:
                # if no neighbors, all blocks are independent, so simply evaluate mean and covariance for each block
                for i in range(1, self._n_blocks):
                    mean_list.append(mean_module(x1[self.blocks[i]]))
                    cov_list.append(covar_module(x1[self.blocks[i]], x2[self.blocks[i]]))

            else:
                for i in range(1, self._n_blocks):
                    # these calculations come from bottom of P7, Quiroz et al, 2021
                    c_within = covar_module(x1[self.blocks[i]], x2[self.blocks[i]])
                    c_between = covar_module(x1[self.blocks[i]], x2[self.neighbors[i]])
                    c_neighbors = covar_module(x1[self.neighbors[i]], x2[self.neighbors[i]])

                    # use cholesky decomposition to compute inverse, may be numerically unstable with large n_neighbors
                    l_inv = c_neighbors.cholesky().inverse()
                    # compute mean
                    b = c_between @ l_inv.t() @ l_inv
                    mean = mean_module(x1[self.blocks[i]]) + \
                           b @ (y[self.neighbors[i]] - mean_module(x2[self.neighbors[i]]))
                    # compute covariance
                    f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

                    mean_list.append(mean)
                    cov_list.append(f)

        else:
            for i in range(0, len(self.blocks)):
                c_within = covar_module(x1[self.test_blocks[i]], x1[self.test_blocks[i]])
                c_between = covar_module(x1[self.test_blocks[i]], x2[self.test_neighbors[i]])
                c_neighbors = covar_module(x2[self.test_neighbors[i]], x2[self.test_neighbors[i]])

                # use cholesky decomposition to compute needed terms, may be numerically unstable with large n_neighbors
                l_inv = c_neighbors.cholesky().inverse()
                # compute mean
                b = c_between @ l_inv.t() @ l_inv
                mean = mean_module(x1[self.test_blocks[i]]) + \
                       b @ (y[self.test_neighbors[i]] - mean_module(x2[self.test_neighbors[i]]))
                # compute covariance
                f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

                mean_list.append(mean)
                cov_list.append(f)

        return mean_list, cov_list
