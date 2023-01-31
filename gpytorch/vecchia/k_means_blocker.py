#!/usr/bin/env python3

import faiss
import torch
import numpy as np
from typing import List

from ._blocker import BaseBlocker


class KMeansBlocker(BaseBlocker):
    """
    This blocker performs K-Means clustering on a given feature set, computes neighboring blocks, enables
    evaluating block membership for test points, and enables reordering of the blocks based on block centroids.
`
    @param data: Features to cluster via K-Means, typically an n x 2 tensor of spatial lat-long coordinates.
    @param n_blocks: Number of desired clusters. Note that this does not guarantee similarly-sized clusters.
    @param n_neighbors: Number of neighboring clusters per cluster.
    """

    def __init__(self, data: torch.tensor, n_blocks: int, n_neighbors: int):
        self.n_blocks = n_blocks
        self.n_neighbors = n_neighbors
        self.centroids = None

        # this call executes set_blocks and set_neighbors, then superclass computes all dependent quantities
        super(KMeansBlocker, self).__init__(set_blocks_kwargs={"data": data}, set_neighbors_kwargs={})

    def _get_cluster_membership(self, data: torch.tensor) -> List[torch.LongTensor]:
        # tensor of len(new_data) where the element i is the index of the block that element i of new_data belongs to
        block_membership = torch.cdist(data, self.centroids).argsort()[:, 0]

        # create array where the ith element contains the set of indices of new_data belonging to the ith block
        blocking_indices = [[] for _ in range(self.n_blocks)]
        argsorted = block_membership.argsort()
        for i in range(0, len(block_membership)):
            blocking_indices[block_membership[argsorted[i]]].append(argsorted[i])

        # convert each block to n-d tensor and return
        blocks = [torch.LongTensor(blocking_index) for blocking_index in blocking_indices]
        return blocks

    def set_blocks(self, data: torch.tensor) -> List[torch.LongTensor]:
        # create and train faiss k-means object
        kmeans = faiss.Kmeans(data.shape[1], self.n_blocks, niter=10)
        kmeans.train(np.array(data.float()))

        # k-means gives centroids directly, so save centroids
        self.centroids = torch.tensor(kmeans.centroids)

        # determine indices of data points that belong to each cluster block and return
        return self._get_cluster_membership(data)

    def set_neighbors(self, **kwargs) -> List[torch.LongTensor]:
        # if there are no neighbors, we want a list of empty tensors
        if self.n_neighbors == 0:
            return [torch.LongTensor([]) for _ in range(0, self.n_blocks)]

        else:
            # get distance matrix and find ordered distances
            sorter = torch.cdist(self.centroids, self.centroids).argsort().long()
            return [sorter[i][sorter[i] < i][0:self.n_neighbors] for i in range(0, len(sorter))]

    def set_test_blocks(self, new_data: torch.tensor) -> List[torch.LongTensor]:
        # determine indices of new data points that belong to each cluster block and return
        return self._get_cluster_membership(new_data)

    def reorder(self, ordering_strategy):
        new_order = ordering_strategy(self.centroids)
        self.centroids = self.centroids[new_order]
        super().reorder(new_order)
