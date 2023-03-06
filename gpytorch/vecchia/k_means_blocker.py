#!/usr/bin/env python3

import faiss
import torch
import numpy as np
from typing import List

from ._blocker import BaseBlocker
from .distance_metrics import AbstractDistanceMetric


class KMeansBlocker(BaseBlocker):
    """
    This blocker performs K-Means clustering on a given feature set, computes neighboring blocks, enables
    evaluating block membership for test points, and enables reordering of the blocks based on block centroids.
`
    @param data: Features to cluster via K-Means, typically an n x 2 tensor of spatial lat-long coordinates.
    @param n_blocks: Number of desired clusters. Note that this does not guarantee similarly-sized clusters.
    @param n_neighbors: Number of neighboring clusters per cluster.
    """

    def __init__(self, data: torch.tensor, n_blocks: int, n_neighbors: int, distance_metric: AbstractDistanceMetric):
        self.n_blocks = n_blocks
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.centroids = None

        # this call executes set_blocks and set_neighbors, then superclass computes all dependent quantities
        super(KMeansBlocker, self).__init__(set_blocks_kwargs={"data": data}, set_neighbors_kwargs={})

    def _get_cluster_membership(self, data: torch.tensor) -> List[torch.LongTensor]:
        """
        Determines which K-Means cluster each point in the provided data belongs to.

        @param data: Tensor for which to evaluate cluster membership. If any of these points are outside the domain
            of the points used to train the K-Means clusters, you may get nonsensical results.

        @return: List of tensors, where the ith tensor contains the indices of the points in data that belong to the
            ith K-Means cluster.
        """
        blocks = []
        block_per_point = torch.cdist(data, self.centroids).argsort()[:, 0]

        for block in range(len(self.centroids)):
            these_members = (block_per_point == block).nonzero().squeeze()
            blocks.append(these_members)

        return blocks

    def set_blocks(self, data: torch.tensor) -> List[torch.LongTensor]:
        # create and train faiss k-means object
        kmeans = faiss.Kmeans(data.shape[1], self.n_blocks, niter=10)
        kmeans.train(np.array(data.float()))

        # k-means gives centroids directly, so save centroids
        self.centroids = torch.tensor(kmeans.centroids)

        # determine indices of data points that belong to each cluster block and return
        return self._get_cluster_membership(data)

    def set_neighbors(self) -> List[torch.LongTensor]:
        # if there are no neighbors, we want a list of empty tensors
        if self.n_neighbors == 0:
            return [torch.LongTensor([]) for _ in range(0, self.n_blocks)]

        else:
            # get distance matrix and find ordered distances
            sorter = self.distance_metric(self.centroids, self.centroids).argsort().long()
            return [sorter[i][sorter[i] < i][0:self.n_neighbors] for i in range(0, len(sorter))]

    def set_test_blocks(self, new_data: torch.tensor) -> List[torch.LongTensor]:
        # determine indices of new data points that belong to each cluster block and return
        return self._get_cluster_membership(new_data)

    def reorder(self, ordering_strategy):
        # new order is defined as some reordering of the K-means block centroids
        new_order = ordering_strategy(self.centroids)

        # reorder the instance attributes that depend on the ordering
        self.centroids = self.centroids[new_order]

        # reorder superclass attributes and recompute neighbors under new ordering
        super()._reorder(new_order)
