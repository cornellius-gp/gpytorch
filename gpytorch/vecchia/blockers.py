#!/usr/bin/env python3

import torch
import faiss
import numpy as np

from .abstract_blocker import AbstractBlocker


class KMeansBlocker(AbstractBlocker):

    def __init__(self, data, n_blocks, n_neighbors):

        self.k_means = None
        self.centroids = None
        self.ordering = None

        super(KMeansBlocker, self).__init__(set_blocks_kwargs={"data": data, "n_blocks": n_blocks},
                                            set_neighbors_kwargs={"n_blocks": n_blocks, "n_neighbors": n_neighbors})

    def set_blocks(self, data, n_blocks):

        # create and train faiss k-means object
        kmeans = faiss.Kmeans(data.shape[1], n_blocks, niter=10)
        kmeans.train(np.array(data.float()))

        # store kmeans for finding block membership of test points
        self.k_means = kmeans
        # k-means gives centroids directly, so save centroids
        self.centroids = torch.tensor(kmeans.centroids)

        # create ordering with l2 norm of the centroids
        self.ordering = torch.argsort(torch.linalg.norm(self.centroids, axis=1))

        # get list of len(data) where the ith element indicates which block the ith element of data belongs to
        block_membership = kmeans.index.search(np.array(data.float()), 1)[1].squeeze()

        # create array where the ith element contains the set of indices of data points corresponding to the ith block
        blocking_indices = [[] for _ in range(n_blocks)]
        argsorted = block_membership.argsort()
        for i in range(0, len(block_membership)):
            blocking_indices[block_membership[argsorted[i]]].append(argsorted[i])

        # convert each block to a tensor and return in the order specified by self.ordering
        blocks = [torch.tensor(blocking_index) for blocking_index in blocking_indices]
        return [blocks[idx] for idx in self.ordering]

    def set_neighbors(self, n_blocks, n_neighbors):

        if n_neighbors == 0:
            return [torch.tensor([]) for _ in range(0, n_blocks)]

        else:
            # euclidean distance matrix
            dist_matrix = torch.cdist(self.centroids[self.ordering], self.centroids[self.ordering])
            # sort by distances
            sorter = dist_matrix.argsort()

            return [sorter[i][sorter[i] < i][0:n_neighbors] for i in range(0, len(sorter))]

    def set_test_blocks(self, new_data, n_blocks):

        # get list of len(data) where the ith element indicates which block the ith element of data belongs to
        block_membership = self.k_means.index.search(np.array(new_data.float()), 1)[1].squeeze()
        # create array where the ith element contains the set of indices of data corresponding to the ith block
        blocking_indices = [[] for _ in range(n_blocks)]

        argsorted = block_membership.argsort()
        for i in range(0, len(block_membership)):
            blocking_indices[block_membership[argsorted[i]]].append(argsorted[i])

        # convert each block to a tensor and return in the order specified by self.ordering
        test_blocks = [torch.tensor(blocking_index) for blocking_index in blocking_indices]
        return [test_blocks[idx] for idx in self.ordering]
