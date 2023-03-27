#!/usr/bin/env python3

from .distance_metrics import *
from .ordering_strategies import *

from ._index import BaseIndex
from .k_means_index import KMeansIndex
from .voronoi_index import VoronoiIndex

__all__ = [
    "DistanceMetrics",
    "OrderingStrategies",
    "BaseIndex",
    "KMeansIndex",
    "VoronoiIndex",
]


# TODO: Where to put this???
# this function uses an index to compute block mean and covariance for a Vecchia-style GP. Until we have a more
# concrete vecchia module, I do not know where to put this.
def compute_mean_covar(index, x1, x2, y, mean_module, covar_module, training):

    # extract relevant info from index
    n_blocks = len(index.blocks)
    n_neighbors = len(index.neighbors[-1])

    # create empty lists to hold block means and covariances
    mean_list = []
    cov_list = []

    if training:
        # append mean function applied to first block in first spot
        mean_list.append(mean_module(x1[index.blocks[0]]))
        # append within covariance block to first spot
        cov_list.append(covar_module(x1[index.blocks[0]], x2[index.blocks[0]]))

        if n_neighbors == 0:
            # if no neighbors, all index are independent, so simply evaluate mean and covariance for each block
            for i in range(1, n_blocks):
                mean_list.append(mean_module(x1[index.blocks[i]]))
                cov_list.append(covar_module(x1[index.blocks[i]], x2[index.blocks[i]]))

        else:
            for i in range(1, n_blocks):
                # these calculations come from bottom of P7, Quiroz et al, 2021
                c_within = covar_module(x1[index.blocks[i]], x2[index.blocks[i]])
                c_between = covar_module(x1[index.blocks[i]], x2[index.neighbors[i]])
                c_neighbors = covar_module(x1[index.neighbors[i]], x2[index.neighbors[i]])

                # use cholesky decomposition to compute inverse, may be numerically unstable with large n_neighbors
                l_inv = c_neighbors.cholesky().inverse()
                # compute mean
                b = c_between @ l_inv.t() @ l_inv
                mean = mean_module(x1[index.blocks[i]]) + b @ (
                    y[index.neighbors[i]] - mean_module(x2[index.neighbors[i]])
                )
                # compute covariance
                f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

                mean_list.append(mean)
                cov_list.append(f)

    else:
        for i in range(0, len(index.blocks)):
            c_within = covar_module(x1[index.test_blocks[i]], x1[index.test_blocks[i]])
            c_between = covar_module(x1[index.test_blocks[i]], x2[index.test_neighbors[i]])
            c_neighbors = covar_module(x2[index.test_neighbors[i]], x2[index.test_neighbors[i]])

            # use cholesky decomposition to compute needed terms, may be numerically unstable with large n_neighbors
            l_inv = c_neighbors.cholesky().inverse()
            # compute mean
            b = c_between @ l_inv.t() @ l_inv
            mean = mean_module(x1[index.test_blocks[i]]) + b @ (
                y[index.test_neighbors[i]] - mean_module(x2[index.test_neighbors[i]])
            )
            # compute covariance
            f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

            mean_list.append(mean)
            cov_list.append(f)

    return mean_list, cov_list
